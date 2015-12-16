#include <time.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "CvxCompress.hxx"
#include "Wavelet_Transform_Fast.hxx"
#include "Wavelet_Transform_Slow.hxx"  // for comparison in module test
#include "Block_Copy.hxx"
#include "Run_Length_Encode_Slow.hxx"  // turns out, it isn't that slow after all
#include "Read_Raw_Volume.hxx"

CvxCompress::CvxCompress()
{
}

CvxCompress::~CvxCompress()
{
}

static int Find_Pow2(int val)
{
	int cnt = -1;
	while (val > 0)
	{
		val = val >> 1;
		++cnt;
	}
	return cnt;
}

bool CvxCompress::_Valid_Block_Size(int bx, int by, int bz)
{
	if (
		((1 << Find_Pow2(bx)) == bx) && 
		((1 << Find_Pow2(by)) == by) &&
		((1 << Find_Pow2(bz)) == bz) &&
		(bx >= Min_BX() && bx <= Max_BX()) &&
		(by >= Min_BY() && by <= Max_BY()) &&
		(bz >= Min_BZ() && bz <= Max_BZ())
	)
	{
		return true;
	}
	else
	{
		return false;
	}
}

static float Compute_Global_RMS(float* vol, int nx, int ny, int nz)
{
	long nn = (long)nx * (long)ny * (long)nz;
	long _mm_nn = nn >> 2;
	long num_threads;
#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
	}
	long* loop_start = new long[num_threads+1];
	loop_start[0] = 0;
	for (long iThr = 0;  iThr < num_threads;  ++iThr) loop_start[iThr+1] = _mm_nn * (iThr+1) / num_threads;
	
	double rms = 0.0;
#pragma omp parallel for reduction(+:rms) schedule(static,1)
	for (long iThr = 0;  iThr < num_threads;  ++iThr)
	{
		__m256d acc = _mm256_setzero_pd();
		for (long i = loop_start[iThr];  i < loop_start[iThr+1];  ++i)
		{
			__m128 _mm_val = _mm_loadu_ps((float*)(((__m128*)vol)+i));
			__m256d val = _mm256_cvtps_pd(_mm_val);
#ifdef __AVX2__
			acc = _mm256_fmadd_pd(val,val,acc);
#else
			acc = _mm256_add_pd(acc,_mm256_mul_pd(val,val));
#endif
		}
		acc = _mm256_hadd_pd(acc,acc);
		__m128d acc0 = _mm256_extractf128_pd(acc,0);
		__m128d acc1 = _mm256_extractf128_pd(acc,1);
		acc0 = _mm_add_pd(acc0,acc1);
		double v[2];
		_mm_store_pd(v,acc0);
		rms += v[0];
	}
	for (long i = loop_start[num_threads]*4;  i < nn;  ++i)
	{
		double dval = (double)vol[i];
		rms += dval * dval;
	}
	rms = sqrt(rms/((double)nx*(double)ny*(double)nz));
	delete [] loop_start;
	return (float)rms;
}

inline void memcpy_avx(void* dst, void* src, int len)
{
	// lead-in
	int lead_in = (int)(32 - (((long)dst) & 31)) & 31;
	//printf("dst = %p, lead_in = %d, len = %d\n",dst,lead_in,len);
	for (int i = 0;  i < lead_in;  ++i) ((char*)dst)[i] = ((char*)src)[i];
	__m256i* dst_avx = (__m256i*)(((char*)dst)+lead_in);  // aligned 32b
	__m256i* src_avx = (__m256i*)(((char*)src)+lead_in);  // possibly misaligned
	len -= lead_in;
	//printf("dst_avx = %p, len = %d\n",dst_avx,len);
	// aligned stream copy
	int n = len>>5;
	for (int i = 0;  i < n;  ++i)
        {
                __m256i v = _mm256_loadu_si256(src_avx+i);
		_mm256_stream_si256(dst_avx+i,v);
	}
	// lead-out
	for (int i = n<<5;  i < len;  ++i) ((char*)dst_avx)[i] = ((char*)src_avx)[i];
}

float CvxCompress::Compress(
	float scale,
	float* vol,
	int nx,
	int ny,
	int nz,
	int bx,
	int by,
	int bz,
	unsigned int* compressed,
	long& compressed_length 
	)
{
	float global_rms = Compute_Global_RMS(vol,nx,ny,nz);
	float scalefac = 1.0f / (scale * global_rms);
	
	int num_threads;
#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
	}
#define MAX(a,b) (a>b?a:b)
	int max_bs = MAX(bx,MAX(by,bz));
#undef MAX
	int priv_blkoff_len = 262144 / (bx*by*bz);
	priv_blkoff_len = priv_blkoff_len > 1 ? priv_blkoff_len : 1;
	int work_blkoff_buffer_size = priv_blkoff_len + 2;
	int work_compress_buffer_size = priv_blkoff_len*bx*by*bz;
	int work_wave_transform_buffer_size = bx*by*bz;
	int work_wave_transform_tmp_buffer_size = max_bs*8;
	int work_size_one_thread = 2*work_blkoff_buffer_size + work_compress_buffer_size + work_wave_transform_buffer_size + work_wave_transform_tmp_buffer_size;
	work_size_one_thread = (((work_size_one_thread + 15 ) >> 4) << 4);  // round to full 64b page
	int work_size = work_size_one_thread * num_threads;
	if (work_size_one_thread != (work_size / num_threads)) {printf("Error! work buffer too large!\n"); exit(-1);}
	float* work;
	posix_memalign((void**)&work, 64, sizeof(float)*work_size);
#pragma omp parallel for schedule(static,1)
	for (int iThread = 0;  iThread < num_threads;  ++iThread)
	{
		int thread_id = omp_get_thread_num();
		int* priv_work = (int*)(work + thread_id * work_size_one_thread);
		for (int i = 0;  i < work_size_one_thread;  ++i) priv_work[i] = 0;
	}

	int nbx = (nx+bx-1)/bx;
	int nby = (ny+by-1)/by;
	int nbz = (nz+bz-1)/bz;
	int nnn = nbx*nby*nbz;
	
	compressed[0] = nx;
	compressed[1] = ny;
	compressed[2] = nz;
	compressed[3] = bx;
	compressed[4] = by;
	compressed[5] = bz;
	
	float mulfac = 1.0f / (global_rms * scale);
	compressed[6] = *((unsigned int*)&mulfac);

	compressed[7] = 0;  // not used

	long* glob_blkoffs = (long*)(compressed+8);  // no need to initialize
	
	unsigned int* bytes = (unsigned int*)(glob_blkoffs+nnn);
	long byte_offset = 0l;

#pragma omp parallel for
	for (long iBlk = 0;  iBlk < nnn;  ++iBlk)
	{
		long iiz = iBlk / (nbx*nby);
		long iix = iBlk - iiz*nbx*nby;
		long iiy = iix / nbx;
		iix = iix - iiy*nbx;

		int x0 = iix*bx;
		int y0 = iiy*by;
		int z0 = iiz*bz;

		//printf("iBlk=%d, x0=%d, y0=%d, z0=%d\n",iBlk,x0,y0,z0);

		int thread_id = omp_get_thread_num();
		int* priv_blkstore_idx = (int*)(work + thread_id * work_size_one_thread);
		int* priv_blkoff = (int*)(priv_blkstore_idx + 1);
		int* priv_iBlk = (int*)(priv_blkstore_idx + work_blkoff_buffer_size);
                unsigned int* priv_compress_buffer = (unsigned int*)(priv_iBlk + work_blkoff_buffer_size);
                float* priv_work = (float*)(priv_compress_buffer + work_compress_buffer_size);
                float* priv_tmp = priv_work + work_wave_transform_buffer_size;

		priv_iBlk[*priv_blkstore_idx] = iBlk;
		int blkoff = priv_blkoff[*priv_blkstore_idx];
		blkoff = blkoff < 0 ? -blkoff : blkoff;
		unsigned long* priv_compressed = (unsigned long*)(((char*)priv_compress_buffer) + blkoff);

		Copy_To_Block(vol,x0,y0,z0,nx,ny,nz,(__m128*)priv_work,bx,by,bz);
		Wavelet_Transform_Fast_Forward((__m256*)priv_work,(__m256*)priv_tmp,bx,by,bz);
		int bytepos = 0, error = 0;
		Run_Length_Encode_Slow(mulfac,priv_work,bx*by*bz,priv_compressed,bytepos,error);
		//Run_Length_Encode_Fast(mulfac,priv_work,bx*by*bz,priv_compressed,bytepos,error);

		++(*priv_blkstore_idx);
		if (error)
		{
			priv_blkoff[*priv_blkstore_idx] = -(blkoff+bx*by*bz);
			// TMJ 12/14/2015 - Deliberate use of mempcy instead of memcpy_avx.
			// We want the copy to end up in the cache, hence we don't use memcpy_avx, which does a straight-to-DRAM stream copy.
			memcpy(priv_compressed,priv_work,sizeof(float)*bx*by*bz);
		}
		else
		{
			priv_blkoff[*priv_blkstore_idx] = blkoff + bytepos;
		}
		if (*priv_blkstore_idx >= priv_blkoff_len)
		{
			// copy compressed blocks from private area to global area.
			int priv_blklen = priv_blkoff[*priv_blkstore_idx];
			priv_blklen = priv_blklen < 0 ? -priv_blklen : priv_blklen;
			//printf("MEMCPY :: iBlk=%d, priv_blkstore_idx=%d, priv_blklen=%d\n",iBlk,*priv_blkstore_idx,priv_blklen);
			char* glob_dst = 0L;
#pragma omp critical
			{
				glob_dst = ((char*)bytes) + byte_offset;
				byte_offset += (long)priv_blklen;
			}
			for (int i = 0;  i < *priv_blkstore_idx;  ++i) 
			{
				int dst_iBlk = priv_iBlk[i];
				int blkoff = priv_blkoff[i];
				bool uncompressed = blkoff < 0 ? true : false;
				blkoff = uncompressed ? -blkoff : blkoff;
				long new_glob_blkoff = (glob_dst + blkoff) - (char*)bytes;
				glob_blkoffs[dst_iBlk] = uncompressed ? -new_glob_blkoff : new_glob_blkoff;
			}
			memcpy_avx(glob_dst,priv_compress_buffer,priv_blklen);
			*priv_blkstore_idx = 0;
			priv_blkoff[0] = 0;
		}
	}
#pragma omp parallel for schedule(static,1)
	for (int iThr = 0;  iThr < num_threads;  ++iThr)
	{
		int thread_id = omp_get_thread_num();
		int* priv_blkstore_idx = (int*)(work + thread_id * work_size_one_thread);
		int* priv_blkoff = (int*)(priv_blkstore_idx + 1);
		int* priv_iBlk = (int*)(priv_blkstore_idx + work_blkoff_buffer_size);
                unsigned int* priv_compress_buffer = (unsigned int*)(priv_iBlk + work_blkoff_buffer_size);
                float* priv_work = (float*)(priv_compress_buffer + work_compress_buffer_size);
                float* priv_tmp = priv_work + work_wave_transform_buffer_size;
		if (*priv_blkstore_idx >= 1)
		{
			// copy compressed blocks from private area to global area.
			int priv_blklen = priv_blkoff[*priv_blkstore_idx];
			priv_blklen = priv_blklen < 0 ? -priv_blklen : priv_blklen;
			//printf("MEMCPY :: priv_blkstore_idx=%d, priv_blklen=%d\n",*priv_blkstore_idx,priv_blklen);
			char* glob_dst = 0L;
#pragma omp critical
			{
				glob_dst = ((char*)bytes) + byte_offset;
				byte_offset += (long)priv_blklen;
			}
			for (int i = 0;  i < *priv_blkstore_idx;  ++i) 
			{
				int dst_iBlk = priv_iBlk[i];
				int blkoff = priv_blkoff[i];
				bool uncompressed = blkoff < 0 ? true : false;
				blkoff = uncompressed ? -blkoff : blkoff;
				long new_glob_blkoff = (glob_dst + blkoff) - (char*)bytes;
				glob_blkoffs[dst_iBlk] = uncompressed ? -new_glob_blkoff : new_glob_blkoff;
			}
			memcpy_avx(glob_dst,priv_compress_buffer,priv_blklen);
			*priv_blkstore_idx = 0;
			priv_blkoff[0] = 0;
		}
	}
	compressed_length = 28 + 8*nnn + byte_offset;

	free(work);
	double ratio = ((double)nx * (double)ny * (double)nz * (double)sizeof(float)) / (double)compressed_length;
	return (float)ratio;
}

void CvxCompress::Decompress(
	float*& vol,
	int& nx,
	int& ny,
	int& nz,
	unsigned int* compressed,
	long compressed_length 
	)
{
	nx = ((int*)compressed)[0];
	ny = ((int*)compressed)[1];
	nz = ((int*)compressed)[2];
	int bx = ((int*)compressed)[3];
	int by = ((int*)compressed)[4];
	int bz = ((int*)compressed)[5];
	float mulfac = ((float*)compressed)[6];
	//printf("nx=%d, ny=%d, nz=%d, bx=%d, by=%d, bz=%d, mulfac=%e\n",nx,ny,nz,bx,by,bz,mulfac);

	posix_memalign((void**)&vol, 64, (long)nx*(long)ny*(long)nz*(long)sizeof(float));
	
	int nbx = (nx+bx-1)/bx;
	int nby = (ny+by-1)/by;
	int nbz = (nz+bz-1)/bz;
	int nnn = nbx*nby*nbz;
	//printf("nbx=%d, nby=%d, nbz=%d, nnn=%d\n",nbx,nby,nbz,nnn);

	long* glob_blkoffs = (long*)(compressed+8);
	
	unsigned int* bytes = (unsigned int*)(glob_blkoffs+nnn);

	int num_threads;
#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
	}
#define MAX(a,b) (a>b?a:b)
	int max_bs = MAX(bx,MAX(by,bz));
#undef MAX
	int work_size_one_thread = ((bx*by*bz) + max_bs*8);
	work_size_one_thread = (((work_size_one_thread + 15 ) >> 4) << 4);  // round to full 64b page
	int work_size = work_size_one_thread * num_threads;
	float* work;
	posix_memalign((void**)&work, 64, sizeof(float)*work_size);

#pragma omp parallel for 
	for (long iBlk = 0;  iBlk < nnn;  ++iBlk)
	{
		long iiz = iBlk / (nbx*nby);
		long iix = iBlk - iiz*nbx*nby;
		long iiy = iix / nbx;
		iix = iix - iiy*nbx;

		int x0 = iix*bx;
		int y0 = iiy*by;
		int z0 = iiz*bz;
		
		//printf("iBlk=%d, x0=%d, y0=%d, z0=%d\n",iBlk,x0,y0,z0);

		int thread_id = omp_get_thread_num();
		float* priv_work = work + thread_id * work_size_one_thread;
		float* priv_tmp = priv_work + bx*by*bz;
		long priv_blkoff = glob_blkoffs[iBlk];
		bool Is_Uncompressed = priv_blkoff < 0 ? true : false;
		priv_blkoff = Is_Uncompressed ? -priv_blkoff : priv_blkoff;
		unsigned long* priv_compressed = (unsigned long*)(((char*)bytes) + priv_blkoff);
		
		if (Is_Uncompressed)
		{
			Wavelet_Transform_Fast_Inverse((__m256*)priv_compressed,(__m256*)priv_tmp,bx,by,bz);
			Copy_From_Block((__m128*)priv_compressed,bx,by,bz,vol,x0,y0,z0,nx,ny,nz);
		}
		else
		{
			Run_Length_Decode_Slow(mulfac,priv_work,bx*by*bz,priv_compressed);
			//printf("...Run_Length_Decode_Slow done\n");
			Wavelet_Transform_Fast_Inverse((__m256*)priv_work,(__m256*)priv_tmp,bx,by,bz);
			//printf("...Wavelet_Transform_Fast_Inverse done\n");
			Copy_From_Block((__m128*)priv_work,bx,by,bz,vol,x0,y0,z0,nx,ny,nz);
			//printf("...Copy_From_Block done\n");
		}
	}

	free(work);
}

//
// Module tests.
// 

static void Fill_Block(float* data1, float* data2, int bx, int by, int bz)
{
	srand48(time(NULL));
	for (int i = 0;  i < bx*by*bz;  ++i)
	{
		data1[i] = data2[i] = drand48();
	}
}

static bool Compare_Blocks(float* data1, float* data2, int bx, int by, int bz)
{
	float rms1 = 0.0f, rms_diff = 0.0f;
	for (int i = 0;  i < bx*by*bz;  ++i)
	{
		rms1 += (data1[i]*data1[i]);
		float diff = data1[i] - data2[i];
		rms_diff += (diff*diff);
	}
	rms1 = sqrtf(rms1/(float)(bx*by*bz));
	rms_diff = sqrtf(rms_diff/(float)(bx*by*bz));
	if (fabs(rms_diff/rms1) < 1e-5f) return true; else return false;
}

static float* omp_allocate(long num_floats)
{
	long tot_size = (long)sizeof(float) * num_floats;
	long num_pages = (tot_size + 4095) / 4096;
	tot_size = num_pages * 4096;
	__m128* ptr = 0L;
	posix_memalign((void**)&ptr, 64, tot_size);
#pragma omp parallel for schedule(static,1)
        for (long iPage = 0;  iPage < num_pages;  ++iPage)
        {
                __m128* p = ptr + iPage * 256;
                for (int idx = 0;  idx < 256;  ++idx) p[idx] = _mm_setzero_ps();
        }
	return (float*)ptr;
}

static void Fill_Volume_With_Pattern(float* vol, long cnx, long cny, long cnz, long seed)
{
	for (long i = 0;  i < cnx*cny*cnz;  ++i) ((unsigned int*)vol)[i] = i + seed;
}

static bool Check_Block_For_Pattern(float* block, int x0, int y0, int z0, int bx, int by, int bz, float* vol, long cnx, long cny, long cnz)
{
	for (long iz = 0;  iz < bz;  ++iz)
	{
		for (long iy = 0;  iy < by;  ++iy)
		{
			for (long ix = 0;  ix < bx;  ++ix)
			{
				long block_idx = (iz*by+iy)*bx+ix;
				unsigned int block_val = ((unsigned int*)block)[block_idx];
				long x = x0 + ix;
				long y = y0 + iy;
				long z = z0 + iz;
				unsigned int vol_val = 0;
				if (x >= 0 && x < cnx && y >= 0 && y < cny && z >= 0 && z < cnz)
				{
					long vol_idx = ((iz+z0)*cny+(iy+y0))*cnx+(ix+x0);
					vol_val = ((unsigned int*)vol)[vol_idx];
				}
				if (block_val != vol_val)
				{
					//printf("Error! Check_Block_For_Pattern(x0=%d,y0=%d,z0=%d,bx=%d,by=%d,bz=%d,cnx=%d,cny=%d,cnz=%d) @ix=%d,iy=%d,iz=%d -- found value %d, expected %d\n",
					//	x0,y0,z0,bx,by,bz,cnx,cny,cnz,ix,iy,iz,block_val,vol_val);
					return false;
				}
			}
		}
	}
	return true;
}

static bool Check_Volume(float* vol, float* vol2, int nx, int ny, int nz)
{
	long nn = (long)nx * (long)ny * (long)nz;
	for (long i = 0;  i < nn;  ++i)
		if (vol[i] != vol2[i])
		{
			return false;
		}
	return true;
}

bool CvxCompress::Run_Module_Tests(bool verbose, bool exhaustive_throughput_tests)
{
	int num_threads;
#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
	}

	printf("\n*\n* CvxCompress module tests ");
#ifdef __AVX2__
	printf(" (AVX 2.0 version).\n");
#else
	printf(" (AVX version).\n");
#endif
	printf("*\n\n");

	printf("0. Verify correctness of memcpy_avx...");  fflush(stdout);
	int *test_src = 0L, *test_dst = 0L, *test_dst2 = 0L;
	posix_memalign((void**)&test_src, 64, sizeof(int)*128*1024);
	posix_memalign((void**)&test_dst, 64, sizeof(int)*128*1024);
	posix_memalign((void**)&test_dst2, 64, sizeof(int)*128*1024);
	for (int i = 0;  i < 128*1024;  ++i) {test_src[i] = i; test_dst[i] = 0;}
	
	bool memcpy_avx_passed = true;
	for (int src_start_off = 0;  memcpy_avx_passed && src_start_off < 32;  ++src_start_off)
	{
		for (int dst_start_off = 0;  memcpy_avx_passed && dst_start_off < 32;  ++dst_start_off)
		{
			for (int len_diff = 0;  len_diff < 32;  ++len_diff)
			{
				int len = sizeof(int)*96*1024 + len_diff;
				memset(test_dst,0,sizeof(int)*128*1024);
				memset(test_dst2,0,sizeof(int)*128*1024);
				char* src = ((char*)test_src) + src_start_off;
				char* dst = ((char*)test_dst) + dst_start_off;
				char* ref = ((char*)test_dst2) + dst_start_off;
				memcpy_avx(dst,src,len);
				memcpy(ref,src,len);
				for (int i = 0;  i < 128*1024;  ++i)
				{
					if (test_dst[i] != test_dst2[i])
					{
						memcpy_avx_passed = false;
						printf("\n -> src_start_off=%d, dst_start_off=%d, len_diff=%d;  blocks differ at index %d!\n",src_start_off,dst_start_off,len_diff,i*4);
						break;
					}
				}
			}
		}
	}
	if (memcpy_avx_passed)
	{
		printf("[\x1B[32mPassed!\x1B[0m]\n");
	}
	else
	{
		printf("[\x1B[31mFailed!\x1B[0m]\n");
	}

	free(test_dst2);
	free(test_dst);
	free(test_src);

	bool forward_passed = true;
	printf("1. Verify correctness of forward wavelet transform...");  fflush(stdout);
	if (verbose) printf("\n");
#define MIN(a,b) (a<b?a:b)
#define MAX(a,b) (a>b?a:b)
	int max_bs = MAX(Max_BX(),MAX(Max_BY(),Max_BZ()));
	int buf_size = (3*Max_BX()*Max_BY()*Max_BZ() + max_bs*8);
	float* data1 = omp_allocate((long)buf_size*(long)num_threads);
	float* data2 = data1 + Max_BX()*Max_BY()*Max_BZ();
	float* work = data2 + Max_BX()*Max_BY()*Max_BZ();
	int min_i = Find_Pow2(Min_BX());
	int max_i = Find_Pow2(Max_BX());
	int min_j = Find_Pow2(Min_BY());
	int max_j = Find_Pow2(Max_BY());
	int min_k = Find_Pow2(Min_BZ());
	int max_k = Find_Pow2(Max_BZ());
	for (int k = min_k;  k <= max_k;  ++k)
	{
		int bz = 1 << k;
		for (int j = min_j;  j <= max_j;  ++j)
		{
			int by = 1 << j;
			for (int i = min_i;  i <= max_i;  ++i)
			{
				int bx = 1 << i;
				if (verbose) printf("\x1B[0m -> %dx%dx%d ",bx,by,bz);  fflush(stdout);
				Fill_Block(data1,data2,bx,by,bz);
				Wavelet_Transform_Slow_Forward(data1,work,bx,by,bz,0,0,0,bx,by,bz);
				Wavelet_Transform_Fast_Forward((__m256*)data2,(__m256*)work,bx,by,bz);
				if (Compare_Blocks(data1,data2,bx,by,bz))
				{
					if (verbose) printf("\x1B[32mPassed!\n");
				}
				else
				{
					if (verbose) printf("\x1B[31mFailed!\n");
					forward_passed = false;
				}
			}
		}
	}
	if (verbose)
	{
		printf("\x1B[0m\n");
	}
	else
	{
		if (forward_passed)
			printf("[\x1B[32mPassed\x1B[0m]\n"); 
		else 
			printf("[\x1B[31mFailed\x1B[0m]\n");
	}

	printf("2. Verify correctness of inverse wavelet transform...");
	if (verbose) printf("\n");
	bool inverse_passed = true;
	for (int k = min_k;  k <= max_k;  ++k)
	{
		int bz = 1 << k;
		for (int j = min_j;  j <= max_j;  ++j)
		{
			int by = 1 << j;
			for (int i = min_i;  i <= max_i;  ++i)
			{
				int bx = 1 << i;
				if (verbose) printf("\x1B[0m -> %dx%dx%d ",bx,by,bz);  fflush(stdout);
				Fill_Block(data1,data2,bx,by,bz);
				Wavelet_Transform_Slow_Inverse(data1,work,bx,by,bz,0,0,0,bx,by,bz);
				Wavelet_Transform_Fast_Inverse((__m256*)data2,(__m256*)work,bx,by,bz);
				if (Compare_Blocks(data1,data2,bx,by,bz))
				{
					if (verbose) printf("\x1B[32mPassed!\n");
				}
				else
				{
					if (verbose) printf("\x1B[31mFailed!\n");
					inverse_passed = false;
				}
			}
		}
	}
	if (verbose)
	{
		printf("\x1B[0m\n");
	}
	else
	{
		if (inverse_passed)
			printf("[\x1B[32mPassed\x1B[0m]\n");
		else 
			printf("[\x1B[31mFailed\x1B[0m]\n");
	}

	printf("3. Test throughput of wavelet transform (forward + inverse)...\n");
	for (int k = min_k;  k <= max_k;  ++k)
	{
		int bz = 1 << k;
		for (int j = min_j;  j <= max_j;  ++j)
		{
			int by = 1 << j;
			for (int i = min_i;  i <= max_i;  ++i)
			{
				int bx = 1 << i;
				if (exhaustive_throughput_tests || (bx == by && by == bz))
				{
					char* memtype = 0L;
					long block_size = (long)bx * (long)by * (long)bz;
					if (block_size <= 4096)
						memtype = " L1 ";
					else if (block_size <= 32768)
						memtype = " L2 ";
					else if (block_size <= 262144)
						memtype = " L3 ";
					else
						memtype = "DRAM";
					printf("\x1B[0m -> %3d x %3d x %3d (%s) ",bx,by,bz,memtype);  fflush(stdout);
					int niter = (int)((long)num_threads * (1024*1024*1024+((bx*by*bz)-1)) / (bx*by*bz));

					for (int iThr = 0;  iThr < num_threads;  ++iThr)
					{
						float* priv_data1 = data1 + iThr * buf_size;
						float* priv_data2 = priv_data1 + bx * by * bz;
						Fill_Block(priv_data1,priv_data2,bx,by,bz);
					}

					struct timespec before, after;
					clock_gettime(CLOCK_REALTIME,&before);
#pragma omp parallel for schedule(static,1)
					for (int iter = 0;  iter < niter;  ++iter)
					{
						int thread_id = omp_get_thread_num();
						float* priv_data1 = data1 + thread_id * buf_size;
						float* priv_data2 = priv_data1 + bx * by * bz;
						float* priv_work = priv_data2 + bx * by * bz;
						Wavelet_Transform_Fast_Forward((__m256*)priv_data2,(__m256*)priv_work,bx,by,bz);
						Wavelet_Transform_Fast_Inverse((__m256*)priv_data2,(__m256*)priv_work,bx,by,bz);
					}
					clock_gettime(CLOCK_REALTIME,&after);
					double elapsed = (double)after.tv_sec + (double)after.tv_nsec * 1e-9 - (double)before.tv_sec - (double)before.tv_nsec * 1e-9;
					double mcells_per_second = (double)(bx*by*bz) * (double)niter / (elapsed * 1e6);
					double GF_per_second = mcells_per_second * 1e-3 * 2.0 * 69.0;
					printf(":: %6.3f secs - %.0f MCells/s - %.0f GF/s\n",elapsed,mcells_per_second,GF_per_second);
				}
			}
		}
	}

	printf("\n4. Verify correctness of Copy_To_Block method...");  fflush(stdout);
	bool copy_to_block_passed = true;
	long nx = 1024;
	long ny = 1024;
	long nz = 1024;
	float* vol = 0L;
	float* vol2 = 0L;
	float* block = 0L;
	if (nx < 2*Max_BX() || ny < 2*Max_BY() || nz < 2*Max_BZ())
	{
		printf("Skipped. Check code.");  fflush(stdout);
		copy_to_block_passed = false;
		if (verbose) printf("\n");
	}
	else
	{
		if (verbose) printf("\n");
		vol = omp_allocate((long)2*nx*ny*nz);
		vol2 = vol + nx*ny*nz;
		block = omp_allocate(Max_BX()*Max_BY()*Max_BZ());
		for (int k = min_k;  k <= max_k;  ++k)
		{
			int bz = 1 << k;
			for (int j = min_j;  j <= max_j;  ++j)
			{
				int by = 1 << j;
				for (int i = min_i;  i <= max_i;  ++i)
				{
					int bx = 1 << i;

					bool copy_to_this_block_passed = true;
					int cnx = bx + 3;
					int cny = by + 5;
					int cnz = bz + 7;

					if (verbose) {printf(" -> %3d x %3d x %3d ... ",bx,by,bz);  fflush(stdout);}
					Fill_Volume_With_Pattern(vol,cnx,cny,cnz,0);
					for (int k_off = 0;  k_off <= 1;  ++k_off)
					{
						for (int j_off = 0;  j_off <= 1;  ++j_off)
						{
							for (int i_off = 0;  i_off <= 1;  ++i_off)
							{
								int x0 = i_off*bx;
								int y0 = j_off*by;
								int z0 = k_off*bz;
								Copy_To_Block(vol,x0,y0,z0,cnx,cny,cnz,(__m128*)block,bx,by,bz);
								if (!Check_Block_For_Pattern(block,x0,y0,z0,bx,by,bz,vol,cnx,cny,cnz))
								{
									// add a useful error message
									copy_to_block_passed = false;
									copy_to_this_block_passed = false;
								}
							}
						}
					}
					if (copy_to_this_block_passed)
					{
						if (verbose) printf("\x1B[0m[\x1B[32mPassed!\x1B[0m]\n");
					}
					else
					{
						if (verbose) printf("\x1B[0m[\x1B[31mFailed!\x1B[0m]\n");
					}
				}
			}
		}
	}
	if (!verbose)
		if (copy_to_block_passed)
			printf("\x1B[0m[\x1B[32mPassed!\x1B[0m]\n");
		else
			printf("\x1B[0m[\x1B[31mFailed!\x1B[0m]\n");
	
	printf("5. Verify correctness of Copy_From_Block method...");  fflush(stdout);
	bool copy_from_block_passed = true;
	if (vol == 0L || block == 0L)
	{
		printf("Skipped. Check code.");  fflush(stdout);
		copy_from_block_passed = false;
		if (!verbose) printf("\n");
	}
	else
	{
		if (verbose) printf("\n");
		for (int k = min_k;  k <= max_k;  ++k)
		{
			int bz = 1 << k;
			for (int j = min_j;  j <= max_j;  ++j)
			{
				int by = 1 << j;
				for (int i = min_i;  i <= max_i;  ++i)
				{
					int bx = 1 << i;

					bool copy_from_this_block_passed = true;
					int cnx = bx + 3;
					int cny = by + 5;
					int cnz = bz + 7;

					if (verbose) {printf(" -> %3d x %3d x %3d ... ",bx,by,bz);  fflush(stdout);}
					Fill_Volume_With_Pattern(vol,cnx,cny,cnz,0);
					for (int k_off = 0;  k_off <= 1;  ++k_off)
					{
						for (int j_off = 0;  j_off <= 1;  ++j_off)
						{
							for (int i_off = 0;  i_off <= 1;  ++i_off)
							{
								int x0 = i_off*bx;
								int y0 = j_off*by;
								int z0 = k_off*bz;
								Copy_To_Block(vol,x0,y0,z0,cnx,cny,cnz,(__m128*)block,bx,by,bz);
								Copy_From_Block((__m128*)block,bx,by,bx,vol2,x0,y0,z0,cnx,cny,cnz);
								if (!Check_Block_For_Pattern(block,x0,y0,z0,bx,by,bz,vol2,cnx,cny,cnz))
								{
									// add a useful error message
									copy_from_block_passed = false;
									copy_from_this_block_passed = false;
								}
							}
						}
					}
					if (copy_from_this_block_passed)
					{
						if (verbose) printf("\x1B[0m[\x1B[32mPassed!\x1B[0m]\n");
					}
					else
					{
						if (verbose) printf("\x1B[0m[\x1B[31mFailed!\x1B[0m]\n");
					}
				}
			}
		}
	}
	if (!verbose)
		if (copy_from_block_passed)
			printf("\x1B[0m[\x1B[32mPassed!\x1B[0m]\n");
		else
			printf("\x1B[0m[\x1B[31mFailed!\x1B[0m]\n");

	printf("6. Test throughput of block copy...");  fflush(stdout);
	bool copy_round_trip_passed = true;
	if (vol == 0L)
	{
		printf("Skipped. Check code.\n");
	}
	else
	{
		printf("\n");
		Fill_Volume_With_Pattern(vol,nx,ny,nz,0);
		Fill_Volume_With_Pattern(vol2,nx,ny,nz,1);
		for (int k = min_k;  k <= max_k;  ++k)
		{
			int bz = 1 << k;
			for (int j = min_j;  j <= max_j;  ++j)
			{
				int by = 1 << j;
				for (int i = min_i;  i <= max_i;  ++i)
				{
					int bx = 1 << i;
					if (exhaustive_throughput_tests || (bx == by && by == bz))
					{
						printf("\x1B[0m -> %3d x %3d x %3d ",bx,by,bz);  fflush(stdout);

						int nbx = (nx+bx-1)/bx;
						int nby = (ny+by-1)/by;
						int nbz = (nz+bz-1)/bz;
						int nnn = nbx*nby*nbz;

						struct timespec before, after;
						clock_gettime(CLOCK_REALTIME,&before);
#pragma omp parallel for schedule(static,8)
						for (int iBlk = 0;  iBlk < nnn;  ++iBlk)
						{
							int iiz = iBlk / (nbx*nby);
							int iix = iBlk - (iiz*nbx*nby);
							int iiy = iix / nbx;
							iix = iix - (iiy*nbx);

							int x0 = iix*bx;
							int y0 = iiy*by;
							int z0 = iiz*bz;

							int thread_id = omp_get_thread_num();
							float* priv_data1 = data1 + thread_id * buf_size;

							Copy_To_Block(vol,x0,y0,z0,nx,ny,nz,(__m128*)priv_data1,bx,by,bz);
							Copy_From_Block((__m128*)priv_data1,bx,by,bz,vol2,x0,y0,z0,nx,ny,nz);
						}
						clock_gettime(CLOCK_REALTIME,&after);
						double elapsed = (double)after.tv_sec + (double)after.tv_nsec * 1e-9 - (double)before.tv_sec - (double)before.tv_nsec * 1e-9;
						double mcells_per_sec = (double)nx * (double)ny * (double)nz / (elapsed * 1e6);
						double GB_per_sec = (double)sizeof(float) * (double)nx * (double)ny * (double)nz * 3.0 / (elapsed * 1e9);
					
						if (!Check_Volume(vol,vol2,nx,ny,nz))
						{
							printf("\x1B[0m[\x1B[31mFailed!\x1B[0m]\n");
							copy_round_trip_passed = false;
						}
						else
						{					
							printf("\x1B[0m[\x1B[32mPassed!\x1B[0m] :: %6.3f secs - %.0f MCells/s - %.2f GB/s\n",elapsed,mcells_per_sec,GB_per_sec);
						}
					}
				}
			}
		}
	}
	printf("\n");

	printf("7. Verify correctness of Global_RMS method...");  fflush(stdout);
	bool global_rms_passed = true;
	if (vol == 0L || block == 0L)
	{
		printf("Skipped. Check code.");  fflush(stdout);
		global_rms_passed = false;
		if (!verbose) printf("\n");
	}
	else
	{
		if (verbose) printf("\n");
		int cnx = 37;
		int cny = 41;
		int cnz = 43;
		Fill_Block(vol,vol2,cnx,cny,cnz);
		float global_rms = Compute_Global_RMS(vol,cnx,cny,cnz);
		double acc = 0.0;
		for (long i = 0;  i < (long)cnx*(long)cny*(long)cnz;  ++i) acc += vol[i] * vol[i];
		float slow_global_rms = (float)sqrt(acc/((double)cnx*(double)cny*(double)cnz));
		float ratio = (global_rms - slow_global_rms) / slow_global_rms;
		ratio = ratio < 0.0f ? -ratio : ratio;
		if (ratio < 1e-5f)
		{
			printf("\x1B[0m[\x1B[32mPassed!\x1B[0m]\n");
		}
		else
		{
			printf("\x1B[0m[\x1B[31mFailed!\x1B[0m]\n");
			global_rms_passed = false;
		}
	}

	float scale = 1e-1f;

	printf("8. Test throughput of Compress() method...\n");
	int nx3,ny3,nz3;
	float* vol3;
	Read_Raw_Volume("pressure_at_t=7512.bin",nx3,ny3,nz3,vol3);
	unsigned long* compressed3;
	posix_memalign((void**)&compressed3, 64, (long)sizeof(float)*(long)nx3*(long)ny3*(long)nz3);
	for (int k = min_k;  k <= max_k;  ++k)
	{
		int bz = 1 << k;
		for (int j = min_j;  j <= max_j;  ++j)
		{
			int by = 1 << j;
			for (int i = min_i;  i <= max_i;  ++i)
			{
				int bx = 1 << i;
				if (exhaustive_throughput_tests || (bx == by && by == bz))
				{
					char* memtype = 0L;
					long block_size = (long)bx * (long)by * (long)bz;
					if (block_size <= 4096)
						memtype = " L1 ";
					else if (block_size <= 32768)
						memtype = " L2 ";
					else if (block_size <= 262144)
						memtype = " L3 ";
					else
						memtype = "DRAM";
					printf("\x1B[0m -> %3d x %3d x %3d (%s) ",bx,by,bz,memtype);  fflush(stdout);

					struct timespec before, after;
					clock_gettime(CLOCK_REALTIME,&before);
					double elapsed = 0.0;

					float ratio = 0.0f;
					int niter = 0;
					do
					{
						long compressed_length = 0l;
						ratio = Compress(scale,vol3,nx3,ny3,nz3,bx,by,bz,(unsigned int*)compressed3,compressed_length);
						clock_gettime(CLOCK_REALTIME,&after);
						++niter;
						elapsed = (double)after.tv_sec + (double)after.tv_nsec * 1e-9 - (double)before.tv_sec - (double)before.tv_nsec * 1e-9;
						double mcells_per_sec = (double)niter * (double)nx3 * (double)ny3 * (double)nz3 / (elapsed * 1e6);
						printf("\r\x1B[0m -> %3d x %3d x %3d (%s) %2d iterations - %6.3f secs - %.0f MCells/s - ratio %.2f:1",bx,by,bz,memtype,niter,elapsed,mcells_per_sec,ratio);
						fflush(stdout);
					} while (elapsed < 10.0);
					printf("\n");
					//printf("%d iterations - %6.3f secs - %.0f MCells/s - ratio %.2f:1\n",niter,elapsed,mcells_per_sec,ratio);
				}
			}
		}
	}

	printf("9. Test throughput of Decompress() method...\n");
	for (int k = min_k;  k <= max_k;  ++k)
	{
		int bz = 1 << k;
		for (int j = min_j;  j <= max_j;  ++j)
		{
			int by = 1 << j;
			for (int i = min_i;  i <= max_i;  ++i)
			{
				int bx = 1 << i;
				if (exhaustive_throughput_tests || (bx == by && by == bz))
				{
					long compressed_length3 = 0l;
					float ratio = Compress(scale,vol3,nx3,ny3,nz3,bx,by,bz,(unsigned int*)compressed3,compressed_length3);

					char* memtype = 0L;
					long block_size = (long)bx * (long)by * (long)bz;
					if (block_size <= 4096)
						memtype = " L1 ";
					else if (block_size <= 32768)
						memtype = " L2 ";
					else if (block_size <= 262144)
						memtype = " L3 ";
					else
						memtype = "DRAM";
					printf("\x1B[0m -> %3d x %3d x %3d (%s) ",bx,by,bz,memtype);  fflush(stdout);

					struct timespec before, after;
					clock_gettime(CLOCK_REALTIME,&before);
					double elapsed = 0.0;

					int niter = 0;
					do
					{
						int nx4, ny4, nz4;
						float* vol4;
						Decompress(vol4,nx4,ny4,nz4,(unsigned int*)compressed3,compressed_length3);
						clock_gettime(CLOCK_REALTIME,&after);
						++niter;
						elapsed = (double)after.tv_sec + (double)after.tv_nsec * 1e-9 - (double)before.tv_sec - (double)before.tv_nsec * 1e-9;
						double mcells_per_sec = (double)niter * (double)nx3 * (double)ny3 * (double)nz3 / (elapsed * 1e6);
						printf("\r\x1B[0m -> %3d x %3d x %3d (%s) %2d iterations - %6.3f secs - %.0f MCells/s",bx,by,bz,memtype,niter,elapsed,mcells_per_sec);
						fflush(stdout);
						free(vol4);
					} while (elapsed < 10.0);
					printf("\n");
				}
			}
		}
	}
	if (vol3 != 0L) free(vol3);
	if (compressed3 != 0L) free(compressed3);

	if (data1 != 0L) free(data1);
	if (block != 0L) free(block);
	if (vol != 0L) free(vol);

	return forward_passed && inverse_passed && copy_to_block_passed && copy_from_block_passed && copy_round_trip_passed && global_rms_passed;
}
