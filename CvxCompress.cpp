#include <math.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <chrono>
#include "CvxCompress.hxx"
#include "Wavelet_Transform_Fast.hxx"
#include "Wavelet_Transform_Slow.hxx"  // for comparison in module test
#include "Block_Copy.hxx"
#include "Run_Length_Encode_Slow.hxx"  // turns out, it isn't that slow after all
#include "Read_Raw_Volume.hxx"

#ifndef __INTEL_COMPILER
#undef PAPI
#endif

#ifdef PAPI
#include "papi.h"
#endif

using namespace std;
using std::chrono::system_clock;
typedef std::chrono::system_clock Time;
typedef std::chrono::milliseconds ms;
typedef std::chrono::duration<double> fsec;

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

bool CvxCompress::Is_Valid_Block_Size(int bx, int by, int bz)
{
	if (
		((1 << Find_Pow2(bx)) == bx) && 
		((1 << Find_Pow2(by)) == by) &&
		((1 << Find_Pow2(bz)) == bz) &&
		(bx >= Min_BX() && bx <= Max_BX()) &&
		(by >= Min_BY() && by <= Max_BY()) &&
		(bz == 1 || (bz >= Min_BZ() && bz <= Max_BZ()))
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

static float Compute_Local_RMS(__m256* blk, int bx, int by, int bz)
{
	int nn = bz * by * (bx >> 3);
	float rms = 0.0f;
	__m256 acc = _mm256_setzero_ps();
	for (int i = 0;  i < nn;  ++i)
	{
		__m256 val = _mm256_loadu_ps((float*)(blk+i));
#ifdef __AVX2__
		acc = _mm256_fmadd_ps(val,val,acc);
#else
		acc = _mm256_add_ps(acc,_mm256_mul_ps(val,val));
#endif
	}
	acc = _mm256_hadd_ps(acc,acc);
	acc = _mm256_hadd_ps(acc,acc);
	__m128 acc0 = _mm256_extractf128_ps(acc,0);
	__m128 acc1 = _mm256_extractf128_ps(acc,1);
	acc0 = _mm_add_ps(acc0,acc1);
	float v[4];
	_mm_store_ps(v,acc0);
	rms = sqrtf(v[0]/(float)(bx*by*bz));
	return rms;
}

#define GET_PRIVATE_POINTERS(work,thread_id) \
float* priv_work = (float*)(work + thread_id * work_size_one_thread); \
float* priv_tmp = priv_work + work_wave_transform_buffer_size; \
int* priv_blkstore_idx = (int*)(priv_tmp + work_wave_transform_tmp_buffer_size); \
int* priv_blkoff = (int*)(priv_blkstore_idx + 1); \
int* priv_iBlk = (int*)(priv_blkstore_idx + work_blkoff_buffer_size); \
unsigned int* priv_compress_buffer = (unsigned int*)(priv_iBlk + work_blkoff_buffer_size)

#define ASSERT_ALIGNMENT(p) assert(((long)p & 31) == 0)

int is_pow2(int val)
{
        if (val <= 1)
        {
                return 0;
        }
        else
        {
                int shift_val = val >> 1;
                int num_shifts = 0;
                while (shift_val != 0)
                {
                        ++num_shifts;
                        shift_val = shift_val >> 1;
                };
                return (val == (1 << num_shifts)) ? -1 : 0;
        }
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
	bool use_local_RMS = false;
	return Compress(scale,vol,nx,ny,nz,bx,by,bz,use_local_RMS,compressed,compressed_length);
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
	bool use_local_RMS,
	unsigned int* compressed,
	long& compressed_length 
	)
{
	assert(bx >= CvxCompress::Min_BX() && bx <= CvxCompress::Max_BX() && is_pow2(bx));
	assert(by >= CvxCompress::Min_BY() && by <= CvxCompress::Max_BY() && is_pow2(by));
	assert(bz == 1 || (bz >= CvxCompress::Min_BZ() && bz <= CvxCompress::Max_BZ() && is_pow2(bz)));
	float global_rms = use_local_RMS ? 1.0f : Compute_Global_RMS(vol,nx,ny,nz);
	
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
	int work_compress_buffer_size = priv_blkoff_len*bx*by*bz + ((bx*by*bz)>>2);
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
		GET_PRIVATE_POINTERS(work,thread_id);
		ASSERT_ALIGNMENT(priv_work);
		ASSERT_ALIGNMENT(priv_tmp);
		int* p = (int*)(work + thread_id * work_size_one_thread);
		for (int i = 0;  i < work_size_one_thread;  ++i) p[i] = 0;
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
	
	float glob_mulfac = global_rms != 0.0f ? 1.0f / (global_rms * scale) : 1.0f;
	compressed[6] = *((unsigned int*)&glob_mulfac);

	fprintf(stderr,"Made it here (%d)!\n",1);

	// flags:
	// 1 -> use local RMS (global RMS otherwise)
	compressed[7] = use_local_RMS ? 1 : 0;

	long* glob_blkoffs = (long*)(compressed+8);  // no need to initialize

	float* blkmulfac = 0L;
	unsigned int* bytes;
	if (use_local_RMS)
	{
		blkmulfac = (float*)(glob_blkoffs+nnn);
		bytes = (unsigned int*)(blkmulfac+nnn);
	}
	else
	{
		blkmulfac = 0L;
		bytes = (unsigned int*)(glob_blkoffs+nnn);
	}
	long byte_offset = 0l;

fprintf(stderr,"Made it here (%d)!\n",2);
#pragma omp parallel for schedule(dynamic)
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
		GET_PRIVATE_POINTERS(work,thread_id);

		priv_iBlk[*priv_blkstore_idx] = iBlk;
		int blkoff = priv_blkoff[*priv_blkstore_idx];
		unsigned long* priv_compressed = (unsigned long*)(((char*)priv_compress_buffer) + blkoff);

		Copy_To_Block(vol,x0,y0,z0,nx,ny,nz,(__m128*)priv_work,bx,by,bz);
		Wavelet_Transform_Fast_Forward((__m256*)priv_work,(__m256*)priv_tmp,bx,by,bz);
		if(iBlk == 0)
		{
			fprintf(stderr,"Made it here (%d%s)!\n",2,"a");
		}
		int bytepos = 0, error = 0;
		float mulfac = glob_mulfac;
		if (use_local_RMS)
		{
			float local_RMS = Compute_Local_RMS((__m256*)priv_work,bx,by,bz);
			mulfac = local_RMS != 0.0f ? 1.0f / (local_RMS * scale) : 1.0f;
			blkmulfac[iBlk] = mulfac;
		}
		Run_Length_Encode_Slow(mulfac,priv_work,bx*by*bz,priv_compressed,bytepos);
		if(iBlk == 0)
		{
			fprintf(stderr,"Made it here (%d%s)!\n",2,"b");
		}
		error = (bytepos > (4*bx*by*bz)) ? -1 : 0;
		//printf("Compressed block is %d bytes (ratio=%.2f:1, error = %d)\n",bytepos,(double)(4*bx*by*bz)/(double)bytepos,error);
		//Run_Length_Encode_Fast(mulfac,priv_work,bx*by*bz,priv_compressed,bytepos,error);

		++(*priv_blkstore_idx);
		if (error)
		{
			priv_blkoff[(*priv_blkstore_idx)-1] |= -2147483648;
			priv_blkoff[*priv_blkstore_idx] = blkoff+sizeof(float)*bx*by*bz;
			memcpy(priv_compressed,priv_work,sizeof(float)*bx*by*bz);
		}
		else
		{
			priv_blkoff[*priv_blkstore_idx] = blkoff + bytepos;
		}
				if(iBlk == 0)
		{
			fprintf(stderr,"Made it here (%d%s)!\n",2,"c");
		}
		if (*priv_blkstore_idx >= priv_blkoff_len)
		{
			// copy compressed blocks from private area to global area.
			int priv_blklen = priv_blkoff[*priv_blkstore_idx];
			char* glob_dst = 0L;
#pragma omp critical
			{
				glob_dst = ((char*)bytes) + byte_offset;
				byte_offset += (long)priv_blklen;
			}
			//printf("MEMCPY :: GLOB byte_offset=%ld, priv_blkstore_idx=%d, priv_blklen=%d\n",byte_offset,*priv_blkstore_idx,priv_blklen);
			for (int i = 0;  i < *priv_blkstore_idx;  ++i) 
			{
				int dst_iBlk = priv_iBlk[i];
				int blkoff = priv_blkoff[i];
				bool uncompressed = (blkoff & 0x80000000) ? true : false;
				blkoff = blkoff & 0x7FFFFFFF;
				long new_glob_blkoff = (glob_dst + blkoff) - (char*)bytes;
				new_glob_blkoff = uncompressed ? (new_glob_blkoff | 0x8000000000000000) : new_glob_blkoff;
				glob_blkoffs[dst_iBlk] = new_glob_blkoff;
				//printf("  uncompressed=%s, blkoff=%ld, glob_blkoffs[%d]=%ld\n",uncompressed?"true":"false",blkoff,dst_iBlk,glob_blkoffs[dst_iBlk]);
			}
			memcpy(glob_dst,priv_compress_buffer,priv_blklen);
			*priv_blkstore_idx = 0;
			priv_blkoff[0] = 0;
		}
	}
	fprintf(stderr,"Made it here (%d)!\n",3);
	for (int thread_id = 0;  thread_id < num_threads;  ++thread_id)
	{
		GET_PRIVATE_POINTERS(work,thread_id);
		if (*priv_blkstore_idx >= 1)
		{
			// copy compressed blocks from private area to global area.
			int priv_blklen = priv_blkoff[*priv_blkstore_idx];
                        char* glob_dst = 0L;
                        {
                                glob_dst = ((char*)bytes) + byte_offset;
                                byte_offset += (long)priv_blklen;
                        }
                        //printf("MEMCPY :: GLOB byte_offset=%ld, priv_blkstore_idx=%d, priv_blklen=%d\n",byte_offset,*priv_blkstore_idx,priv_blklen);
                        for (int i = 0;  i < *priv_blkstore_idx;  ++i)
                        {
                                int dst_iBlk = priv_iBlk[i];
                                int blkoff = priv_blkoff[i];
                                bool uncompressed = (blkoff & 0x80000000) ? true : false;
                                blkoff = blkoff & 0x7FFFFFFF;
                                long new_glob_blkoff = (glob_dst + blkoff) - (char*)bytes;
                                new_glob_blkoff = uncompressed ? (new_glob_blkoff | 0x8000000000000000) : new_glob_blkoff;
                                glob_blkoffs[dst_iBlk] = new_glob_blkoff;
                                //printf("  uncompressed=%s, blkoff=%ld, glob_blkoffs[%d]=%ld\n",uncompressed?"true":"false",blkoff,dst_iBlk,glob_blkoffs[dst_iBlk]);
                        }
                        memcpy(glob_dst,priv_compress_buffer,priv_blklen);
                        *priv_blkstore_idx = 0;
                        priv_blkoff[0] = 0;
		}
	}
	fprintf(stderr,"Made it here (%d)!\n",4);
	compressed_length = 32 + 8*nnn + byte_offset + 7;
	if (use_local_RMS) compressed_length += 4*nnn;

	free(work);
	double ratio = ((double)nx * (double)ny * (double)nz * (double)sizeof(float)) / (double)compressed_length;
	return (float)ratio;
}

float* CvxCompress::Decompress(
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
	float* vol;
	posix_memalign((void**)&vol, 64, (long)nx*(long)ny*(long)nz*(long)sizeof(float));
	Decompress(vol, nx, ny, nz, compressed, compressed_length);
	return vol;
}

void CvxCompress::Decompress(
	float *vol,
	int nx,
	int ny,
	int nz,
	unsigned int* compressed,
	long compressed_length 
	)
{
	int nx_check = ((int*)compressed)[0];
	int ny_check = ((int*)compressed)[1];
	int nz_check = ((int*)compressed)[2];
	assert(nx == nx_check);
	assert(ny == ny_check);
	assert(nz == nz_check);

	int bx = ((int*)compressed)[3];
	int by = ((int*)compressed)[4];
	int bz = ((int*)compressed)[5];
	float glob_mulfac = ((float*)compressed)[6];
	int flags = ((int*)compressed)[7];
	bool use_local_RMS = (flags & 1) ? true : false;
	//printf("nx=%d, ny=%d, nz=%d, bx=%d, by=%d, bz=%d, mulfac=%e\n",nx,ny,nz,bx,by,bz,mulfac);

	int nbx = (nx+bx-1)/bx;
	int nby = (ny+by-1)/by;
	int nbz = (nz+bz-1)/bz;
	int nnn = nbx*nby*nbz;
	//printf("nbx=%d, nby=%d, nbz=%d, nnn=%d\n",nbx,nby,nbz,nnn);

	long* glob_blkoffs = (long*)(compressed+8);
	
	float* blkmulfac = 0L;
	unsigned int* bytes;
	if (use_local_RMS)
	{
		blkmulfac = (float*)(glob_blkoffs+nnn);
		bytes = (unsigned int*)(blkmulfac+nnn);
	}
	else
	{
		blkmulfac = 0L;
		bytes = (unsigned int*)(glob_blkoffs+nnn);
	}

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
		
		//printf("  iBlk=%d, x0=%d, y0=%d, z0=%d\n",iBlk,x0,y0,z0);

		int thread_id = omp_get_thread_num();
		float* priv_work = work + thread_id * work_size_one_thread;
		float* priv_tmp = priv_work + bx*by*bz;
		long priv_blkoff = glob_blkoffs[iBlk];
		bool Is_Uncompressed = (priv_blkoff & 0x8000000000000000) ? true : false;
		priv_blkoff = Is_Uncompressed ? (priv_blkoff & 0x7FFFFFFFFFFFFFFF) : priv_blkoff;
		unsigned long* priv_compressed = (unsigned long*)(((char*)bytes) + priv_blkoff);
		float mulfac = use_local_RMS ? blkmulfac[iBlk] : glob_mulfac;
		//printf("  Is_Uncompressed=%s, priv_blkoff=%ld\n",Is_Uncompressed?"true":"false",priv_blkoff);
		
		if (Is_Uncompressed)
		{
			//printf("  iBlk=%ld is uncompressed!\n",iBlk);
			memcpy(priv_work,priv_compressed,sizeof(float)*bx*by*bz);
			Wavelet_Transform_Fast_Inverse((__m256*)priv_work,(__m256*)priv_tmp,bx,by,bz);
			Copy_From_Block((__m128*)priv_work,bx,by,bz,vol,x0,y0,z0,nx,ny,nz);
		}
		else
		{
			Run_Length_Decode_Slow(mulfac,priv_work,bx*by*bz,priv_compressed);
			//printf("...Run_Length_Decode_Slow done\n");  fflush(stdout);
			Wavelet_Transform_Fast_Inverse((__m256*)priv_work,(__m256*)priv_tmp,bx,by,bz);
			//printf("...Wavelet_Transform_Fast_Inverse done\n");  fflush(stdout);
			Copy_From_Block((__m128*)priv_work,bx,by,bz,vol,x0,y0,z0,nx,ny,nz);
			//printf("...Copy_From_Block done\n");  fflush(stdout);
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

static double Compute_FLOPS_Single_Dimension(int bx)
{
	int flop = 0;
	for (int i = 2;  i <= bx;  i=i<<1)
	{
		flop += ((23*i)>>1);
	}
	return (double)flop / (double)bx;
}

bool CvxCompress::Run_Module_Tests(bool verbose, bool exhaustive_throughput_tests)
{
	int num_threads;
#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
	}

	printf("\n*\n* CvxCompress module tests (");
#ifdef __INTEL_COMPILER
	printf("ICC%d",__INTEL_COMPILER);
#else
	printf("GCC%d.%d",__GNUC__,__GNUC_MINOR__);
#endif
#ifdef __AVX2__
	printf(", AVX 2.0).\n");
#else
	printf(", AVX).\n");
#endif
	printf("*\n\n");

	bool forward_passed = true;
	printf("2. Verify correctness of forward wavelet transform...");  fflush(stdout);
	if (verbose) printf("\n");
#define MIN(a,b) (a<b?a:b)
#define MAX(a,b) (a>b?a:b)
	long max_bs = MAX(Max_BX(),MAX(Max_BY(),Max_BZ()));
	long buf_size = (3*Max_BX()*Max_BY()*Max_BZ() + max_bs*8);
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

	printf("3. Verify correctness of inverse wavelet transform...");
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

#ifdef PAPI
        int retval = PAPI_library_init( PAPI_VER_CURRENT );
        assert(retval == PAPI_VER_CURRENT);
        int sp_ops_events[1]{PAPI_SP_OPS};
        bool sp_ops_counter_is_available = (PAPI_query_event(sp_ops_events[0]) == PAPI_OK);
        if (sp_ops_counter_is_available)
        {
                assert(PAPI_thread_init((unsigned long (*)(void))(omp_get_num_threads)) == PAPI_OK);
        }
        else
        {
                printf("PAPI does not support PAPI_VEC_SP counter on this machine.\n");
                PAPI_shutdown();
        }
#endif

	printf("4. Test throughput of wavelet transform (forward + inverse)...\n");
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
					const char* memtype = 0L;
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

					for (long iThr = 0;  iThr < num_threads;  ++iThr)
					{
						float* priv_data1 = data1 + (long)iThr * buf_size;
						float* priv_data2 = priv_data1 + bx * by * bz;
						Fill_Block(priv_data1,priv_data2,bx,by,bz);
					}

#ifdef PAPI
					long long sp_ops1 = 0;
					if (sp_ops_counter_is_available)
					{
#pragma omp parallel for schedule(static,1)
						for (int iThr = 0;  iThr < num_threads;  ++iThr)
						{
							PAPI_start_counters(sp_ops_events,1);
						}
					}
#endif

					auto start = Time::now();
#pragma omp parallel for schedule(static,1)
					for (int iter = 0;  iter < niter;  ++iter)
					{
						int thread_id = omp_get_thread_num();
						float* priv_data1 = data1 + (long)thread_id * buf_size;
						float* priv_data2 = priv_data1 + bx * by * bz;
						float* priv_work = priv_data2 + bx * by * bz;
						Wavelet_Transform_Fast_Forward((__m256*)priv_data2,(__m256*)priv_work,bx,by,bz);
						Wavelet_Transform_Fast_Inverse((__m256*)priv_data2,(__m256*)priv_work,bx,by,bz);
					}
					auto stop = Time::now();
#ifdef PAPI
					if (sp_ops_counter_is_available)
					{
#pragma omp parallel for schedule(static,1) reduction(+:sp_ops1)
						for (int iThr = 0;  iThr < num_threads;  ++iThr)
						{
							long long curr_sp_ops = 0L;
							PAPI_stop_counters(&curr_sp_ops,1);
							sp_ops1 += curr_sp_ops;
						}
					}
#endif
					fsec elapsed = (stop - start);
					double mcells_per_second = (double)(bx*by*bz) * (double)niter / (elapsed.count() * 1e6);
					double FLOPS_per_cell = Compute_FLOPS_Single_Dimension(bx) + Compute_FLOPS_Single_Dimension(by) + Compute_FLOPS_Single_Dimension(bz);
					double GF_per_second = mcells_per_second * 1e-3 * 2.0 * FLOPS_per_cell;
#ifdef PAPI
					if (sp_ops_counter_is_available)
					{
						double PAPI_GF_per_second = (double)sp_ops1 / (elapsed.count() * 1e9);
						printf(":: %6.3f secs - %.0f MCells/s - %.0f GF/s - PAPI %.0f GF/s\n",elapsed.count(),mcells_per_second,GF_per_second,PAPI_GF_per_second);
					}
					else
					{
						printf(":: %6.3f secs - %.0f MCells/s - %.0f GF/s\n",elapsed.count(),mcells_per_second,GF_per_second);
					}
#else
					printf(":: %6.3f secs - %.0f MCells/s - %.0f GF/s\n",elapsed.count(),mcells_per_second,GF_per_second);
#endif
				}
			}
		}
	}

	printf("\n5. Verify correctness of Copy_To_Block method...");  fflush(stdout);
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
	
	printf("6. Verify correctness of Copy_From_Block method...");  fflush(stdout);
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

	printf("7. Test throughput of block copy...");  fflush(stdout);
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

						auto start = Time::now();
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
							float* priv_data1 = data1 + (long)thread_id * buf_size;

							Copy_To_Block(vol,x0,y0,z0,nx,ny,nz,(__m128*)priv_data1,bx,by,bz);
							Copy_From_Block((__m128*)priv_data1,bx,by,bz,vol2,x0,y0,z0,nx,ny,nz);
						}
						auto stop = Time::now();
						fsec elapsed = start-stop; 
						double mcells_per_sec = (double)nx * (double)ny * (double)nz / (elapsed.count() * 1e6);
						double GB_per_sec = (double)sizeof(float) * (double)nx * (double)ny * (double)nz * 3.0 / (elapsed.count() * 1e9);
					
						if (!Check_Volume(vol,vol2,nx,ny,nz))
						{
							printf("\x1B[0m[\x1B[31mFailed!\x1B[0m]\n");
							copy_round_trip_passed = false;
						}
						else
						{					
							printf("\x1B[0m[\x1B[32mPassed!\x1B[0m] :: %6.3f secs - %.0f MCells/s - %.2f GB/s\n",elapsed.count(),mcells_per_sec,GB_per_sec);
						}
					}
				}
			}
		}
	}
	printf("\n");

	printf("8. Verify correctness of Global_RMS method...");  fflush(stdout);
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

	printf("9. Test throughput of Compress() method...\n");
	int nx3,ny3,nz3;
	float* vol3;
	Read_Raw_Volume("/cpfs/lfs02/ESDRD/tjhc/pressure_at_t=7512.bin",nx3,ny3,nz3,vol3);
	//Read_Raw_Volume("/cpfs/lfs01/ESDRD/tjhc/fdmod2/trunk/CvxCompress/empty.bin",nx3,ny3,nz3,vol3);
	unsigned long* compressed3;
	posix_memalign((void**)&compressed3, 64, (long)sizeof(float)*(long)nx3*(long)ny3*(long)nz3);
	for (int k = min_k;  k <= max_k-1;  ++k)
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
					const char* memtype = 0L;
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


					auto start = Time::now();
					double elapsed = 0.0;

					float ratio = 0.0f;
					int niter = 0;
					do
					{
						long compressed_length = 0l;
						ratio = Compress(scale,vol3,nx3,ny3,nz3,bx,by,bz,false,(unsigned int*)compressed3,compressed_length);
						auto stop = Time::now();
						++niter;
						fsec elapsed = stop-start;
						double mcells_per_sec = (double)niter * (double)nx3 * (double)ny3 * (double)nz3 / (elapsed.count() * 1e6);
						printf("\r\x1B[0m -> %3d x %3d x %3d (%s) %2d iterations - %6.3f secs - %.0f MCells/s - ratio %.2f:1",bx,by,bz,memtype,niter,elapsed.count(),mcells_per_sec,ratio);
						fflush(stdout);
					} while (elapsed < 10.0);
					printf("\n");
					//printf("%d iterations - %6.3f secs - %.0f MCells/s - ratio %.2f:1\n",niter,elapsed,mcells_per_sec,ratio);
				}
			}
		}
	}

	printf("10. Test throughput of Decompress() method...\n");
	for (int k = min_k;  k <= max_k-2;  ++k)
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
					float ratio = Compress(scale,vol3,nx3,ny3,nz3,bx,by,bz,false,(unsigned int*)compressed3,compressed_length3);

					const char* memtype = 0L;
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

					auto start = Time::now();
					double elapsed = 0.0;

					int niter = 0;
					do
					{
						int nx4, ny4, nz4;
						float* vol4;
						auto stop = Time::now();
						++niter;
						fsec elapsed = stop-start;
						double mcells_per_sec = (double)niter * (double)nx3 * (double)ny3 * (double)nz3 / (elapsed.count() * 1e6);
						printf("\r\x1B[0m -> %3d x %3d x %3d (%s) %2d iterations - %6.3f secs - %.0f MCells/s",bx,by,bz,memtype,niter,elapsed.count(),mcells_per_sec);
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

extern "C"
{
//
float
cvx_compress(
	float         scale,
	float        *vol,
	int           nx,
	int           ny,
	int           nz,
	int           bx,
	int           by,
	int           bz,
	unsigned int *compressed,
	long         *compressed_length)
{
	CvxCompress c;
	return c.Compress(scale, vol, nx, ny, nz, bx, by, bz, false, compressed, *compressed_length);
}

float* 
cvx_decompress_outofplace(
	int           *nx,
	int           *ny,
	int           *nz,
	unsigned int  *compressed,
	long           compressed_length)
{
	CvxCompress c;
	return c.Decompress(*nx, *ny, *nz, compressed, compressed_length);
}

void 
cvx_decompress_inplace(
	float         *vol,
	int            nx,
	int            ny,
	int            nz,
	unsigned int  *compressed,
	long           compressed_length)
{
	CvxCompress c;
	c.Decompress(vol, nx, ny, nz, compressed, compressed_length);
}
//
}
