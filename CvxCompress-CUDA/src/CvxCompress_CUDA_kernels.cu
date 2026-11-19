#include <stdio.h>
#include <cuda_runtime.h>
#include <Wavelet_Transform_16.cu>
#include <Run_Length_Encoding.cu>
#include <chrono>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#include <cmath>

#define WANT_BLOCK 1234

/*
 * Block compress subvolume using wavelet transform, quantization and run length encoding.
 * The compressed format is as follows:
 * 
 * (header)
 * int		origin (user defined, but for instance 1->source_forward, 2->receiver_reverse)
 * int		timestep
 * int		x0
 * int		y0
 * int		z0
 * int		x1
 * int		y1
 * int		z1
 * int		voi_x0
 * int		voi_y0
 * int		voi_z0
 * int		voi_x1
 * int		voi_y1
 * int		voi_z1
 * (block_header) * num_blocks
 * float	mean_value
 * float	multiplier
 * short	escape_length
 * short	payload_length
 * (compressed_bytes) * num_blocks
 * char		compressed
 *
 * Total number of blocks can be calculated from the subvolume dimensions in the following manner:
 *
 * num_blocks_x = (x1-x0+16)/16
 * num_blocks_y = (y1-y0+16)/16
 * num_blocks_z = (z1-z0+16)/16
 * num_blocks = num_blocks_x * num_blocks_y * num_blocks_z
 *
 * The header portion is of fixed size: total_header_size = (14*sizeof(int)) + num_blocks * (2*sizeof(float)+2*sizeof(short)).
 * The compressed portion is of varied size. Compressed blocks are packed tightly in ascending order along X-Y-Z.
 * The first compressed block will be [[x0,x0+15],[y0,y0+15],[z0,z0+15]], the second [[x0+16,x0+31],[y0,y0+15],[z0,z0+15]]
 *
 * This format allows individual blocks to be stored uncompressed when that is beneficial. The maximum compression size is
 * larger than the input size, hence storing uncompressed blocks can sometimes decrease the total compressed size. The block
 * header indicates if a block is stored uncompressed, escape_length + payload_length will be equal to 16384.
 */

#if __CUDA_ARCH__ == 370
__launch_bounds__(256,8)
#elif __CUDA_ARCH__ == 300
__launch_bounds__(256,5)
#endif
__global__ void cuCvxCompress_CompressSubvolume(
	int origin,
	int timestep,
	float* pq0,
	float* pq1,
	float scale,
	char* compressed_sparse,
	int sv_x0,
	int sv_y0,
	int sv_z0,
	int sv_y1,
	int sv_z1,
	int voi_x0,
	int voi_y0,
	int voi_z0,
	int voi_x1,
	int voi_y1,
	int voi_z1,
	float global_mean,
	float global_rms
	)
{
	__shared__ short buf[4262];
	// sv_x1 = sv_x0 + 15;
	// nx = 16;
	int ny = sv_y1 - sv_y0 + 1;
	int nz = sv_z1 - sv_z0 + 1;
	// nbx = 1
	int nby = (ny+15)/16;
	int nbz = (nz+15)/16;
	// ibx = 0
	int iby = blockIdx.y;
	int ibz = blockIdx.z;
	int y0 = sv_y0 + iby * 16;
	int z0 = sv_z0 + ibz * 16;
	int y1 = y0 + 15;
	int z1 = z0 + 15;
	int block_num = ibz*nby+iby;
	// use_global / blk_hdr_bytes are derived from the kernel parameter — no shared state needed.
	const bool use_global    = (global_rms > 0.0f);
	const int  blk_hdr_bytes = use_global ? 4 : 12;
	char* c_hdr = compressed_sparse + 64 + block_num * blk_hdr_bytes;
	char* c_blk = compressed_sparse + 64 + nby * nbz * blk_hdr_bytes + block_num * 16384;
	short thrid = threadIdx.z*16+threadIdx.x;
	if (blockIdx.y == 0 && blockIdx.z == 0 && thrid == 0)
	{
		int* header = (int*)compressed_sparse;
		header[ 0] = origin;
		header[ 1] = timestep;
		header[ 2] = sv_x0;
		header[ 3] = sv_y0;
		header[ 4] = sv_z0;
		header[ 5] = sv_x0+15;
		header[ 6] = sv_y1;
		header[ 7] = sv_z1;
		header[ 8] = voi_x0;
		header[ 9] = voi_y0;
		header[10] = voi_z0;
		header[11] = voi_x1;
		header[12] = voi_y1;
		header[13] = voi_z1;
		header[14] = use_global ? 1 : 0;  // 1=global RMS, 0=local RMS
		((float*)header)[15] = use_global ? (1.0f / (global_rms * scale)) : 0.0f;  // global multiplier
	}
	__syncthreads();
	if (((sv_x0 > voi_x1 || (sv_x0+15) < voi_x0)) || ((y0 > voi_y1 || y1 < voi_y0)) || ((z0 > voi_z1 || z1 < voi_z0)))
	{
		// block is outside of voi, generate bunch of zeros
		if (thrid == 0)
		{
			c_blk[0] = (char)RLESC3;
			c_blk[1] = (char)0;
			c_blk[2] = (char)16;  // rle = 16*256+0 = 4096
			if (use_global)
			{
				((short*)c_hdr)[0] = short(1);
				((short*)c_hdr)[1] = short(3);
			}
			else
			{
				((float*)c_hdr)[0] = 0.0f;
				((float*)c_hdr)[1] = 1.0f;
				((short*)c_hdr)[4] = short(1);
				((short*)c_hdr)[5] = short(3);
			}
		}
	}
	else
	{
		// input data contains interleaved P-Q wavefields in X-Z-Y order
		float v[16];
		int z = z0 + threadIdx.z;
		z = z < nz ? z : nz-1;
		float *inp = (threadIdx.x < 8) ? pq0 + z*8 + threadIdx.x : pq1 + z*8 + (threadIdx.x-8);
		for (int i = 0, yy = y0;  i < 16;  ++i,++yy) {
			yy = yy < ny ? yy : ny-1; 
			v[i]  = inp[yy*8*nz]; 
		}

		// scale data
		float mean {}, multiplier {};
		if (global_rms > 0.0f)
			_cuNormalizeGlobal(buf, scale, v, global_mean, global_rms, mean, multiplier);
		else
			_cuNormalize(buf, scale, v, mean, multiplier);
		//if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && thrid == 0) printf("_cuNormalize - mean=%e, multiplier=%e\n",mean,multiplier);
		//__syncthreads();
		if(multiplier < 0.0f || isnan(mean) || isnan(multiplier) || isinf(mean) || isinf(multiplier))
		{
			// input data contains denormals (NaN, Inf).
			// write raw data to output buffer.
			// output data is ordered X-Z-Y
			float* outp = (float*)c_blk;
			for (int i = 0;  i < 16;  ++i) outp[thrid+i*256] = v[i];
			if (thrid == 0)
			{
				if (use_global)
				{
					((short*)c_hdr)[0] = short(0);
					((short*)c_hdr)[1] = short(16384);
				}
				else
				{
					((float*)c_hdr)[0] =  0.0f;
					((float*)c_hdr)[1] = -1.0f;
					((short*)c_hdr)[4] = short(0);
					((short*)c_hdr)[5] = short(16384);
				}
			}
		}
		else
		{
			_cuDs79XYZ_16((float*)buf,v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10],v[11],v[12],v[13],v[14],v[15]);
			int rle_flags = 0;
			for (int i = 0;  i < 16;  ++i) rle_flags |= (((int)v[i] == 0) << i);
			__cuComputeRunLength4096(buf,rle_flags);
			int escape_length, payload_length;
			__cuComputeCompressedSize4096(buf,v,escape_length,payload_length);
			int compressed_size = escape_length + payload_length;
			if (compressed_size >= 16384)
			{
				// size increases after compression: store uncompressed after wavelet transform.
				__cuTranspose23((float*)buf,v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10],v[11],v[12],v[13],v[14],v[15]);
				// output data is ordered X-Z-Y
				float* outp = (float*)c_blk;
				for (int i = 0;  i < 16;  ++i) outp[thrid+i*256] = v[i];
				if (thrid == 0)
				{
					if (use_global)
					{
						((short*)c_hdr)[0] = short(0);
						((short*)c_hdr)[1] = short(16384);
					}
					else
					{
						((float*)c_hdr)[0] = mean;
						((float*)c_hdr)[1] = multiplier;
						((short*)c_hdr)[4] = short(0);
						((short*)c_hdr)[5] = short(16384);
					}
				}
			}
			else
			{
				// input data has been normalized at this point
				// output data is ordered X-Y-Z
				_cuEncode(buf,v,c_blk,escape_length);
				if (thrid == 0)
				{
					int block_length = (escape_length + payload_length + 3) & 0xFFFC;
					if (use_global)
					{
						((short*)c_hdr)[0] = (short)escape_length;
						((short*)c_hdr)[1] = (short)(block_length - escape_length);
					}
					else
					{
						((float*)c_hdr)[0] = mean;
						((float*)c_hdr)[1] = multiplier;
						((short*)c_hdr)[4] = escape_length;
						((short*)c_hdr)[5] = block_length - escape_length;
					}
					//if (block_num == WANT_BLOCK) printf("x0=%d, escape_length=%d, payload_length=%d, block_length=%d\n",sv_x0,escape_length,payload_length,block_length);
				}
			}
		}
	}
}

/*
 * Multiple compressed volumes in sparse format are densely packed into a single container before transfer to host.
 * The container format is as follows:
 * 
 * (header)
 * int		max_num_subvolumes
 * int		num_subvolumes
 * long		max_container_payload_size_in_bytes (excludes header)
 * (block_offset_header)*(max_num_subvolumes+1)
 * long		subvolume_byte_offset
 * (compressed_spare_format)*num_subvolumes
 */

/*
 * This method initializes a newly allocated container
 */
__global__ void cuCvxCompress_InitContainer(
	char* compressed_container,
	int max_num_subvolumes,
	long long max_container_payload_size_in_bytes
	)
{
	short num_threads = blockDim.z*blockDim.y*blockDim.x;
	short thrid = (threadIdx.z*blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x;
	int* container_header = (int*)compressed_container;
	long long* block_offsets = (long long*)(container_header+4);
	if (thrid == 0)
	{
		container_header[0] = max_num_subvolumes;
		container_header[1] = 0;
		((long long*)container_header)[1] = max_container_payload_size_in_bytes;
	}
	for (int i = thrid;  i <= max_num_subvolumes;  i+=num_threads)
	{
		if (i <= max_num_subvolumes) block_offsets[i] = 0l;
	}
}

/*
 * Set aside space for new subvolume in container header.
 * Must be launched with a single thread block with shape 256,1,1
 */
__global__ void cuCvxCompress_AllocNewSubvol(
	char* compressed_container,
	char* compressed_sparse
	)
{
	// container info
	int* container_header = (int*)compressed_container;
	//int max_num_subvolumes = container_header[0];
	int num_subvolumes = container_header[1];
	//long long max_container_payload_size_in_bytes = ((long long*)container_header)[1];
	long long* block_offsets = (long long*)(container_header+4);
	long long offset = block_offsets[num_subvolumes];
	// compressed sparse block
	int* subvol_header = (int*)compressed_sparse;
	int sv_x0 = subvol_header[2];
	int sv_y0 = subvol_header[3];
	int sv_z0 = subvol_header[4];
	int sv_x1 = subvol_header[5];
	int sv_y1 = subvol_header[6];
	int sv_z1 = subvol_header[7];
	int nbx = (sv_x1-sv_x0+16)/16;
	int nby = (sv_y1-sv_y0+16)/16;
	int nbz = (sv_z1-sv_z0+16)/16;
	int num_blocks = nbx*nby*nbz;
	int use_global = subvol_header[14];  // 1=global RMS (4-byte block hdrs), 0=local RMS (12-byte)
	int blk_hdr_bytes = use_global ? 4 : 12;
	char* block_header = (char*)(subvol_header + 16);  // subvolume header is always 64 bytes = 16 ints
	long long subvol_length = 0l;
	for (int i = threadIdx.x;  i < num_blocks;  i+=blockDim.x)
	{
		char* curr_block_header = block_header + blk_hdr_bytes * i;
		short escape_length, payload_length;
		if (use_global) {
			escape_length  = ((short*)curr_block_header)[0];
			payload_length = ((short*)curr_block_header)[1];
		} else {
			escape_length  = ((short*)curr_block_header)[4];
			payload_length = ((short*)curr_block_header)[5];
		}
		subvol_length += (long long)(escape_length + payload_length);  // guaranteed to be multiple of 4 bytes
	}
	// reduce(+:subvol_length)	
	// ..warp
	short thrx = threadIdx.x & 31;
#ifdef OLDCUDA
	subvol_length = __shfl(subvol_length,thrx+1,32) + subvol_length;
	subvol_length = __shfl(subvol_length,thrx+2,32) + subvol_length;
	subvol_length = __shfl(subvol_length,thrx+4,32) + subvol_length;
	subvol_length = __shfl(subvol_length,thrx+8,32) + subvol_length;
	subvol_length = __shfl(subvol_length,thrx+16,32) + subvol_length;
#else 
	subvol_length = __shfl_sync(0xFFFFFFFF,subvol_length,thrx+1,32) + subvol_length;
	subvol_length = __shfl_sync(0xFFFFFFFF,subvol_length,thrx+2,32) + subvol_length;
	subvol_length = __shfl_sync(0xFFFFFFFF,subvol_length,thrx+4,32) + subvol_length;
	subvol_length = __shfl_sync(0xFFFFFFFF,subvol_length,thrx+8,32) + subvol_length;
	subvol_length = __shfl_sync(0xFFFFFFFF,subvol_length,thrx+16,32) + subvol_length;
#endif
	int warp_num = threadIdx.x >> 5;
	int num_warps = blockDim.x >> 5;
	__shared__ long long buf[32];
	if (thrx == 0) buf[warp_num] = subvol_length;
	__syncthreads();
	if (threadIdx.x == 0)
	{
		subvol_length = buf[0];
		for (int i = 1;  i < num_warps;  ++i) subvol_length += buf[i];
		long long next_offset = offset + subvol_length + 64l + (long long)blk_hdr_bytes * num_blocks;
		container_header[1] = num_subvolumes+1;
		block_offsets[num_subvolumes+1] = next_offset;
		/*
		if (blockIdx.x < 20) printf("blockIdx=[%d,%d,%d] - Subvol %d/%d - x,y,z=[%d,%d],[%d,%d],[%d,%d] - nbx,nby,nbz=%d,%d,%d - offset,next_offset=%ld,%ld\n",
				blockIdx.x,blockIdx.y,blockIdx.z,num_subvolumes+1,max_num_subvolumes,
				sv_x0,sv_x1,sv_y0,sv_y1,sv_z0,sv_z1,
				nbx,nby,nbz,
				offset,next_offset);
		*/
	}
}

/*
 * Pack sparsely compressed volume into container.
 * Can only be called after AllocNewSubvol.
 */
#if __CUDA_ARCH__ == 370
__launch_bounds__(128,16)
#elif __CUDA_ARCH__ == 300
__launch_bounds__(128,16)
#endif
__global__ void cuCvxCompress_PackCompressedSparse(
	char* compressed_container,
	char* compressed_sparse
	)
{
	// container info
	int* container_header = (int*)compressed_container;
	int max_num_subvolumes = container_header[0];
	long long max_container_payload_size_in_bytes = ((long long*)container_header)[1];
	int curr_subvolume = container_header[1] - 1;
	long long* block_offsets = (long long*)(container_header+4);
	long long offset = block_offsets[curr_subvolume];
	// compressed sparse block
	int use_global = ((int*)compressed_sparse)[14];
	int blk_hdr_bytes = use_global ? 4 : 12;
	char* block_header = compressed_sparse + 64;  // subvolume header is always 64 bytes
	int subvol_block_offset = 0;
	// each thread does a partial sum
	for (int i = threadIdx.x;  i < blockIdx.x;  i+=blockDim.x)
	{
		char* curr_block_header = block_header + blk_hdr_bytes * i;
		short escape_length, payload_length;
		if (use_global) {
			escape_length  = ((short*)curr_block_header)[0];
			payload_length = ((short*)curr_block_header)[1];
		} else {
			escape_length  = ((short*)curr_block_header)[4];
			payload_length = ((short*)curr_block_header)[5];
		}
		int curr_block_length = (int)escape_length + (int)payload_length;
		subvol_block_offset += curr_block_length;
	}
	char* last_block_header_p = block_header + blk_hdr_bytes * blockIdx.x;
	int subvol_block_length;
	if (use_global)
		subvol_block_length = ((short*)last_block_header_p)[0] + ((short*)last_block_header_p)[1];
	else
		subvol_block_length = ((short*)last_block_header_p)[4] + ((short*)last_block_header_p)[5];
	// reduce(+:subvol_block_offset)
	// ..warp reduction with __shfl
	short thrx = threadIdx.x & 31;
#ifdef OLDCUDA
	subvol_block_offset = __shfl(subvol_block_offset,thrx+1,32) + subvol_block_offset;
	subvol_block_offset = __shfl(subvol_block_offset,thrx+2,32) + subvol_block_offset;
	subvol_block_offset = __shfl(subvol_block_offset,thrx+4,32) + subvol_block_offset;
	subvol_block_offset = __shfl(subvol_block_offset,thrx+8,32) + subvol_block_offset;
	subvol_block_offset = __shfl(subvol_block_offset,thrx+16,32) + subvol_block_offset;
#else
	subvol_block_offset = __shfl_sync(0xFFFFFFFF,subvol_block_offset,thrx+1,32) + subvol_block_offset;
	subvol_block_offset = __shfl_sync(0xFFFFFFFF,subvol_block_offset,thrx+2,32) + subvol_block_offset;
	subvol_block_offset = __shfl_sync(0xFFFFFFFF,subvol_block_offset,thrx+4,32) + subvol_block_offset;
	subvol_block_offset = __shfl_sync(0xFFFFFFFF,subvol_block_offset,thrx+8,32) + subvol_block_offset;
	subvol_block_offset = __shfl_sync(0xFFFFFFFF,subvol_block_offset,thrx+16,32) + subvol_block_offset;
#endif
	int warp_num = threadIdx.x >> 5;
	int num_warps = blockDim.x >> 5;
	// ..thread block reduction with shared memory
	__shared__ int buf[64];
	if (thrx == 0)
	{
		buf[warp_num] = subvol_block_offset;
	}
	__syncthreads();
	subvol_block_offset = buf[0];
	for (int i = 1;  i < num_warps;  ++i)
	{
		subvol_block_offset += buf[i];
	}
	long long subvol_first_byte_index = offset + 64l + (long long)blk_hdr_bytes * gridDim.x + (long long)subvol_block_offset;
	long long subvol_last_byte_index = subvol_first_byte_index + subvol_block_length;  // +1
	subvol_last_byte_index = min(subvol_last_byte_index,max_container_payload_size_in_bytes);
	long long clipped_subvol_block_length = subvol_last_byte_index - subvol_first_byte_index;
	if (clipped_subvol_block_length == subvol_block_length)
	{
		int subvol_block_num_words = subvol_block_length/4;  // compute subvol_block_length in number of int.
		int *dst_w = (int*)(compressed_container + 16l + 8l * (long long)(max_num_subvolumes + 1) + subvol_first_byte_index);
		int *src_w = (int*)(compressed_sparse + 64l + (long long)blk_hdr_bytes * gridDim.x + blockIdx.x * 16384l);
		for (int j = threadIdx.x;  j < subvol_block_num_words;  j+=blockDim.x) dst_w[j] = src_w[j];
		if (blockIdx.x == 0)
		{
			// copy subvolume header (16 ints) + block headers
			dst_w = (int*)(compressed_container + 16l + 8l * (long long)(max_num_subvolumes + 1) + offset);
			src_w = (int*)compressed_sparse;
			int hdr_word_count = 16 + (blk_hdr_bytes / 4) * gridDim.x;  // 16+gridDim.x (global) or 16+3*gridDim.x (local)
			for (int j = threadIdx.x;  j < hdr_word_count;  j+=blockDim.x) dst_w[j] = src_w[j];
		}
	}
}

__device__ void _cuCvxCompress_DecompressSubvolume(
	short* buf,
	int thrid,
        char* compressed,
        float* decompressed
        )
{
	long long icompressed = (long long)compressed;
	if (thrid == 0 && (icompressed&3) != 0) printf("compressed volume is misaligned\n");
	int* header = (int*)compressed;
	int sv_x0 = header[ 2];
	int sv_y0 = header[ 3];
	int sv_z0 = header[ 4];
	int sv_x1 = header[ 5];
	int sv_y1 = header[ 6];
	int sv_z1 = header[ 7];
	int voi_x0 = header[ 8];
	int voi_y0 = header[ 9];
	int voi_z0 = header[10];
	int voi_x1 = header[11];
	int voi_y1 = header[12];
	int voi_z1 = header[13];
	int nx = sv_x1 - sv_x0 + 1;
	//int ny = sv_y1 - sv_y0 + 1;
	int nz = sv_z1 - sv_z0 + 1;
	int nbx = gridDim.x;
	int nby = gridDim.y;
	int nbz = gridDim.z;
	int num_blocks = nbx*nby*nbz;
	int ibx = blockIdx.x;
	int iby = blockIdx.y;
	int ibz = blockIdx.z;
	//if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && thrid == 0) printf("&&&&&  nbx,nby,nbz=%d,%d,%d - ibx,iby,ibz=%d,%d,%d\n",nbx,nby,nbz,ibx,iby,ibz);
	int x0 = sv_x0 + ibx * 16;
	int y0 = sv_y0 + iby * 16;
	int z0 = sv_z0 + ibz * 16;
	int block_num = (ibz*nby+iby)*nbx+ibx;
	int use_global        = header[14];  // 1=global RMS, 0=local RMS
	int blk_hdr_bytes     = use_global ? 4 : 12;
	float global_mult     = use_global ? ((float*)header)[15] : 0.0f;
	int thread_block_offset = 0;
	// determine block offset
	for (int iBlk = thrid;  iBlk < block_num;  iBlk+=256)  // assumes threadblock has 256 threads
	{
		if (iBlk < block_num)
		{
			char* curr_hdr = (char*)(compressed + 64) + iBlk * blk_hdr_bytes;
			int curr_block_length;
			if (use_global)
				curr_block_length = ((short*)curr_hdr)[0] + ((short*)curr_hdr)[1];
			else
				curr_block_length = ((short*)curr_hdr)[4] + ((short*)curr_hdr)[5];
			thread_block_offset += curr_block_length;
		}
	}
	int warp_thrid = thrid & 31;
#ifdef OLDCUDA
	int warp_block_offset = thread_block_offset + __shfl(thread_block_offset,warp_thrid+1,32);
	warp_block_offset = warp_block_offset + __shfl(warp_block_offset,warp_thrid+2,32);
	warp_block_offset = warp_block_offset + __shfl(warp_block_offset,warp_thrid+4,32);
	warp_block_offset = warp_block_offset + __shfl(warp_block_offset,warp_thrid+8,32);
	warp_block_offset = warp_block_offset + __shfl(warp_block_offset,warp_thrid+16,32);
#else
	int warp_block_offset = thread_block_offset + __shfl_sync(0xFFFFFFFF,thread_block_offset,warp_thrid+1,32);
	warp_block_offset = warp_block_offset + __shfl_sync(0xFFFFFFFF,warp_block_offset,warp_thrid+2,32);
	warp_block_offset = warp_block_offset + __shfl_sync(0xFFFFFFFF,warp_block_offset,warp_thrid+4,32);
	warp_block_offset = warp_block_offset + __shfl_sync(0xFFFFFFFF,warp_block_offset,warp_thrid+8,32);
	warp_block_offset = warp_block_offset + __shfl_sync(0xFFFFFFFF,warp_block_offset,warp_thrid+16,32);
#endif
	int warp_num = thrid >> 5;
	if (warp_thrid == 0) ((int*)buf)[warp_num] = warp_block_offset;
	__syncthreads();
	int block_offset = 0;
	for (int i = 0;  i < 8;  ++i) block_offset += ((int*)buf)[i];  // assumes threadblock has 256 threads, 8 warps * 32.
	char* c_hdr = (char*)(compressed + 64) + block_num * blk_hdr_bytes;
	float mean, multiplier;
	int escape_length, payload_length;
	if (use_global)
	{
		mean = 0.0f;
		multiplier = global_mult;
		escape_length = (int)((short*)c_hdr)[0];
		payload_length = (int)((short*)c_hdr)[1];
	}
	else
	{
		mean = ((float*)c_hdr)[0];
		multiplier = ((float*)c_hdr)[1];
		escape_length = (int)((short*)c_hdr)[4];
		payload_length = (int)((short*)c_hdr)[5];
	}
	char* c_blk = (char*)(compressed + 64) + blk_hdr_bytes * num_blocks + block_offset;
	//if (block_num == 1 && thrid == 0) printf("sv::x=[%d,%d],y=[%d,%d],z=[%d,%d] - num_blocks=%d, block_offset=%d\n",sv_x0,sv_x1,sv_y0,sv_y1,sv_z0,sv_z1,num_blocks,block_offset);
	__syncthreads();
	float v[16];
	if (escape_length == 1)
	{
		// all zeros block
		for (int i = 0;  i < 16;  ++i) v[i] = 0.0f;
	}
	else if (escape_length + payload_length >= 16384)
	{
		// block was stored uncompressed
		for (int i = 0;  i < 16;  ++i) v[i] = ((float*)c_blk)[thrid+i*16];
	}
	else
	{
		__cuDecode(buf,c_blk,escape_length,v);
	}
	if (multiplier > 0.0f && !(escape_length == 1))
	{
		// block was stored after forward wavelet transform was applied
		float demultiplier = 1.0f / multiplier;
		_cuUs79XYZ_16((float*)buf,v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10],v[11],v[12],v[13],v[14],v[15]);
		for (int i = 0;  i < 16;  ++i) v[i] = v[i] * demultiplier + mean;
	}
	int zz = z0 + threadIdx.z;
	if (zz <= sv_z1)
	{
		int xx = x0 + threadIdx.x;
		bool inside_voi_xz = (xx >= voi_x0 && xx <= voi_x1) && (zz >= voi_z0 && zz <= voi_z1);
		float* dst = decompressed + (blockIdx.y*16*nz+(blockIdx.z*16+threadIdx.z))*nx+(blockIdx.x*16+threadIdx.x);
		for (int i = 0;  i < 16;  ++i)
		{
			int yy = y0 + i;
			if (yy <= sv_y1)
			{
				dst[i*nx*nz] = (inside_voi_xz && (yy >= voi_y0 && yy <= voi_y1)) ? v[i] : 0.0f;
			}
		}
	}
}

#if __CUDA_ARCH__ == 370
__launch_bounds__(256,8)
#elif __CUDA_ARCH__ == 300
__launch_bounds__(256,5)
#endif
__global__ void cuCvxCompress_DecompressSubvolume(
        char* compressed,
        float* decompressed
        )
{
	__shared__ short buf[4262];
	int thrid = threadIdx.z*16+threadIdx.x;
	_cuCvxCompress_DecompressSubvolume(buf,thrid,compressed,decompressed);
}

#if __CUDA_ARCH__ == 370
__launch_bounds__(256,8)
#elif __CUDA_ARCH__ == 300
__launch_bounds__(256,5)
#endif
__global__ void cuCvxCompress_DecompressSubvolume_In_Container(
	char* compressed_container,
	int origin,
	int timestep,
	int x0,
	float* decompressed,
	bool crash_on_error
	)
{
	__shared__ short buf[4262];
	int thrid = (threadIdx.z*blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x;
	// scan subvolume headers to find correct subvolume
	int* container_header = (int*)compressed_container;
	int max_num_subvolumes = container_header[0];
	int num_subvolumes = container_header[1];
	long long max_container_payload_size_in_bytes = ((long long*)container_header)[1];
	long long* block_offsets = (long long*)(container_header+4);
	if (thrid == 0) ((char**)buf)[0] = 0l;
	__syncthreads();
	for (int i = thrid;  i < num_subvolumes;  i+=blockDim.x*blockDim.y*blockDim.z)
	{
		if (block_offsets[i+1] < max_container_payload_size_in_bytes)  // ensure last byte is present
		{
			int *subvol_header = (int*)(compressed_container + 16l + 8l * (max_num_subvolumes + 1) + block_offsets[i]);
			int sv_x0 = subvol_header[ 2];
			int sv_origin = subvol_header[ 0];
			int sv_timestep = subvol_header[ 1];
			//int block_num = blockIdx.z*gridDim.y+blockIdx.y*gridDim.x+blockIdx.x;
			//if (block_num == 1) printf("Subvolume index %d/%d :: origin=%d, timestep=%d, x0=%d, block_offset=%ld\n",i,max_num_subvolumes,sv_origin,sv_timestep,sv_x0,block_offsets[i]);
			if (sv_x0 == x0 && sv_origin == origin && sv_timestep == timestep)
			{
				((char**)buf)[0] = (char*)subvol_header;
				//if (thrid == 0 && block_num == WANT_BLOCK) printf("cuCvxCompress_DecompressSubvolume_In_Container(x0=%d) - subvolume_offset=%ld\n",x0,8+8*(max_num_subvolumes+1)+block_offsets[i]);
				//if (block_num == 1) printf("Found subvolume origin=%d, timestep=%d, x0=%d at subvolume index %d with block offset %ld\n",origin,timestep,x0,i,block_offsets[i]);
			}
		}
	}
	__syncthreads();
	char* compressed_subvolume = ((char**)buf)[0];
	__syncthreads();
	// this will trigger illegal memory access if subvolume was not found and crash_on_error = true
	if (crash_on_error || compressed_subvolume != 0l) _cuCvxCompress_DecompressSubvolume(buf,thrid,compressed_subvolume,decompressed);
}

#if __CUDA_ARCH__ == 370
__launch_bounds__(256,8)
#elif __CUDA_ARCH__ == 300
__launch_bounds__(256,5)
#endif
__global__ void cuCvxCompress_DecompressIndexedSubvolume_In_Container(
	char* compressed_container,
	int subvolume_index,
	float* decompressed,
	bool crash_on_error
	)
{
	__shared__ short buf[4262];
	int thrid = (threadIdx.z*blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x;
	// scan subvolume headers to find correct subvolume
	int* container_header = (int*)compressed_container;
	int max_num_subvolumes = container_header[0];
	int num_subvolumes = container_header[1];
	//long long max_container_payload_size_in_bytes = ((long long*)container_header)[1];
	long long* subvolume_offsets = (long long*)(container_header+4);
	char* compressed_subvolume = (subvolume_index >= 0 && subvolume_index < num_subvolumes) ? (char*)(compressed_container + 16l + 8l * (max_num_subvolumes + 1) + subvolume_offsets[subvolume_index]) : 0l;
	//if (thrid == 0) printf("cuCvxCompress_DecompressIndexedSubvolume_In_Container(subvolume_index=%d) - num_subvolumes=%d/%d, subvolume_offset[%d]=%ld\n",subvolume_index,num_subvolumes,max_num_subvolumes,subvolume_index,subvolume_offsets[subvolume_index]);
	// this will trigger illegal memory access if subvolume was not found and crash_on_error = true
	if (crash_on_error || compressed_subvolume != 0l) _cuCvxCompress_DecompressSubvolume(buf,thrid,compressed_subvolume,decompressed);
}

/*
 * Copy 8 consecutive x-planes (starting at x0) from a device volume into a
 * replication-padded PQ slab buffer.
 * Source layout: vol[(iz * ny + iy) * nx + ix]  (ZYX).
 * Dest  layout: d_pq[(iy * nz8 + iz) * 8 + ix]  (YZX, stride-8).
 * Out-of-bounds source elements (padding region) are filled by clamping to the
 * nearest valid voxel.
 */
__global__ void cuCvxCompress_CopyToBuffer(
	const float* __restrict__ vol,
	int nx, 
	int ny, 
	int nz,
	int ny8, 
	int nz8,
	float* d_pq,
	int x0)
{
	int ix = threadIdx.x;                               
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if (iy >= ny8 || iz >= nz8) return;

	int gx = min(x0 + ix, nx - 1);
	int cy = min(iy, ny - 1);
	int cz = min(iz, nz - 1);
	float val = vol[((long long)cz * ny + cy) * nx + gx];

	d_pq[((long long)iy * nz8 + iz) * 8 + ix] = val;
}

/*
 * Copy 16 consecutive x-planes (starting at x0) from a device decomp slab into a
 * device volume.
 * Source layout: d_decomp[(iy * nz8 + iz) * sv_nx + ix]  (YZX, stride-16).
 * Dest  layout: vol[(iz * ny + iy) * nx + gx]  (ZYX).
 * Out-of-bounds source elements (padding region) are written as 0.
 */
__global__ void cuCvxCompress_CopyFromBuffer(
	float* __restrict__ vol,
	int nx, 
	int ny, 
	int nz,
	int ny8, 
	int nz8,
	const float* __restrict__ d_decomp,
	int x0)
{
	int ix = threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if (iy >= ny8 || iz >= nz8) return;

	int gx = x0 + ix;
	if (gx >= nx || iy >= ny || iz >= nz) return;

	float val = d_decomp[((long long)iy * nz8 + iz) * 16 + ix];
	vol[((long long)iz * ny + iy) * nx + gx] = val;
}

/*
 * Compute the global mean and RMS of a device volume using Thrust.
 * Two-pass reduction: global mean is computed first, then the RMS is computed
 * When use_local_RMS is true this is a no-op (outputs remain 0).
 */

struct _square
{
	__device__ float operator()(float x) const { return x * x; }
};

void CUDA_ComputeGlobalMeanRMS(
	cudaStream_t stream,
	const float* vol,
	long long N,
	float& global_mean,
	float& global_rms
	)
{
	thrust::device_ptr<const float> d_ptr(vol);

	const float sum1 = thrust::reduce(thrust::cuda::par.on(stream), d_ptr, d_ptr + N, 0.0f, thrust::plus<float>());
	global_mean = sum1 / static_cast<float>(N);
	
	const float sum2 = thrust::transform_reduce(thrust::cuda::par.on(stream), d_ptr, d_ptr + N, _square{}, 0.0f, thrust::plus<float>());
	global_rms = sqrtf(sum2 / static_cast<float>(N));
}