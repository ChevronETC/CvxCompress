#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>
#include "Common/gpuAssert.h"
#include "CvxCompress_CUDA.hxx"
#include "CvxCompress_CUDA_helpers.cuh"

CvxCompress_CUDA::CvxCompress_CUDA()
	:config_()
	{}

CvxCompress_CUDA::~CvxCompress_CUDA(){}

void CvxCompress_CUDA::InitCompress(int nx, int ny, int nz)
{

	int nx8 = ((nx + 7) / 8) * 8;
	int ny8 = ((ny + 7) / 8) * 8;
	int nz8 = ((nz + 7) / 8) * 8;

    // Already allocated for same dimensions - return
    if (config_.d_pq0_ && config_.nx8_ == nx8 && config_.ny8_ == ny8 && config_.nz8_ == nz8)
        return;
	
	config_.nx_ = nx;
	config_.ny_ = ny;
	config_.nz_ = nz;
	config_.nx8_ = nx8;
	config_.ny8_ = ny8;
	config_.nz8_ = nz8;
	config_.nbx_ = config_.nx8_ / 8;

	long long blocks_per_slice = ((config_.ny8_+15)/16) * ((config_.nz8_+15)/16);
	config_.max_compressed_sparse_size_ = 64LL + 12LL*blocks_per_slice + 16384LL*blocks_per_slice;
	// Each ibx iteration adds 2 subvolumes; loop runs ceil(nbx_/4) times
	config_.max_sparse_subvolumes_ = ((config_.nbx_ + 3) / 4) * 2; 
	config_.max_container_payload_size_ = config_.max_sparse_subvolumes_ * config_.max_compressed_sparse_size_;
	config_.max_compressed_size_ = 16LL + 8LL*(config_.max_sparse_subvolumes_+1) + config_.max_container_payload_size_;

    // Free existing allocations if resizing
	if (config_.d_pq3_) gpuErrchk( cudaFree(config_.d_pq3_) );
	if (config_.d_pq2_) gpuErrchk( cudaFree(config_.d_pq2_) );
	if (config_.d_pq1_) gpuErrchk( cudaFree(config_.d_pq1_) );
	if (config_.d_pq0_) gpuErrchk( cudaFree(config_.d_pq0_) );
    if (config_.d_compressed_sparse_23_) gpuErrchk( cudaFree(config_.d_compressed_sparse_23_) );
	if (config_.d_compressed_sparse_01_) gpuErrchk( cudaFree(config_.d_compressed_sparse_01_) );
	if (config_.h_compressed_)
	{
		gpuErrchk(cudaHostUnregister(config_.h_compressed_));
		free(config_.h_compressed_);
	}

    // Allocate
	config_.slice_size_ = (long long)8 * config_.ny8_ * config_.nz8_;
	// Device PQ buffers
	gpuErrchk( cudaMalloc((void**)&config_.d_pq0_, sizeof(float)*config_.slice_size_) );
	gpuErrchk( cudaMalloc((void**)&config_.d_pq1_, sizeof(float)*config_.slice_size_) );
	gpuErrchk( cudaMalloc((void**)&config_.d_pq2_, sizeof(float)*config_.slice_size_) );
	gpuErrchk( cudaMalloc((void**)&config_.d_pq3_, sizeof(float)*config_.slice_size_) );
	// Sparse compressed staging buffers
	gpuErrchk( cudaMalloc((void**)&config_.d_compressed_sparse_01_, config_.max_compressed_sparse_size_) );
	gpuErrchk( cudaMemset(config_.d_compressed_sparse_01_, 0, config_.max_compressed_sparse_size_) );
	gpuErrchk( cudaMalloc((void**)&config_.d_compressed_sparse_23_, config_.max_compressed_sparse_size_) );
	gpuErrchk( cudaMemset(config_.d_compressed_sparse_23_, 0, config_.max_compressed_sparse_size_) );
	// Pinned host mirror of the container header (the only part copied back)
	const long long container_header_size = 16LL + 8LL*(config_.max_sparse_subvolumes_+1);
	posix_memalign((void**)&config_.h_compressed_, 64, container_header_size);
	gpuErrchk( cudaHostRegister(config_.h_compressed_, container_header_size, cudaHostRegisterDefault) );
}

void CvxCompress_CUDA::InitDecompress(int nx, int ny, int nz)
{
	int nx8 = ((nx + 7) / 8) * 8;
	int ny8 = ((ny + 7) / 8) * 8;
	int nz8 = ((nz + 7) / 8) * 8;


	// Already allocated for same dimensions - return
	if (config_.d_decomp_01_ && config_.nx8_ == nx8 && config_.ny8_ == ny8 && config_.nz8_ == nz8)
		return;

	config_.nx_ = nx;
	config_.ny_ = ny;
	config_.nz_ = nz;
	config_.nx8_ = nx8;
	config_.ny8_ = ny8;
	config_.nz8_ = nz8;
	config_.nbx_ = config_.nx8_ / 8;

	// Free existing allocations if resizing
	if (config_.d_decomp_23_) gpuErrchk( cudaFree(config_.d_decomp_23_) );
	if (config_.d_decomp_01_) gpuErrchk( cudaFree(config_.d_decomp_01_) );

	// Allocate decompression buffers
	gpuErrchk( cudaMalloc((void**)&config_.d_decomp_01_, sizeof(float)*16*config_.ny8_*config_.nz8_) );
	gpuErrchk( cudaMalloc((void**)&config_.d_decomp_23_, sizeof(float)*16*config_.ny8_*config_.nz8_) );
}

long long CvxCompress_CUDA::GetMaxCompressedSize(int nx, int ny, int nz)
{
	const int nx8 = ((nx + 7) / 8) * 8;
	const int ny8 = ((ny + 7) / 8) * 8;
	const int nz8 = ((nz + 7) / 8) * 8;
	const int nbx = nx8 / 8;
	const long long blocks_per_slice = ((long long)(ny8 + 15) / 16) * ((long long)(nz8 + 15) / 16);
	const long long max_sparse_subvolumes = (long long)((nbx + 3) / 4) * 2;
	const long long max_compressed_sparse_size = 64LL + 12LL * blocks_per_slice + 16384LL * blocks_per_slice;
	const long long max_container_payload_size = max_sparse_subvolumes * max_compressed_sparse_size;
	return 16LL + 8LL * (max_sparse_subvolumes + 1) + max_container_payload_size;
}

float CvxCompress_CUDA::Compress(
	float scale,
	float* vol,
	int nx,
	int ny,
	int nz,
	int bx,
	int by,
	int bz,
	bool use_local_RMS,
	char* compressed,
	long& compressed_length 
	)
{
	return Compress(scale, vol, nx, ny, nz, use_local_RMS, compressed, compressed_length, 0);
}

float CvxCompress_CUDA::Compress(
		float scale,
		float* vol,
		int nx,
		int ny,
		int nz,
		int bx,
		int by,
		int bz,
		char* compressed,
		long& compressed_length 
	)
{
	return Compress(scale, vol, nx, ny, nz, true, compressed, compressed_length, 0);
}

float CvxCompress_CUDA::Compress(
		float scale,
		float* vol,
		int nx,
		int ny,
		int nz,
		bool use_local_RMS,
		char* compressed,
		long& compressed_length,
		int voi_pad
	)
{
	// Synchronous compression: enqueue all GPU work (compression kernels and the
	// container-header D2H copy) on stream_, then block once and parse the size.
	EnqueueCompress_(scale, vol, nx, ny, nz, use_local_RMS, compressed, voi_pad);
	gpuErrchk( cudaStreamSynchronize(stream_) );
	return ParseCompressedLength_(compressed_length, use_local_RMS);
}

void CvxCompress_CUDA::CompressAsync(
		float scale,
		float* vol,
		int nx,
		int ny,
		int nz,
		int bx,
		int by,
		int bz,
		bool use_local_RMS,
		char* compressed
	)
{
	// Asynchronous enqueue only. The compressed size is not available until the
	// caller synchronizes stream_ and calls GetCompressedSize().
	EnqueueCompress_(scale, vol, nx, ny, nz, use_local_RMS, compressed, 0);
}

float CvxCompress_CUDA::GetCompressedSize(
		const char* compressed,
		long& compressed_length
	)
{
	// The container header was already copied to the pinned host mirror by the
	// preceding CompressAsync() (enqueued on stream_, pipelined behind the
	// compression kernels). It is valid to read here provided the caller has
	// synchronized the plan's stream.
	(void) compressed;
	return ParseCompressedLength_(compressed_length);
}

void CvxCompress_CUDA::EnqueueCompress_(
		float scale,
		float* vol,
		int nx,
		int ny,
		int nz,
		bool use_local_RMS,
		char* compressed,
		int voi_pad
	)
{
	// Initialize compression buffers
	InitCompress(nx, ny, nz);

	config_.scale_ = scale;

	// VOI defaults to the full volume minus padding.
	config_.voi_pad_ = voi_pad;
	config_.voi_x0_ = voi_pad;
	config_.voi_x1_ = nx - 1 - voi_pad;   // internal X range: [0, nx_-1]
	config_.voi_y0_ = voi_pad;
	config_.voi_y1_ = ny - 1 - voi_pad;
	config_.voi_z0_ = voi_pad;
	config_.voi_z1_ = nz - 1 - voi_pad;   // internal Z range: [0, nz_-1]

	// Must be called before the compression loop
	CUDA_CvxCompress_InitContainer(stream_, compressed, config_.max_sparse_subvolumes_, config_.max_container_payload_size_);

	// Compute global mean and RMS once over the full volume when requested.
	// global_rms == 0 tells the compress kernel to fall back to local per-block RMS.
	float global_mean = 0.0f, global_rms = 0.0f;
	if (!use_local_RMS)
	{
		const long long N = (long long)nx * ny * nz;
		CUDA_ComputeGlobalMeanRMS(stream_, vol, N, global_mean, global_rms);
	}

	for (int ibx = 0;  ibx < config_.nbx_;  ibx += 4)
        {
            int x0 = ibx * 8;
			// Copy 4 slabs (each 8 x-planes) from device volume into PQ buffers
			CUDA_Copy_To_Buffer(stream_, vol, config_.nx_, config_.ny_, config_.nz_, config_.ny8_, config_.nz8_, config_.d_pq0_, x0);
			CUDA_Copy_To_Buffer(stream_, vol, config_.nx_, config_.ny_, config_.nz_, config_.ny8_, config_.nz8_, config_.d_pq1_, x0 + 8);
			CUDA_Copy_To_Buffer(stream_, vol, config_.nx_, config_.ny_, config_.nz_, config_.ny8_, config_.nz8_, config_.d_pq2_, x0 + 16);
			CUDA_Copy_To_Buffer(stream_, vol, config_.nx_, config_.ny_, config_.nz_, config_.ny8_, config_.nz8_, config_.d_pq3_, x0 + 24);

            // Compress Batch
            CUDA_CvxCompress_CompressSubvolume_To_Container(
                stream_, 1, 12,
                config_.d_pq0_, config_.d_pq1_, config_.scale_,
                config_.d_compressed_sparse_01_, compressed,
                x0, 0, 0, config_.ny8_-1, config_.nz8_-1,
                config_.voi_x0_, config_.voi_y0_, config_.voi_z0_, config_.voi_x1_, config_.voi_y1_, config_.voi_z1_, global_mean, global_rms);
            CUDA_CvxCompress_CompressSubvolume_To_Container(
                stream_, 1, 12,
                config_.d_pq2_, config_.d_pq3_, config_.scale_,
                config_.d_compressed_sparse_23_, compressed,
                x0+16, 0, 0, config_.ny8_-1, config_.nz8_-1,
                config_.voi_x0_, config_.voi_y0_, config_.voi_z0_, config_.voi_x1_, config_.voi_y1_, config_.voi_z1_, global_mean, global_rms);

    }

	// Enqueue the container-header D2H copy right after the compression kernels
	// so it pipelines on stream_ without requiring a separate synchronization.
	long long container_header_size = 16LL + 8LL*(config_.max_sparse_subvolumes_+1);
	gpuErrchk( cudaMemcpyAsync(config_.h_compressed_, compressed, container_header_size, cudaMemcpyDeviceToHost, stream_) );
}

float CvxCompress_CUDA::ParseCompressedLength_(
		long& compressed_length,
		bool use_local_RMS
	)
{
	// Parse the container header that was already copied to the pinned host
	// mirror (config_.h_compressed_) by EnqueueCompress_. The caller MUST have
	// synchronized stream_ before calling this so the copy has completed.
	long long container_header_size = 16LL + 8LL*(config_.max_sparse_subvolumes_+1);

	int* container_header = (int*)config_.h_compressed_;
	long long* subvolume_offsets = (long long*)(container_header+4);
	int num_subvolumes = container_header[1];
	long long max_container_bytes = ((long long*)container_header)[1] + container_header_size;
	long long container_payload = subvolume_offsets[num_subvolumes];
	long long container_size = container_header_size + container_payload;

	if (container_size > max_container_bytes)
	{
		printf("Error! Container size %lld exceeds maximum size of %lld bytes.\n", container_size, max_container_bytes);
		compressed_length = -1;
		return -1.0;
	}
	compressed_length = container_size;

	double ratio = (double) config_.nx_ * (double) config_.ny_ * (double)config_.nz_ * (double)sizeof(float) / (double) compressed_length;

#ifdef PRINT_COMPRESSION_BREAKDOWN
	{
		long long uncompressed_bytes = (long long)config_.nx_ * config_.ny_ * config_.nz_ * sizeof(float);
		long long blocks_per_subvol  = ((config_.ny8_ + 15) / 16) * ((config_.nz8_ + 15) / 16);
		long long blk_hdr_sz         = use_local_RMS ? 12LL : 4LL;
		long long subvol_overhead    = num_subvolumes * (64LL + blk_hdr_sz * blocks_per_subvol);
		long long format_overhead    = container_header_size + subvol_overhead;
		long long pure_rle_payload   = container_size - format_overhead;
		double payload_ratio = (pure_rle_payload > 0) ? (double)uncompressed_bytes / (double)pure_rle_payload : 0.0;
		printf("[GPU] Post RLE Compression Ratio: %f\n", payload_ratio);
		printf("[GPU] Post RLE Compressed Bytes: %f\n",  (double)pure_rle_payload);
		printf("[GPU] Final Compressed Bytes: %f\n",     (double)container_size);
	}
#endif

	return (float) ratio;
}

void CvxCompress_CUDA::Decompress(
		float* vol,
		int nx,
		int ny,
		int nz,
		char* compressed,
	long compressed_length 
)
{
	// Initialize decompression buffers
	InitDecompress(nx, ny, nz);

	for (int ibx = 0;  ibx < config_.nbx_;  ibx += 4)
	{
		int x0 = ibx * 8;
		// Decompress Batch
		CUDA_CvxCompress_DecompressSubvolume_In_Container(
			stream_, compressed, 1, 12,
			x0, 0, 0, config_.ny8_-1, config_.nz8_-1,
			config_.d_decomp_01_, false);
		CUDA_CvxCompress_DecompressSubvolume_In_Container(
			stream_, compressed, 1, 12,
			x0+16, 0, 0, config_.ny8_-1, config_.nz8_-1,
			config_.d_decomp_23_, false);
		// cudaDeviceSynchronize();

		// D2D Copy decompressed slabs back into the device volume
		CUDA_Copy_From_Buffer(stream_, vol, nx, ny, nz, config_.ny8_, config_.nz8_, config_.d_decomp_01_, x0);
		CUDA_Copy_From_Buffer(stream_, vol, nx, ny, nz, config_.ny8_, config_.nz8_, config_.d_decomp_23_, x0+16);
	}

}

void CvxCompress_CUDA::SetStream(cudaStream_t stream)
{
	stream_ = stream;
}

// 

