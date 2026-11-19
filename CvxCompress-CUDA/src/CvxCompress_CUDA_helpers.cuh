#include <cuda_runtime_api.h>

/**
 * @brief Compute the global mean and RMS of a device volume using Thrust.
 * Pass global_rms > 0 to CUDA_CvxCompress_CompressSubvolume_To_Container to
 * enable global RMS scale
 * Pass global_rms = 0 to use local per-block RMS (default behaviour).
 *
 * @param stream      CUDA stream for Thrust execution.
 * @param vol         Device pointer to the input volume.
 * @param N           Total number of elements in vol.
 * @param global_mean Output: mean over all voxels.
 * @param global_rms  Output: RMS about the global mean.
 */
void CUDA_ComputeGlobalMeanRMS(
	cudaStream_t stream,
	const float* vol,
	long long N,
	float& global_mean,
	float& global_rms
	);

/** 
 * @brief Initialize an empty compressed container on the GPU.
 *
 * @param stream CUDA stream used for the kernel launch.
 * @param compressed_container Device pointer to the container buffer.
 * @param max_num_subvolumes Maximum number of subvolumes that can be stored.
 * @param max_container_payload_size_in_bytes Maximum payload size excluding the container header.
 */
void CUDA_CvxCompress_InitContainer(
	cudaStream_t stream,
	char* compressed_container,
	int max_num_subvolumes,
	long long max_container_payload_size_in_bytes
	);

/**
 * @brief Compress a raw scalar subvolume and pack it into a container.
 *
 * @param stream CUDA stream used for the kernel launch.
 * @param origin User-defined origin tag for the subvolume.
 * @param timestep Timestep associated with the subvolume.
 * @param pq0 Device pointer to the first component of the quantized subvolume.
 * @param pq1 Device pointer to the second component of the quantized subvolume.
 * @param scale Compression scale factor.
 * @param compressed_sparse Temporary device buffer used for sparse compressed output.
 * @param compressed_container Destination packed container buffer.
 * @param sv_x0 Subvolume minimum X coordinate.
 * @param sv_y0 Subvolume minimum Y coordinate.
 * @param sv_z0 Subvolume minimum Z coordinate.
 * @param sv_y1 Subvolume maximum Y coordinate.
 * @param sv_z1 Subvolume maximum Z coordinate.
 * @param voi_x0 Volume-of-interest minimum X coordinate.
 * @param voi_y0 Volume-of-interest minimum Y coordinate.
 * @param voi_z0 Volume-of-interest minimum Z coordinate.
 * @param voi_x1 Volume-of-interest maximum X coordinate.
 * @param voi_y1 Volume-of-interest maximum Y coordinate.
 * @param voi_z1 Volume-of-interest maximum Z coordinate.
 * @param use_local_RMS flag to set RMS computation
 */
void CUDA_CvxCompress_CompressSubvolume_To_Container(
    cudaStream_t stream,
    int origin,
    int timestep,
    float* pq0,
	float* pq1,
    float scale,
    char* compressed_sparse,
    char* compressed_container,
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
);

/**
 * @brief Decompress a subvolume from a packed container using metadata lookup.
 *
 * @param stream CUDA stream used for the kernel launch.
 * @param compressed_container Packed container buffer.
 * @param origin Origin tag used when the subvolume was stored.
 * @param timestep Timestep used when the subvolume was stored.
 * @param sv_x0 Subvolume minimum X coordinate.
 * @param sv_y0 Subvolume minimum Y coordinate.
 * @param sv_z0 Subvolume minimum Z coordinate.
 * @param sv_y1 Subvolume maximum Y coordinate.
 * @param sv_z1 Subvolume maximum Z coordinate.
 * @param decompressed Destination scalar output buffer.
 * @param crash_on_error If true, allow an invalid lookup to fail loudly; if false, skip when not found.
 */
void CUDA_CvxCompress_DecompressSubvolume_In_Container(
	cudaStream_t stream,
	char* compressed_container,
	int origin,
	int timestep,
	int sv_x0,
	int sv_y0,
	int sv_z0,
	int sv_y1,
	int sv_z1,
	float* decompressed,
	bool crash_on_error
);

/**
 * @brief Copy 8 consecutive x-planes from a device volume into a zero-padded PQ slab.
 *
 * @param stream      CUDA stream for the kernel launch.
 * @param vol         Device pointer to source volume (layout: vol[ix*ny*nz + iy*nz + iz]).
 * @param nx          Source volume x-dimension.
 * @param ny          Source volume y-dimension.
 * @param nz          Source volume z-dimension.
 * @param ny8         Padded y-dimension of the destination slab (multiple of 8).
 * @param nz8         Padded z-dimension of the destination slab (multiple of 8).
 * @param d_pq        Device pointer to destination buffer (8 * ny8 * nz8 floats).
 * @param x0          First x-plane index to copy from the source volume.
 */
void CUDA_Copy_To_Buffer(
	cudaStream_t stream,
	const float* vol,
	int nx, int ny, int nz,
	int ny8, int nz8,
	float* d_pq,
	int x0);

/**
 * @brief Copy 16 consecutive x-planes from a device buffer into a device volume
 *
 * @param stream      CUDA stream for the kernel launch.
 * @param vol         Device pointer to destination volume (layout: vol[ix*ny*nz + iy*nz + iz]).
 * @param nx          Source volume x-dimension.
 * @param ny          Source volume y-dimension.
 * @param nz          Source volume z-dimension.
 * @param ny8         Padded y-dimension of the destination slab (multiple of 8).
 * @param nz8         Padded z-dimension of the destination slab (multiple of 8).
 * @param d_decomp    Device pointer to source buffer (ny8 * nz8 * 16 floats).
 * @param x0          First x-plane index to copy from the source buffer.
 */
void CUDA_Copy_From_Buffer(
	cudaStream_t stream,
	float* vol,
	int nx, int ny, int nz,
	int ny8, int nz8,
	const float* d_decomp,
	int x0);