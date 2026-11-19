#ifndef CVXCOMPRESS_CUDA_HXX
#define CVXCOMPRESS_CUDA_HXX

#include <cuda_runtime.h>
#include <stdbool.h>
#include <Common/gpuAssert.h>
#include <Common/profiler.h>

#ifdef __cplusplus

/**
 * @brief Configuration and resource container for CUDA-based compression/decompression.
 *
 * This struct manages device memory, host staging buffers, and metadata required
 * for compression and decompression operations. It owns all allocated resources
 * and frees them upon destruction.
 */
struct CvxCompressCudaConfig{
	int device_;
	float scale_;
	int voi_pad_;

	// volume dimensions in the internal coordinate system
	int nx_, ny_, nz_;
	int nx8_, ny8_, nz8_;
	int nbx_;
	long long slice_size_;

	// device PQ buffers
	float *d_pq0_, *d_pq1_, *d_pq2_, *d_pq3_;

	// container sizing
	long long max_compressed_sparse_size_;
	long long max_sparse_subvolumes_;
	long long max_container_payload_size_;
	long long max_compressed_size_;

	// compressed device container + sparse staging + pinned host mirror
	char* d_compressed_sparse_01_;
	char* d_compressed_sparse_23_;
	char* h_compressed_;

	// decompressed device slabs + host staging
	float* d_decomp_01_;
	float* d_decomp_23_;

	// volume-of-interest bounds (internal coordinate system)
	int voi_x0_, voi_y0_, voi_z0_;
	int voi_x1_, voi_y1_, voi_z1_;

	CvxCompressCudaConfig():
		nx_(0), ny_(0), nz_(0),
		nx8_(0), ny8_(0), nz8_(0),
		nbx_(0), slice_size_(0),
		d_pq0_(nullptr), d_pq1_(nullptr), d_pq2_(nullptr), d_pq3_(nullptr),
		max_compressed_sparse_size_(0), 
		max_sparse_subvolumes_(0), 
		max_container_payload_size_(0), 
		max_compressed_size_(0),
		h_compressed_(nullptr), 
		d_decomp_01_(nullptr), d_decomp_23_(nullptr), 
		d_compressed_sparse_01_(nullptr), d_compressed_sparse_23_(nullptr), 
		voi_x0_(0), voi_y0_(0), voi_z0_(0), voi_x1_(0), voi_y1_(0), voi_z1_(0)
	{}


	~CvxCompressCudaConfig(){
		if (d_decomp_23_) gpuErrchk( cudaFree(d_decomp_23_) );
		if (d_decomp_01_) gpuErrchk( cudaFree(d_decomp_01_) );
		if (d_compressed_sparse_23_) gpuErrchk( cudaFree(d_compressed_sparse_23_) );
		if (d_compressed_sparse_01_) gpuErrchk( cudaFree(d_compressed_sparse_01_) );
		if (d_pq3_) gpuErrchk( cudaFree(d_pq3_) );
		if (d_pq2_) gpuErrchk( cudaFree(d_pq2_) );
		if (d_pq1_) gpuErrchk( cudaFree(d_pq1_) );
		if (d_pq0_) gpuErrchk( cudaFree(d_pq0_) );
		if (h_compressed_)
		{
			gpuErrchk(cudaHostUnregister(h_compressed_));
			free(h_compressed_);
		}
	}
};

class CvxCompress_CUDA {
	public:
		CvxCompress_CUDA();
		virtual ~CvxCompress_CUDA();
		/**
		 * @brief Compress a 3D volume using block dimensions.
		 *
		 * @param scale Compression scale factor
		 * @param vol Device pointer to input volume data
		 * @param nx Volume size in x-dimension
		 * @param ny Volume size in y-dimension
		 * @param nz Volume size in z-dimension
		 * @param bx Block size in x-dimension
		 * @param by Block size in y-dimension
		 * @param bz Block size in z-dimension
		 * @param use_local_RMS flag to set RMS computation
		 * @param compressed Device pointer to output buffer for compressed data
		 * @param compressed_length Output size of compressed data (bytes)
		 * @return Compression ratio
		 */
		float Compress(
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
		);

		/**
		 * @brief Compress a 3D volume using a volume-of-interest (VOI) padding scheme.
		 *
		 * @param scale Compression scale factor
		 * @param vol Device pointer to input volume data
		 * @param nx Volume size in x-dimension
		 * @param ny Volume size in y-dimension
		 * @param nz Volume size in z-dimension
		 * @param use_local_RMS flag to set RMS computation
		 * @param compressed Device pointer to output buffer for compressed data
		 * @param compressed_length Output size of compressed data (bytes)
		 * @param voi_pad Padding applied to the volume-of-interest
		 * @return Compression ratio
		 */
		float Compress(
			float scale,
			float* vol,
			int nx,
			int ny,
			int nz,
			bool use_local_RMS,
			char* compressed,
			long& compressed_length,
			int voi_pad
		);

		/**
		 * @brief Compress a 3D volume using block dimensions.
		 *
		 * @param scale Compression scale factor
		 * @param vol Device pointer to input volume data
		 * @param nx Volume size in x-dimension
		 * @param ny Volume size in y-dimension
		 * @param nz Volume size in z-dimension
		 * @param bx Block size in x-dimension
		 * @param by Block size in y-dimension
		 * @param bz Block size in z-dimension
		 * @param compressed Device pointer to output buffer for compressed data
		 * @param compressed_length Output size of compressed data (bytes)
		 * @return Compression ratio
		 */
		float Compress(
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
		);

		/**
		 * @brief Compress a 3D volume using a volume-of-interest (VOI) padding scheme.
		 *
		 * @param scale Compression scale factor
		 * @param vol Device pointer to input volume data
		 * @param nx Volume size in x-dimension
		 * @param ny Volume size in y-dimension
		 * @param nz Volume size in z-dimension
		 * @param compressed Device pointer to output buffer for compressed data
		 * @param compressed_length Output size of compressed data (bytes)
		 * @param voi_pad Padding applied to the volume-of-interest
		 * @return Compression ratio
		 */
		float Compress(
			float scale,
			float* vol,
			int nx,
			int ny,
			int nz,
			char* compressed,
			long& compressed_length,
			int voi_pad
		);

		/**
		 * @brief Asynchronously enqueue compression of a 3D volume.
		 *
		 * All GPU work (buffer copies and compression kernels) is queued on the
		 * plan's stream and this call returns immediately, without blocking on
		 * the device. The compressed container, including its size header, is
		 * written to @p compressed on the device. The compressed size is NOT
		 * valid until the caller synchronizes the plan's stream (see SetStream)
		 * and calls GetCompressedSize().
		 *
		 * This mirrors bitcompCompressLossy_fp32 (async, on-stream).
		 *
		 * @param scale Compression scale factor
		 * @param vol Device pointer to input volume data
		 * @param nx Volume size in x-dimension
		 * @param ny Volume size in y-dimension
		 * @param nz Volume size in z-dimension
		 * @param bx Block size in x-dimension (ignored, kept for API symmetry)
		 * @param by Block size in y-dimension (ignored, kept for API symmetry)
		 * @param bz Block size in z-dimension (ignored, kept for API symmetry)
		 * @param use_local_RMS flag to set RMS computation
		 * @param compressed Device pointer to output buffer for compressed data
		 */
		void CompressAsync(
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
		);

		/**
		 * @brief Read the final compressed size from a compressed container.
		 *
		 * Blocking. The compression that produced @p compressed must have
		 * completed (i.e. the plan's stream must be synchronized) before calling
		 * this. Performs a small device-to-host copy of the container header and
		 * parses it to recover the byte length.
		 *
		 * This mirrors bitcompGetCompressedSize (blocking, reads size from buffer).
		 *
		 * @param compressed Device pointer to the compressed container
		 * @param compressed_length Output size of compressed data (bytes)
		 * @return Compression ratio
		 */
		float GetCompressedSize(
			const char* compressed,
			long& compressed_length
		);

		/**
		 * @brief Decompress a compressed volume.
		 *
		 * @param vol Device pointer to output buffer for decompressed volume
		 * @param nx Volume size in x-dimension
		 * @param ny Volume size in y-dimension
		 * @param nz Volume size in z-dimension
		 * @param compressed Device pointer to input compressed data
		 * @param compressed_length Size of compressed data in bytes
		 */
		void Decompress(
			float* vol,
			int nx,
			int ny,
			int nz,
			char* compressed,
			long compressed_length 
		);

		/**
		 * @brief Compute the maximum required compressed buffer size.
		 *
		 * This provides a worst-case upper bound for allocating device memory
		 * before compression.
		 *
		 * @param nx Volume size in x-dimension
		 * @param ny Volume size in y-dimension
		 * @param nz Volume size in z-dimension
		 * @return Maximum required buffer size in bytes
		 */
		static long long GetMaxCompressedSize(int nx, int ny, int nz);

		void SetStream(cudaStream_t stream);

	private:
		Profiler profiler_;
		CvxCompressCudaConfig config_;
		cudaStream_t stream_{0};

		void InitCompress(
			int nx,
			int ny,
			int nz
		);

		void InitDecompress(
			int nx,
			int ny,
			int nz
		);

		// Enqueue all compression GPU work (buffer copies, compress kernels, and
		// the container-header D2H copy) on stream_. Does not synchronize.
		void EnqueueCompress_(
			float scale,
			float* vol,
			int nx,
			int ny,
			int nz,
			bool use_local_RMS,
			char* compressed,
			int voi_pad
		);

		// Parse the container header (already copied to the pinned host mirror by
		// EnqueueCompress_) to recover the compressed byte length. Host-only; the
		// caller must have synchronized stream_ first. Returns the ratio.
		float ParseCompressedLength_(
			long& compressed_length,
			bool use_local_RMS = true
		);

};
#endif // __cplusplus

// add c-style methods here

#ifdef __cplusplus
typedef CvxCompress_CUDA * CvxCompress_CUDA_ptr;
#endif

#endif // CVXCOMPRESS_CUDA_HXX