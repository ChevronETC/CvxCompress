#include <stdbool.h>

#ifndef CVX_CVXCOMPRESS_HXX
#define CVX_CVXCOMPRESS_HXX

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus

/*!
 * This class implements lossy compression for seismic wavefields.
 * It uses Antonini's 7-9 tap filter wavelet transform.
 * It was derived from the ChvCompress code by Ergas et.al.
 * Recoded with AVX & AVX2 intrinisics for maximum throughput.
 * Words of praise or heaps of scorn can be directed at thor.johnsen@chevron.com.
 */
class CvxCompress
{
public:
	CvxCompress();
	virtual ~CvxCompress();

	/*!
	 * Compress a 3D wavefield using a given block size and number of threads
	 * nx is fast, nz is slow
	 * scale is a relative threshold for discarding wavelet coefficients.
	 * recommendation for seismic wave-fields: scale=1e-2->1e-5
	 * larger scale means higher compression (more lossy)
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
			unsigned int* compressed,
			int num_threads,
			long& compressed_length
			);
	/*!
	 * Compress a 3D wavefield using a given block size
	 * nx is fast, nz is slow
	 * scale is a relative threshold for discarding wavelet coefficients.
	 * recommendation for seismic wave-fields: scale=1e-2->1e-5
	 * larger scale means higher compression (more lossy)
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
			unsigned int* compressed,
			long& compressed_length
			);
	/*!
 	 * Compress a 3D wavefield using a given block size.
 	 * Works same as above, except parameter use_local_RMS is hardcoded to be false.
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
                        unsigned int* compressed,
                        long& compressed_length
                      );
	float Compress(
                        float scale,
                        float* vol,
                        int nx,
                        int ny,
                        int nz,
                        int bx,
                        int by,
                        int bz,
                        unsigned int* compressed,
						int num_threads,
                        long& compressed_length
                      );
	/*!< Decompress a 3D wavefield that was compressed with Compress(...) method */

	float* Decompress(
			int& nx,
			int& ny,
			int& nz,
			unsigned int* compressed,
			long compressed_length 
			);

	void Decompress(
			float* vol,
			int nx,
			int ny,
			int nz,
			unsigned int* compressed,
			int num_threads,
			long compressed_length 
			);

	void Decompress(
			float* vol,
			int nx,
			int ny,
			int nz,
			unsigned int* compressed,
			long compressed_length 
			);

	bool Is_Valid_Block_Size(int bx, int by, int bz);

	static int Min_BX() {return  8;}  /*!< Get minimum X block size. Will always be a power of two.*/
	static int Max_BX() {return 256;}  /*!< Get maximum X block size. Will always be a power of two.*/
	static int Min_BY() {return  8;}  /*!< Get minimum Y block size. Will always be a power of two.*/
	static int Max_BY() {return 256;}  /*!< Get maximum Y block size. Will always be a power of two.*/
	static int Min_BZ() {return  8;}  /*!< Get minimum Z block size. Will always be a power of two.*/
	static int Max_BZ() {return 256;}  /*!< Get maximum Z block size. Will always be a power of two.*/

	bool Run_Module_Tests(bool verbose, bool exhaustive_throughput_tests);  /*!< Execute module tests.*/

};

#endif // __cplusplus

float cvx_compress(
    float scale,
    float* vol,
    int nx,
    int ny,
    int nz,
    int bx,
    int by,
    int bz,
    unsigned int* compressed,
    long* compressed_length
);

float* cvx_decompress_outofplace(
    int* nx,
    int* ny,
    int* nz,
    unsigned int* compressed,
    long compressed_length
);

void cvx_decompress_inplace(
    float* vol,
    int nx,
    int ny,
    int nz,
    unsigned int* compressed,
    long compressed_length
);

float cvx_compress_th(
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
    int num_threads,
    long* compressed_length
);

void cvx_decompress_inplace_th(
    float* vol,
    int nx,
    int ny,
    int nz,
    unsigned int* compressed,
	int num_threads,
    long compressed_length
);

#ifdef __cplusplus
}
#endif

#endif

