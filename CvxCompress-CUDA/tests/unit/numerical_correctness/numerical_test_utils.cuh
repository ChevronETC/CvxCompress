#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <vector>

#include "Common/gpuAssert.h"
#include "CvxCompress_CUDA.hxx"

namespace nc {

// Smooth, bounded sinusoidal field in [-1, 1]. Full periods over each axis keep
// the mean near zero so RMS-about-zero ~= RMS-about-mean.
inline std::vector<float> MakeSineVolume(int nx, int ny, int nz,
                                         float kx = 20.0f,
                                         float ky = 20.0f,
                                         float kz = 20.0f) {
	std::vector<float> v(static_cast<size_t>(nx) * ny * nz);
	for (int iz = 0; iz < nz; ++iz) {
		float z = static_cast<float>(iz) / nz;
		for (int iy = 0; iy < ny; ++iy) {
			float y = static_cast<float>(iy) / ny;
			for (int ix = 0; ix < nx; ++ix) {
				float x = static_cast<float>(ix) / nx;
				size_t idx = (static_cast<size_t>(iz) * ny + iy) * nx + ix;
				v[idx] = sinf(kx * x) * sinf(ky * y) * sinf(kz * z);
			}
		}
	}
	return v;
}

// RMS about zero, sqrt(sum(x^2)/N), accumulated in double. This matches the
// quantity CUDA_ComputeGlobalMeanRMS reports and drives the quantization step.
inline double HostRmsAboutZero(const float* d, long long n) {
	double s = 0.0;
	for (long long i = 0; i < n; ++i) s += static_cast<double>(d[i]) * d[i];
	return std::sqrt(s / static_cast<double>(n));
}

inline float HostMaxAbsError(const float* a, const float* b, long long n) {
	float m = 0.0f;
	for (long long i = 0; i < n; ++i) {
		float e = std::fabs(a[i] - b[i]);
		if (e > m) m = e;
	}
	return m;
}

struct RoundtripResult {
	float ratio;
	long len;
};

// Full device compress + decompress roundtrip. host_roundtrip receives the
// decompressed volume copied back to the host.
inline RoundtripResult GpuRoundtrip(CvxCompress_CUDA* comp,
                                    cudaStream_t stream,
                                    const std::vector<float>& host_vol,
                                    int nx, int ny, int nz,
                                    float scale,
                                    bool use_local_RMS,
                                    std::vector<float>& host_roundtrip) {
	const long long nvox = static_cast<long long>(nx) * ny * nz;
	const long long nbytes = nvox * static_cast<long long>(sizeof(float));

	float* d_vol = nullptr;
	float* d_rt = nullptr;
	char* d_c = nullptr;
	gpuErrchk(cudaMalloc(&d_vol, nbytes));
	gpuErrchk(cudaMalloc(&d_rt, nbytes));
	gpuErrchk(cudaMalloc(&d_c, CvxCompress_CUDA::GetMaxCompressedSize(nx, ny, nz)));

	gpuErrchk(cudaMemcpyAsync(d_vol, host_vol.data(), nbytes, cudaMemcpyHostToDevice, stream));
	gpuErrchk(cudaStreamSynchronize(stream));

	long len = 0;
	float ratio = comp->Compress(scale, d_vol, nx, ny, nz, 16, 16, 16, use_local_RMS, d_c, len);
	comp->Decompress(d_rt, nx, ny, nz, d_c, len);
	gpuErrchk(cudaStreamSynchronize(stream));

	host_roundtrip.resize(nvox);
	gpuErrchk(cudaMemcpyAsync(host_roundtrip.data(), d_rt, nbytes, cudaMemcpyDeviceToHost, stream));
	gpuErrchk(cudaStreamSynchronize(stream));

	gpuErrchk(cudaFree(d_vol));
	gpuErrchk(cudaFree(d_rt));
	gpuErrchk(cudaFree(d_c));

	RoundtripResult r;
	r.ratio = ratio;
	r.len = len;
	return r;
}

}  // namespace nc
