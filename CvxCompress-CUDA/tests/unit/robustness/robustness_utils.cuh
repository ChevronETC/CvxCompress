#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <vector>

#include "Common/gpuAssert.h"
#include "CvxCompress_CUDA.hxx"

namespace rob {

inline std::vector<float> MakeConstant(long long n, float c) {
	return std::vector<float>(n, c);
}

inline std::vector<float> MakeUniform(long long n, unsigned seed, float lo, float hi) {
	std::srand(seed);
	std::vector<float> v(n);
	for (long long i = 0; i < n; ++i)
		v[i] = lo + (hi - lo) * (static_cast<float>(std::rand()) / RAND_MAX);
	return v;
}

struct RobustResult {
	float ratio;
	long len;
	long long max_len;
	double rel_rms;       // over finite input/output pairs only
	float max_abs_err;    // over finite input/output pairs only
	long long n_nan;
	long long n_inf;
	bool all_finite;
};

// Fresh compressor + stream per call so different shapes never share buffers.
inline RobustResult GpuRoundtrip(const std::vector<float>& in, int nx, int ny, int nz,
                                 float scale, bool use_local_RMS) {
	const long long n = static_cast<long long>(nx) * ny * nz;
	const long long nbytes = n * static_cast<long long>(sizeof(float));
	const long long max_len = CvxCompress_CUDA::GetMaxCompressedSize(nx, ny, nz);

	cudaStream_t stream;
	gpuErrchk(cudaStreamCreate(&stream));
	CvxCompress_CUDA comp;
	comp.SetStream(stream);

	float* d_in = nullptr;
	float* d_out = nullptr;
	char* d_c = nullptr;
	gpuErrchk(cudaMalloc(&d_in, nbytes));
	gpuErrchk(cudaMalloc(&d_out, nbytes));
	gpuErrchk(cudaMalloc(&d_c, max_len));
	gpuErrchk(cudaMemcpy(d_in, in.data(), nbytes, cudaMemcpyHostToDevice));

	long len = 0;
	float ratio = comp.Compress(scale, d_in, nx, ny, nz, 16, 16, 16, use_local_RMS, d_c, len);
	comp.Decompress(d_out, nx, ny, nz, d_c, len);
	gpuErrchk(cudaStreamSynchronize(stream));

	std::vector<float> out(n);
	gpuErrchk(cudaMemcpy(out.data(), d_out, nbytes, cudaMemcpyDeviceToHost));

	gpuErrchk(cudaFree(d_in));
	gpuErrchk(cudaFree(d_out));
	gpuErrchk(cudaFree(d_c));
	gpuErrchk(cudaStreamDestroy(stream));

	RobustResult r{};
	r.ratio = ratio;
	r.len = len;
	r.max_len = max_len;
	double se = 0.0, ss = 0.0;
	r.all_finite = true;
	for (long long i = 0; i < n; ++i) {
		const float o = out[i];
		if (std::isnan(o)) { ++r.n_nan; r.all_finite = false; }
		if (std::isinf(o)) { ++r.n_inf; r.all_finite = false; }
		if (std::isfinite(o) && std::isfinite(in[i])) {
			const double e = static_cast<double>(o) - in[i];
			se += e * e;
			ss += static_cast<double>(in[i]) * in[i];
			const float a = std::fabs(o - in[i]);
			if (a > r.max_abs_err) r.max_abs_err = a;
		}
	}
	r.rel_rms = ss > 0.0 ? std::sqrt(se / ss) : 0.0;
	return r;
}

}  // namespace rob
