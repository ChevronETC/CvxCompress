#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <vector>

#include "Common/gpuAssert.h"
#include "stage_harness.cuh"

namespace {

const dim3 kBlock(16, 1, 16);
const dim3 kGrid(1, 1, 1);
constexpr int N = kBlockVoxels;
constexpr size_t NB = N * sizeof(float);

std::vector<float> RandomBlock(unsigned seed, float lo = -1.0f, float hi = 1.0f) {
	std::srand(seed);
	std::vector<float> v(N);
	for (int i = 0; i < N; ++i)
		v[i] = lo + (hi - lo) * (static_cast<float>(std::rand()) / RAND_MAX);
	return v;
}

// The wavelet transform is only the T stage of the pipeline: forward then
// inverse must reproduce the input to float precision.
TEST(WaveletStage, ForwardInverseIsIdentity) {
	std::vector<float> in = RandomBlock(7), out(N);
	float *d_in, *d_out;
	gpuErrchk(cudaMalloc(&d_in, NB));
	gpuErrchk(cudaMalloc(&d_out, NB));
	gpuErrchk(cudaMemcpy(d_in, in.data(), NB, cudaMemcpyHostToDevice));

	harness_wavelet_roundtrip<<<kGrid, kBlock>>>(d_in, d_out);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(out.data(), d_out, NB, cudaMemcpyDeviceToHost));

	for (int i = 0; i < N; ++i) EXPECT_NEAR(out[i], in[i], 1e-3f) << "index " << i;
	gpuErrchk(cudaFree(d_in));
	gpuErrchk(cudaFree(d_out));
}

// A constant field is a DC signal the transform must preserve exactly through a
// roundtrip.
TEST(WaveletStage, ConstantBlockPreserved) {
	std::vector<float> in(N, 2.71828f), out(N);
	float *d_in, *d_out;
	gpuErrchk(cudaMalloc(&d_in, NB));
	gpuErrchk(cudaMalloc(&d_out, NB));
	gpuErrchk(cudaMemcpy(d_in, in.data(), NB, cudaMemcpyHostToDevice));

	harness_wavelet_roundtrip<<<kGrid, kBlock>>>(d_in, d_out);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(out.data(), d_out, NB, cudaMemcpyDeviceToHost));

	for (int i = 0; i < N; ++i) EXPECT_NEAR(out[i], in[i], 1e-4f) << "index " << i;
	gpuErrchk(cudaFree(d_in));
	gpuErrchk(cudaFree(d_out));
}

// The forward transform is linear: F(a*x) == a*F(x).
TEST(WaveletStage, ForwardIsLinear) {
	const float a = 3.0f;
	std::vector<float> x = RandomBlock(11), ax(N);
	for (int i = 0; i < N; ++i) ax[i] = a * x[i];

	float *d_x, *d_ax, *d_fx, *d_fax;
	gpuErrchk(cudaMalloc(&d_x, NB));
	gpuErrchk(cudaMalloc(&d_ax, NB));
	gpuErrchk(cudaMalloc(&d_fx, NB));
	gpuErrchk(cudaMalloc(&d_fax, NB));
	gpuErrchk(cudaMemcpy(d_x, x.data(), NB, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_ax, ax.data(), NB, cudaMemcpyHostToDevice));

	harness_wavelet_forward<<<kGrid, kBlock>>>(d_x, d_fx);
	harness_wavelet_forward<<<kGrid, kBlock>>>(d_ax, d_fax);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	std::vector<float> fx(N), fax(N);
	gpuErrchk(cudaMemcpy(fx.data(), d_fx, NB, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(fax.data(), d_fax, NB, cudaMemcpyDeviceToHost));

	for (int i = 0; i < N; ++i)
		EXPECT_NEAR(fax[i], a * fx[i], 1e-3f * (1.0f + std::fabs(a * fx[i]))) << "index " << i;

	gpuErrchk(cudaFree(d_x));
	gpuErrchk(cudaFree(d_ax));
	gpuErrchk(cudaFree(d_fx));
	gpuErrchk(cudaFree(d_fax));
}

}  // namespace
