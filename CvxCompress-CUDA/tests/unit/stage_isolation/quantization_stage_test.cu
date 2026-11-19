#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>

#include "Common/gpuAssert.h"
#include "stage_harness.cuh"

namespace {

const dim3 kBlock(16, 1, 16);
const dim3 kGrid(1, 1, 1);
constexpr int N = kBlockVoxels;
constexpr size_t NB = N * sizeof(float);

std::vector<float> RandomBlock(unsigned seed, float lo, float hi) {
	std::srand(seed);
	std::vector<float> v(N);
	for (int i = 0; i < N; ++i)
		v[i] = lo + (hi - lo) * (static_cast<float>(std::rand()) / RAND_MAX);
	return v;
}

// The mean/RMS reduction is isolated and checked against a double-precision
// host reference using the library's own rms = sqrt(sum2/n - sum^2/n^3).
TEST(QuantizationStage, MeanRmsMatchesHost) {
	std::vector<float> in = RandomBlock(3, -5.0f, 5.0f);
	float *d_in, *d_mean, *d_rms;
	gpuErrchk(cudaMalloc(&d_in, NB));
	gpuErrchk(cudaMalloc(&d_mean, sizeof(float)));
	gpuErrchk(cudaMalloc(&d_rms, sizeof(float)));
	gpuErrchk(cudaMemcpy(d_in, in.data(), NB, cudaMemcpyHostToDevice));

	harness_mean_rms<<<kGrid, kBlock>>>(d_in, d_mean, d_rms);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	float mean = 0, rms = 0;
	gpuErrchk(cudaMemcpy(&mean, d_mean, sizeof(float), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&rms, d_rms, sizeof(float), cudaMemcpyDeviceToHost));

	float ref_mean, ref_rms;
	HostBlockMeanRms(in.data(), N, ref_mean, ref_rms);
	EXPECT_NEAR(mean, ref_mean, std::fabs(ref_mean) * 1e-3f + 1e-4f);
	EXPECT_NEAR(rms, ref_rms, ref_rms * 1e-3f);

	gpuErrchk(cudaFree(d_in));
	gpuErrchk(cudaFree(d_mean));
	gpuErrchk(cudaFree(d_rms));
}

// Global normalization applies a fixed multiplier 1/(rms*scale) with zero mean.
TEST(QuantizationStage, GlobalNormalizeScalesByMultiplier) {
	const float scale = 0.01f, gmean = 0.0f, grms = 2.0f;
	const float expected_mult = 1.0f / (grms * scale);
	std::vector<float> in = RandomBlock(5, -3.0f, 3.0f), out(N);

	float *d_in, *d_out, *d_mean, *d_mult;
	gpuErrchk(cudaMalloc(&d_in, NB));
	gpuErrchk(cudaMalloc(&d_out, NB));
	gpuErrchk(cudaMalloc(&d_mean, sizeof(float)));
	gpuErrchk(cudaMalloc(&d_mult, sizeof(float)));
	gpuErrchk(cudaMemcpy(d_in, in.data(), NB, cudaMemcpyHostToDevice));

	harness_normalize_global<<<kGrid, kBlock>>>(d_in, scale, gmean, grms, d_out, d_mean, d_mult);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	float mean = -1, mult = -1;
	gpuErrchk(cudaMemcpy(&mean, d_mean, sizeof(float), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&mult, d_mult, sizeof(float), cudaMemcpyDeviceToHost));
	std::vector<float> out_h(N);
	gpuErrchk(cudaMemcpy(out_h.data(), d_out, NB, cudaMemcpyDeviceToHost));

	EXPECT_FLOAT_EQ(mean, 0.0f);
	EXPECT_FLOAT_EQ(mult, expected_mult);
	for (int i = 0; i < N; ++i) EXPECT_FLOAT_EQ(out_h[i], in[i] * expected_mult) << "index " << i;

	gpuErrchk(cudaFree(d_in));
	gpuErrchk(cudaFree(d_out));
	gpuErrchk(cudaFree(d_mean));
	gpuErrchk(cudaFree(d_mult));
}

// A block containing a NaN cannot be quantized; global normalize signals this
// with a negative multiplier so the caller stores the block raw.
TEST(QuantizationStage, GlobalNormalizeFlagsNaN) {
	const float scale = 0.01f, grms = 2.0f;
	std::vector<float> in = RandomBlock(9, -1.0f, 1.0f);
	in[123] = std::numeric_limits<float>::quiet_NaN();

	float *d_in, *d_out, *d_mean, *d_mult;
	gpuErrchk(cudaMalloc(&d_in, NB));
	gpuErrchk(cudaMalloc(&d_out, NB));
	gpuErrchk(cudaMalloc(&d_mean, sizeof(float)));
	gpuErrchk(cudaMalloc(&d_mult, sizeof(float)));
	gpuErrchk(cudaMemcpy(d_in, in.data(), NB, cudaMemcpyHostToDevice));

	harness_normalize_global<<<kGrid, kBlock>>>(d_in, scale, 0.0f, grms, d_out, d_mean, d_mult);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	float mult = 0;
	gpuErrchk(cudaMemcpy(&mult, d_mult, sizeof(float), cudaMemcpyDeviceToHost));
	EXPECT_LT(mult, 0.0f);

	gpuErrchk(cudaFree(d_in));
	gpuErrchk(cudaFree(d_out));
	gpuErrchk(cudaFree(d_mean));
	gpuErrchk(cudaFree(d_mult));
}

// Local normalization subtracts the block mean, so a constant block quantizes
// to all zeros.
TEST(QuantizationStage, LocalNormalizeZeroCentersConstant) {
	const float scale = 0.01f;
	std::vector<float> in(N, 7.0f), out(N);

	float *d_in, *d_out, *d_mean, *d_mult;
	gpuErrchk(cudaMalloc(&d_in, NB));
	gpuErrchk(cudaMalloc(&d_out, NB));
	gpuErrchk(cudaMalloc(&d_mean, sizeof(float)));
	gpuErrchk(cudaMalloc(&d_mult, sizeof(float)));
	gpuErrchk(cudaMemcpy(d_in, in.data(), NB, cudaMemcpyHostToDevice));

	harness_normalize_local<<<kGrid, kBlock>>>(d_in, scale, d_out, d_mean, d_mult);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	float mean = 0, mult = 0;
	gpuErrchk(cudaMemcpy(&mean, d_mean, sizeof(float), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&mult, d_mult, sizeof(float), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(out.data(), d_out, NB, cudaMemcpyDeviceToHost));

	EXPECT_NEAR(mean, 7.0f, 1e-4f);
	EXPECT_GT(mult, 0.0f);
	for (int i = 0; i < N; ++i) EXPECT_NEAR(out[i], 0.0f, 1e-2f) << "index " << i;

	gpuErrchk(cudaFree(d_in));
	gpuErrchk(cudaFree(d_out));
	gpuErrchk(cudaFree(d_mean));
	gpuErrchk(cudaFree(d_mult));
}

}  // namespace
