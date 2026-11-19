#include <gtest/gtest.h>

#include <vector>

#include "numerical_test_utils.cuh"
#include "CvxCompress_CUDA_helpers.cuh"

using nc::HostRmsAboutZero;
using nc::MakeSineVolume;

namespace {

// The Thrust reduction behind global-RMS scaling must agree with a
// double-precision host reference to within float accumulation error.
TEST(DeviceRms, MatchesHostReference) {
	const int nx = 128, ny = 128, nz = 128;
	const long long n = static_cast<long long>(nx) * ny * nz;
	std::vector<float> vol = MakeSineVolume(nx, ny, nz);

	cudaStream_t stream;
	gpuErrchk(cudaStreamCreate(&stream));

	float* d = nullptr;
	gpuErrchk(cudaMalloc(&d, n * sizeof(float)));
	gpuErrchk(cudaMemcpyAsync(d, vol.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream));
	gpuErrchk(cudaStreamSynchronize(stream));

	float mean = 0.0f, rms = 0.0f;
	CUDA_ComputeGlobalMeanRMS(stream, d, n, mean, rms);
	gpuErrchk(cudaStreamSynchronize(stream));

	double ref = HostRmsAboutZero(vol.data(), n);
	EXPECT_NEAR(rms, ref, ref * 2e-3) << "device rms=" << rms << " host rms=" << ref;

	gpuErrchk(cudaFree(d));
	gpuErrchk(cudaStreamDestroy(stream));
}

// Degenerate volumes must not produce NaN/Inf: a zero field has RMS 0 and a
// constant field c has RMS |c|.
TEST(DeviceRms, ZeroAndConstantVolumes) {
	const int nx = 64, ny = 64, nz = 64;
	const long long n = static_cast<long long>(nx) * ny * nz;

	cudaStream_t stream;
	gpuErrchk(cudaStreamCreate(&stream));
	float* d = nullptr;
	gpuErrchk(cudaMalloc(&d, n * sizeof(float)));

	{
		std::vector<float> zero(n, 0.0f);
		gpuErrchk(cudaMemcpyAsync(d, zero.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream));
		gpuErrchk(cudaStreamSynchronize(stream));
		float mean = -1.0f, rms = -1.0f;
		CUDA_ComputeGlobalMeanRMS(stream, d, n, mean, rms);
		gpuErrchk(cudaStreamSynchronize(stream));
		EXPECT_TRUE(std::isfinite(rms));
		EXPECT_TRUE(std::isfinite(mean));
		EXPECT_FLOAT_EQ(rms, 0.0f);
		EXPECT_FLOAT_EQ(mean, 0.0f);
	}

	{
		const float c = 3.5f;
		std::vector<float> constant(n, c);
		gpuErrchk(cudaMemcpyAsync(d, constant.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream));
		gpuErrchk(cudaStreamSynchronize(stream));
		float mean = 0.0f, rms = 0.0f;
		CUDA_ComputeGlobalMeanRMS(stream, d, n, mean, rms);
		gpuErrchk(cudaStreamSynchronize(stream));
		EXPECT_NEAR(rms, c, c * 1e-4);
		EXPECT_NEAR(mean, c, c * 1e-4);
	}

	gpuErrchk(cudaFree(d));
	gpuErrchk(cudaStreamDestroy(stream));
}

}  // namespace
