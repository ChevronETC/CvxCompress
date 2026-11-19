#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "numerical_test_utils.cuh"

using nc::MakeSineVolume;

namespace {

// Compressing identical input twice must be functionally deterministic: the
// container length is identical and the decompressed volume is bit-for-bit
// identical. (The compressed byte stream carries a small amount of
// order-dependent per-block metadata, so stream byte-identity is not asserted.)
TEST(Determinism, RepeatedCompressReconstructsIdentically) {
	const int nx = 128, ny = 128, nz = 128;
	const float scale = 1e-2f;
	const long long nvox = static_cast<long long>(nx) * ny * nz;
	const long long nbytes = nvox * static_cast<long long>(sizeof(float));
	const long long max_bytes = CvxCompress_CUDA::GetMaxCompressedSize(nx, ny, nz);

	cudaStream_t stream;
	gpuErrchk(cudaStreamCreate(&stream));
	CvxCompress_CUDA comp;
	comp.SetStream(stream);

	std::vector<float> vol = MakeSineVolume(nx, ny, nz);

	float* d_vol = nullptr;
	char* d_c1 = nullptr;
	char* d_c2 = nullptr;
	float* d_r1 = nullptr;
	float* d_r2 = nullptr;
	gpuErrchk(cudaMalloc(&d_vol, nbytes));
	gpuErrchk(cudaMalloc(&d_c1, max_bytes));
	gpuErrchk(cudaMalloc(&d_c2, max_bytes));
	gpuErrchk(cudaMalloc(&d_r1, nbytes));
	gpuErrchk(cudaMalloc(&d_r2, nbytes));
	gpuErrchk(cudaMemcpyAsync(d_vol, vol.data(), nbytes, cudaMemcpyHostToDevice, stream));
	gpuErrchk(cudaStreamSynchronize(stream));

	long len1 = 0, len2 = 0;
	comp.Compress(scale, d_vol, nx, ny, nz, 16, 16, 16, false, d_c1, len1);
	gpuErrchk(cudaStreamSynchronize(stream));
	comp.Compress(scale, d_vol, nx, ny, nz, 16, 16, 16, false, d_c2, len2);
	gpuErrchk(cudaStreamSynchronize(stream));

	ASSERT_GT(len1, 0L);
	EXPECT_EQ(len1, len2);

	comp.Decompress(d_r1, nx, ny, nz, d_c1, len1);
	comp.Decompress(d_r2, nx, ny, nz, d_c2, len2);
	gpuErrchk(cudaStreamSynchronize(stream));

	std::vector<float> r1(nvox), r2(nvox);
	gpuErrchk(cudaMemcpyAsync(r1.data(), d_r1, nbytes, cudaMemcpyDeviceToHost, stream));
	gpuErrchk(cudaMemcpyAsync(r2.data(), d_r2, nbytes, cudaMemcpyDeviceToHost, stream));
	gpuErrchk(cudaStreamSynchronize(stream));

	EXPECT_EQ(0, std::memcmp(r1.data(), r2.data(), nbytes));

	gpuErrchk(cudaFree(d_vol));
	gpuErrchk(cudaFree(d_c1));
	gpuErrchk(cudaFree(d_c2));
	gpuErrchk(cudaFree(d_r1));
	gpuErrchk(cudaFree(d_r2));
	gpuErrchk(cudaStreamDestroy(stream));
}

}  // namespace
