#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <vector>

#include "Common/gpuAssert.h"
#include "stage_harness.cuh"

namespace {

const dim3 kBlock(16, 1, 16);
const dim3 kGrid(1, 1, 1);
constexpr int N = kBlockVoxels;
constexpr size_t NB = N * sizeof(float);
constexpr int kMaxBlockBytes = 16384;

// Encode then decode an integer-valued block and return the reconstruction.
// The RLE codec is lossless for integer inputs, so out must equal in exactly.
std::vector<float> EncodeDecode(const std::vector<float>& in, int& esc, int& pay) {
	float* d_in = nullptr;
	char* d_bytes = nullptr;
	int *d_esc = nullptr, *d_pay = nullptr;
	float* d_out = nullptr;
	gpuErrchk(cudaMalloc(&d_in, NB));
	gpuErrchk(cudaMalloc(&d_bytes, kMaxBlockBytes));
	gpuErrchk(cudaMalloc(&d_esc, sizeof(int)));
	gpuErrchk(cudaMalloc(&d_pay, sizeof(int)));
	gpuErrchk(cudaMalloc(&d_out, NB));
	gpuErrchk(cudaMemset(d_bytes, 0, kMaxBlockBytes));
	gpuErrchk(cudaMemcpy(d_in, in.data(), NB, cudaMemcpyHostToDevice));

	harness_rle_encode<<<kGrid, kBlock>>>(d_in, d_bytes, d_esc, d_pay);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaMemcpy(&esc, d_esc, sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&pay, d_pay, sizeof(int), cudaMemcpyDeviceToHost));

	harness_rle_decode<<<kGrid, kBlock>>>(d_bytes, esc, d_out);
	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());
	std::vector<float> out(N);
	gpuErrchk(cudaMemcpy(out.data(), d_out, NB, cudaMemcpyDeviceToHost));

	gpuErrchk(cudaFree(d_in));
	gpuErrchk(cudaFree(d_bytes));
	gpuErrchk(cudaFree(d_esc));
	gpuErrchk(cudaFree(d_pay));
	gpuErrchk(cudaFree(d_out));
	return out;
}

// An all-zero block collapses to a single run; decode must restore all zeros.
TEST(RleStage, RoundtripAllZeros) {
	std::vector<float> in(N, 0.0f);
	int esc = 0, pay = 0;
	std::vector<float> out = EncodeDecode(in, esc, pay);
	EXPECT_LT(esc + pay, kMaxBlockBytes);
	for (int i = 0; i < N; ++i) EXPECT_EQ(out[i], in[i]) << "index " << i;
}

// Dense small integers (no runs) encode as direct single-byte escape codes.
TEST(RleStage, RoundtripDenseSmallInts) {
	std::srand(21);
	std::vector<float> in(N);
	for (int i = 0; i < N; ++i) {
		int val = (std::rand() % 249) - 124;   // [-124, 124], direct-encoded range
		in[i] = static_cast<float>(val == 0 ? 1 : val);  // avoid zeros: keep it a no-run block
	}
	int esc = 0, pay = 0;
	std::vector<float> out = EncodeDecode(in, esc, pay);
	EXPECT_LT(esc + pay, kMaxBlockBytes);
	for (int i = 0; i < N; ++i) EXPECT_EQ(out[i], in[i]) << "index " << i;
}

// Sparse blocks exercise interleaved zero runs and direct values.
TEST(RleStage, RoundtripSparseWithZeroRuns) {
	std::srand(37);
	std::vector<float> in(N, 0.0f);
	for (int i = 0; i < N; i += 7)
		in[i] = static_cast<float>((std::rand() % 200) - 100);
	int esc = 0, pay = 0;
	std::vector<float> out = EncodeDecode(in, esc, pay);
	EXPECT_LT(esc + pay, kMaxBlockBytes);
	for (int i = 0; i < N; ++i) EXPECT_EQ(out[i], in[i]) << "index " << i;
}

// Values outside the direct range use multi-byte value-escape payloads.
TEST(RleStage, RoundtripWideValues) {
	std::vector<float> in(N, 0.0f);
	for (int i = 0; i < N; i += 16)
		in[i] = static_cast<float>(((i / 16) % 2 == 0) ? 1000 : -2000);
	int esc = 0, pay = 0;
	std::vector<float> out = EncodeDecode(in, esc, pay);
	EXPECT_LT(esc + pay, kMaxBlockBytes);
	for (int i = 0; i < N; ++i) EXPECT_EQ(out[i], in[i]) << "index " << i;
}

}  // namespace
