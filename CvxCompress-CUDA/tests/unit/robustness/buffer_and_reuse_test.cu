#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "robustness_utils.cuh"

using rob::GpuRoundtrip;
using rob::MakeUniform;
using rob::RobustResult;

namespace {

constexpr int NX = 64, NY = 64, NZ = 64;
constexpr long long N = static_cast<long long>(NX) * NY * NZ;
constexpr float SCALE = 1e-2f;

// High-entropy input barely compresses (may even expand) but must always fit in
// GetMaxCompressedSize and still roundtrip within the error bound.
TEST(RobustnessBuffer, IncompressibleInputFitsBuffer) {
	auto in = MakeUniform(N, 3, -1.0f, 1.0f);
	RobustResult r = GpuRoundtrip(in, NX, NY, NZ, SCALE, false);
	EXPECT_TRUE(r.all_finite);
	EXPECT_GT(r.len, 0L);
	EXPECT_LE(r.len, r.max_len);
	EXPECT_LT(r.rel_rms, 2.0 * SCALE);
}

// Repeated compress/decompress cycles on one instance and shape must be stable:
// identical length and bit-identical reconstruction every cycle.
TEST(RobustnessReuse, RepeatedCyclesAreStable) {
	auto in = MakeUniform(N, 8, -1.0f, 1.0f);
	const long long nbytes = N * static_cast<long long>(sizeof(float));
	const long long max_len = CvxCompress_CUDA::GetMaxCompressedSize(NX, NY, NZ);

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

	long first_len = -1;
	std::vector<float> first_out;
	for (int cycle = 0; cycle < 5; ++cycle) {
		long len = 0;
		comp.Compress(SCALE, d_in, NX, NY, NZ, 16, 16, 16, false, d_c, len);
		comp.Decompress(d_out, NX, NY, NZ, d_c, len);
		gpuErrchk(cudaStreamSynchronize(stream));

		std::vector<float> out(N);
		gpuErrchk(cudaMemcpy(out.data(), d_out, nbytes, cudaMemcpyDeviceToHost));

		if (cycle == 0) {
			first_len = len;
			first_out = out;
		} else {
			EXPECT_EQ(len, first_len) << "cycle " << cycle;
			EXPECT_EQ(0, std::memcmp(out.data(), first_out.data(), nbytes)) << "cycle " << cycle;
		}
	}

	gpuErrchk(cudaFree(d_in));
	gpuErrchk(cudaFree(d_out));
	gpuErrchk(cudaFree(d_c));
	gpuErrchk(cudaStreamDestroy(stream));
}

}  // namespace
