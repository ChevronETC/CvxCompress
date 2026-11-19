#include <gtest/gtest.h>

#include <vector>

#include "numerical_test_utils.cuh"

using nc::HostMaxAbsError;
using nc::HostRmsAboutZero;
using nc::MakeSineVolume;

namespace {

// The length returned by Compress must equal the container length independently
// recovered from the buffer header, be strictly positive, stay within the
// allocated maximum, and be consistent with the reported compression ratio.
TEST(CompressedLength, ReportedLengthIsAccurate) {
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
	char* d_c = nullptr;
	float* d_rt = nullptr;
	gpuErrchk(cudaMalloc(&d_vol, nbytes));
	gpuErrchk(cudaMalloc(&d_rt, nbytes));
	gpuErrchk(cudaMalloc(&d_c, max_bytes));
	gpuErrchk(cudaMemcpyAsync(d_vol, vol.data(), nbytes, cudaMemcpyHostToDevice, stream));
	gpuErrchk(cudaStreamSynchronize(stream));

	long len1 = 0;
	float ratio1 = comp.Compress(scale, d_vol, nx, ny, nz, 16, 16, 16, false, d_c, len1);
	gpuErrchk(cudaStreamSynchronize(stream));

	EXPECT_GT(len1, 0L);
	EXPECT_LE(len1, max_bytes);

	// Re-parse the same container header; must recover the identical length.
	long len2 = 0;
	float ratio2 = comp.GetCompressedSize(d_c, len2);
	EXPECT_EQ(len1, len2);
	EXPECT_NEAR(ratio1, ratio2, 1e-4f * ratio1);

	// Ratio is exactly uncompressed_bytes / compressed_bytes.
	double expected_ratio = static_cast<double>(nbytes) / static_cast<double>(len1);
	EXPECT_NEAR(ratio1, static_cast<float>(expected_ratio), static_cast<float>(expected_ratio) * 1e-3f);

	// The reported length must decompress correctly within the error bound,
	// proving it is the true container length and not merely self-consistent.
	comp.Decompress(d_rt, nx, ny, nz, d_c, len1);
	gpuErrchk(cudaStreamSynchronize(stream));
	std::vector<float> rt(nvox);
	gpuErrchk(cudaMemcpyAsync(rt.data(), d_rt, nbytes, cudaMemcpyDeviceToHost, stream));
	gpuErrchk(cudaStreamSynchronize(stream));
	double rms = HostRmsAboutZero(vol.data(), nvox);
	float max_err = HostMaxAbsError(vol.data(), rt.data(), nvox);
	EXPECT_LE(max_err, 10.0 * rms * static_cast<double>(scale));

	gpuErrchk(cudaFree(d_vol));
	gpuErrchk(cudaFree(d_rt));
	gpuErrchk(cudaFree(d_c));
	gpuErrchk(cudaStreamDestroy(stream));
}

}  // namespace
