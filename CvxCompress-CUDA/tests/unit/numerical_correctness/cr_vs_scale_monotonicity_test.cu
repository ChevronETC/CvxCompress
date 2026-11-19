#include <gtest/gtest.h>

#include <limits>
#include <vector>

#include "numerical_test_utils.cuh"

using nc::GpuRoundtrip;
using nc::MakeSineVolume;

namespace {

// A smaller scale retains more wavelet coefficients, producing a larger
// container. The compression ratio must therefore be monotonically
// non-increasing as the scale tightens.
TEST(CompressionRatio, DecreasesAsScaleTightens) {
	const int nx = 128, ny = 128, nz = 128;
	const float scales[] = {1e-1f, 5e-2f, 1e-2f, 5e-3f, 1e-3f};

	cudaStream_t stream;
	gpuErrchk(cudaStreamCreate(&stream));
	CvxCompress_CUDA comp;
	comp.SetStream(stream);

	std::vector<float> vol = MakeSineVolume(nx, ny, nz);
	float prev_ratio = std::numeric_limits<float>::max();
	for (float scale : scales) {
		std::vector<float> rt;
		nc::RoundtripResult r = GpuRoundtrip(&comp, stream, vol, nx, ny, nz, scale, false, rt);
		EXPECT_GT(r.ratio, 1.0f) << "scale=" << scale;
		// 2% slack absorbs ties/quantization noise between adjacent scales.
		EXPECT_LE(r.ratio, prev_ratio * 1.02f)
			<< "ratio increased when scale tightened to " << scale;
		prev_ratio = r.ratio;
	}

	gpuErrchk(cudaStreamDestroy(stream));
}

}  // namespace
