#include <gtest/gtest.h>

#include <limits>
#include <vector>

#include "numerical_test_utils.cuh"

using nc::GpuRoundtrip;
using nc::HostMaxAbsError;
using nc::HostRmsAboutZero;
using nc::MakeSineVolume;

namespace {

struct Grid {
	int nx, ny, nz;
};

// Global-RMS compression quantizes wavelet coefficients with step ~ rms*scale,
// so the reconstructed max absolute error is bounded by C*rms*scale plus a
// small scale-independent floor from the wavelet transform's float precision.
TEST(ErrorBound, MaxErrorWithinScaledRmsBound) {
	const float C = 10.0f;
	const float kFloor = 5e-3f;  // fraction of rms; wavelet reconstruction floor
	const Grid grids[] = {{128, 128, 128}, {64, 128, 256}, {96, 128, 160}};
	const float scales[] = {1e-1f, 5e-2f, 1e-2f, 1e-3f};

	for (const Grid& g : grids) {
		// A compressor instance is sized for the first shape it sees, so use a
		// fresh instance per grid.
		cudaStream_t stream;
		gpuErrchk(cudaStreamCreate(&stream));
		CvxCompress_CUDA comp;
		comp.SetStream(stream);

		std::vector<float> vol = MakeSineVolume(g.nx, g.ny, g.nz);
		const long long n = static_cast<long long>(g.nx) * g.ny * g.nz;
		const double rms = HostRmsAboutZero(vol.data(), n);

		for (float scale : scales) {
			std::vector<float> rt;
			nc::RoundtripResult r = GpuRoundtrip(&comp, stream, vol, g.nx, g.ny, g.nz, scale, false, rt);
			float max_err = HostMaxAbsError(vol.data(), rt.data(), n);
			double bound = (C * static_cast<double>(scale) + kFloor) * rms;
			EXPECT_LE(max_err, bound)
				<< "grid " << g.nx << "x" << g.ny << "x" << g.nz
				<< " scale=" << scale << " rms=" << rms << " ratio=" << r.ratio;
		}

		gpuErrchk(cudaStreamDestroy(stream));
	}
}

// Tightening the scale must not increase the reconstruction error.
TEST(ErrorBound, ErrorDecreasesAsScaleTightens) {
	const int nx = 128, ny = 128, nz = 128;
	const long long n = static_cast<long long>(nx) * ny * nz;
	const float scales[] = {1e-1f, 5e-2f, 1e-2f, 1e-3f};

	cudaStream_t stream;
	gpuErrchk(cudaStreamCreate(&stream));
	CvxCompress_CUDA comp;
	comp.SetStream(stream);

	std::vector<float> vol = MakeSineVolume(nx, ny, nz);
	float prev_err = std::numeric_limits<float>::max();
	for (float scale : scales) {
		std::vector<float> rt;
		GpuRoundtrip(&comp, stream, vol, nx, ny, nz, scale, false, rt);
		float max_err = HostMaxAbsError(vol.data(), rt.data(), n);
		EXPECT_LE(max_err, prev_err * 1.05f + 1e-6f)
			<< "error grew when scale tightened to " << scale;
		prev_err = max_err;
	}

	gpuErrchk(cudaStreamDestroy(stream));
}

}  // namespace
