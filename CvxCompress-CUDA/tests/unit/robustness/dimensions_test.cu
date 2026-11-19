#include <gtest/gtest.h>

#include <vector>

#include "robustness_utils.cuh"

using rob::GpuRoundtrip;
using rob::MakeUniform;
using rob::RobustResult;

namespace {

constexpr float SCALE = 1e-2f;

// Dimensions that are not multiples of the 16^3 block must be padded internally
// and still roundtrip within the error bound and inside the size bound.
TEST(RobustnessDims, NonMultipleOf16) {
	const int nx = 17, ny = 19, nz = 23;
	auto in = MakeUniform((long long)nx * ny * nz, 4, -1.0f, 1.0f);
	RobustResult r = GpuRoundtrip(in, nx, ny, nz, SCALE, false);
	EXPECT_TRUE(r.all_finite);
	EXPECT_LT(r.rel_rms, 2.0 * SCALE);
	EXPECT_LE(r.len, r.max_len);
}

// The smallest possible volume is a single voxel padded to a full block.
TEST(RobustnessDims, SingleVoxel) {
	std::vector<float> in{0.7f};
	RobustResult r = GpuRoundtrip(in, 1, 1, 1, SCALE, false);
	EXPECT_TRUE(r.all_finite);
	EXPECT_GT(r.len, 0L);
	EXPECT_LE(r.len, r.max_len);
}

// A sub-block volume (smaller than 16 in every dimension).
TEST(RobustnessDims, SubBlockVolume) {
	const int nx = 8, ny = 8, nz = 8;
	auto in = MakeUniform((long long)nx * ny * nz, 5, -1.0f, 1.0f);
	RobustResult r = GpuRoundtrip(in, nx, ny, nz, SCALE, false);
	EXPECT_TRUE(r.all_finite);
	EXPECT_LT(r.rel_rms, 2.0 * SCALE);
	EXPECT_LE(r.len, r.max_len);
}

// Extreme aspect ratios along each axis.
TEST(RobustnessDims, ThinAlongX) {
	auto in = MakeUniform(512, 6, -1.0f, 1.0f);
	RobustResult r = GpuRoundtrip(in, 512, 1, 1, SCALE, false);
	EXPECT_TRUE(r.all_finite);
	EXPECT_LT(r.rel_rms, 2.0 * SCALE);
	EXPECT_LE(r.len, r.max_len);
}

TEST(RobustnessDims, ThinAlongZ) {
	auto in = MakeUniform(512, 7, -1.0f, 1.0f);
	RobustResult r = GpuRoundtrip(in, 1, 1, 512, SCALE, false);
	EXPECT_TRUE(r.all_finite);
	EXPECT_LT(r.rel_rms, 2.0 * SCALE);
	EXPECT_LE(r.len, r.max_len);
}

}  // namespace
