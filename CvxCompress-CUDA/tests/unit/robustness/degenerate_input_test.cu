#include <gtest/gtest.h>

#include <limits>
#include <vector>

#include "robustness_utils.cuh"

using rob::GpuRoundtrip;
using rob::MakeConstant;
using rob::MakeUniform;
using rob::RobustResult;

namespace {

constexpr int NX = 64, NY = 64, NZ = 64;
constexpr long long N = static_cast<long long>(NX) * NY * NZ;
constexpr float SCALE = 1e-2f;

// A zero volume must roundtrip exactly and compress heavily.
TEST(Robustness, ZeroVolume) {
	auto in = MakeConstant(N, 0.0f);
	RobustResult r = GpuRoundtrip(in, NX, NY, NZ, SCALE, false);
	EXPECT_TRUE(r.all_finite);
	EXPECT_EQ(r.max_abs_err, 0.0f);
	EXPECT_GT(r.ratio, 1.0f);
	EXPECT_LE(r.len, r.max_len);
	EXPECT_GT(r.len, 0L);
}

// A non-zero constant volume is essentially lossless.
TEST(Robustness, ConstantVolume) {
	auto in = MakeConstant(N, 3.5f);
	RobustResult r = GpuRoundtrip(in, NX, NY, NZ, SCALE, false);
	EXPECT_TRUE(r.all_finite);
	EXPECT_LT(r.rel_rms, 1e-4);
	EXPECT_LE(r.len, r.max_len);
}

// Tiny magnitudes must not trigger division blow-ups; output stays finite.
TEST(Robustness, NearZeroInput) {
	auto in = MakeConstant(N, 1e-20f);
	RobustResult r = GpuRoundtrip(in, NX, NY, NZ, SCALE, false);
	EXPECT_TRUE(r.all_finite);
	EXPECT_EQ(r.n_nan, 0);
	EXPECT_EQ(r.n_inf, 0);
	EXPECT_LE(r.len, r.max_len);
}

// Moderately large magnitudes (no float overflow) still roundtrip accurately.
TEST(Robustness, LargeMagnitudeRoundtrips) {
	auto in = MakeUniform(N, 11, -1e6f, 1e6f);
	RobustResult r = GpuRoundtrip(in, NX, NY, NZ, SCALE, false);
	EXPECT_TRUE(r.all_finite);
	EXPECT_LT(r.rel_rms, 2.0 * SCALE);
	EXPECT_LE(r.len, r.max_len);
}

// Extreme magnitudes overflow the sum-of-squares RMS and lose signal, but the
// library must degrade gracefully: finite output, no crash, buffer respected.
TEST(Robustness, ExtremeMagnitudeDegradesGracefully) {
	auto in = MakeConstant(N, 1e18f);
	RobustResult r = GpuRoundtrip(in, NX, NY, NZ, SCALE, false);
	EXPECT_TRUE(r.all_finite);
	EXPECT_GT(r.len, 0L);
	EXPECT_LE(r.len, r.max_len);
}

// NaN/Inf inputs must not crash or overflow the output buffer.
TEST(Robustness, NaNInfInputDoesNotCrash) {
	auto in = MakeUniform(N, 13, -1.0f, 1.0f);
	in[100] = std::numeric_limits<float>::quiet_NaN();
	in[200] = std::numeric_limits<float>::infinity();
	in[300] = -std::numeric_limits<float>::infinity();
	RobustResult r = GpuRoundtrip(in, NX, NY, NZ, SCALE, false);
	EXPECT_GT(r.len, 0L);
	EXPECT_LE(r.len, r.max_len);
}

}  // namespace
