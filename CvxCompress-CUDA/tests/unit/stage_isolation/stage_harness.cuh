#pragma once

#include <cmath>

// Every stage harness kernel runs on a single 16x16x16 block: launch with
// dim3 block(16, 1, 16) so thrid = threadIdx.z*16 + threadIdx.x spans 0..255
// and each thread owns 16 cells at index thrid + i*256 (the production layout).
static constexpr int kBlockVoxels = 16 * 16 * 16;

// Host reference for the per-block mean/RMS, matching ___cuComputeMeanRMS:
// rms = sqrt(sum2/n - sum^2/n^3).
inline void HostBlockMeanRms(const float* v, int n, float& mean, float& rms) {
	double sum = 0.0, sum2 = 0.0;
	for (int i = 0; i < n; ++i) {
		sum += v[i];
		sum2 += static_cast<double>(v[i]) * v[i];
	}
	const double fn = n;
	mean = static_cast<float>(sum / fn);
	rms = static_cast<float>(std::sqrt(sum2 / fn - (sum * sum) / (fn * fn * fn)));
}

// Stage harness kernels are defined in stage_harness.cu (the only translation
// unit that includes CvxCompress_CUDA_kernels.cu). Test files launch them.
__global__ void harness_wavelet_forward(const float* in, float* out);
__global__ void harness_wavelet_inverse(const float* in, float* out);
__global__ void harness_wavelet_roundtrip(const float* in, float* out);
__global__ void harness_mean_rms(const float* in, float* out_mean, float* out_rms);
__global__ void harness_normalize_global(const float* in, float scale, float gmean,
                                         float grms, float* out_val, float* out_mean,
                                         float* out_mult);
__global__ void harness_normalize_local(const float* in, float scale, float* out_val,
                                        float* out_mean, float* out_mult);
__global__ void harness_rle_encode(const float* in, char* bytes, int* out_esc, int* out_pay);
__global__ void harness_rle_decode(const char* bytes, int esc, float* out);
