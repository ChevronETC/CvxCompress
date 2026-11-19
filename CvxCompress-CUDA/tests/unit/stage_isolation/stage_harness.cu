#include <cuda_runtime_api.h>
#include <CvxCompress_CUDA_kernels.cu>

#include "stage_harness.cuh"

#define THRID (threadIdx.z * 16 + threadIdx.x)

// ---------- Wavelet stage ----------

__global__ void harness_wavelet_forward(const float* in, float* out) {
	__shared__ short buf[4262];
	int t = THRID;
	float v[16];
	for (int i = 0; i < 16; ++i) v[i] = in[t + i * 256];
	_cuDs79XYZ_16((float*)buf, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
	              v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]);
	for (int i = 0; i < 16; ++i) out[t + i * 256] = v[i];
}

__global__ void harness_wavelet_inverse(const float* in, float* out) {
	__shared__ short buf[4262];
	int t = THRID;
	float v[16];
	for (int i = 0; i < 16; ++i) v[i] = in[t + i * 256];
	_cuUs79XYZ_16((float*)buf, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
	              v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]);
	for (int i = 0; i < 16; ++i) out[t + i * 256] = v[i];
}

__global__ void harness_wavelet_roundtrip(const float* in, float* out) {
	__shared__ short buf[4262];
	int t = THRID;
	float v[16];
	for (int i = 0; i < 16; ++i) v[i] = in[t + i * 256];
	_cuDs79XYZ_16((float*)buf, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
	              v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]);
	_cuUs79XYZ_16((float*)buf, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
	              v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]);
	for (int i = 0; i < 16; ++i) out[t + i * 256] = v[i];
}

// ---------- Quantization stage ----------

__global__ void harness_mean_rms(const float* in, float* out_mean, float* out_rms) {
	__shared__ short buf[4262];
	int t = THRID;
	float v[16];
	for (int i = 0; i < 16; ++i) v[i] = in[t + i * 256];
	float mean, rms;
	___cuComputeMeanRMS((float*)buf, v, mean, rms);
	if (t == 0) {
		*out_mean = mean;
		*out_rms = rms;
	}
}

__global__ void harness_normalize_global(const float* in, float scale, float gmean,
                                         float grms, float* out_val, float* out_mean,
                                         float* out_mult) {
	__shared__ short buf[4262];
	int t = THRID;
	float v[16];
	for (int i = 0; i < 16; ++i) v[i] = in[t + i * 256];
	float mean, mult;
	_cuNormalizeGlobal(buf, scale, v, gmean, grms, mean, mult);
	for (int i = 0; i < 16; ++i) out_val[t + i * 256] = v[i];
	if (t == 0) {
		*out_mean = mean;
		*out_mult = mult;
	}
}

__global__ void harness_normalize_local(const float* in, float scale, float* out_val,
                                        float* out_mean, float* out_mult) {
	__shared__ short buf[4262];
	int t = THRID;
	float v[16];
	for (int i = 0; i < 16; ++i) v[i] = in[t + i * 256];
	float mean, mult;
	_cuNormalize(buf, scale, v, mean, mult);
	for (int i = 0; i < 16; ++i) out_val[t + i * 256] = v[i];
	if (t == 0) {
		*out_mean = mean;
		*out_mult = mult;
	}
}

// ---------- Run-length encoding stage ----------

__global__ void harness_rle_encode(const float* in, char* bytes, int* out_esc, int* out_pay) {
	__shared__ short buf[4262];
	int t = THRID;
	float v[16];
	for (int i = 0; i < 16; ++i) v[i] = in[t + i * 256];
	int flags = 0;
	for (int i = 0; i < 16; ++i) flags |= (((int)v[i] == 0) << i);
	__cuComputeRunLength4096(buf, flags);
	int esc, pay;
	__cuComputeCompressedSize4096(buf, v, esc, pay);
	_cuEncode(buf, v, bytes, esc);
	if (t == 0) {
		*out_esc = esc;
		*out_pay = pay;
	}
}

__global__ void harness_rle_decode(const char* bytes, int esc, float* out) {
	__shared__ short buf[4262];
	int t = THRID;
	float v[16];
	if (esc == 1)
		for (int i = 0; i < 16; ++i) v[i] = 0.0f;
	else
		__cuDecode(buf, (char*)bytes, esc, v);
	for (int i = 0; i < 16; ++i) out[t + i * 256] = v[i];
}
