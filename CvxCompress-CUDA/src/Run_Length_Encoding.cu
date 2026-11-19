#include <cuda_runtime_api.h>
#include <Run_Length_Escape_Codes.hxx>

__device__ void ____cuReduceSum(int* buf, int thread_sum_a, int& sum_a)
{
	int thread_num = (threadIdx.z*16+threadIdx.x) & 31;
#ifdef OLDCUDA
	int warp_sum_a = __shfl(thread_sum_a,thread_num+1) + thread_sum_a;
	warp_sum_a += __shfl(warp_sum_a,thread_num+2);
	warp_sum_a += __shfl(warp_sum_a,thread_num+4);
	warp_sum_a += __shfl(warp_sum_a,thread_num+8);
	warp_sum_a += __shfl(warp_sum_a,thread_num+16);
#else
	int warp_sum_a = __shfl_sync(0xFFFFFFFF,thread_sum_a,thread_num+1) + thread_sum_a;
	warp_sum_a += __shfl_sync(0xFFFFFFFF,warp_sum_a,thread_num+2);
	warp_sum_a += __shfl_sync(0xFFFFFFFF,warp_sum_a,thread_num+4);
	warp_sum_a += __shfl_sync(0xFFFFFFFF,warp_sum_a,thread_num+8);
	warp_sum_a += __shfl_sync(0xFFFFFFFF,warp_sum_a,thread_num+16);
#endif
	int num_warps = (blockDim.z*blockDim.x) / 32;
	int warp_num = (threadIdx.z*16+threadIdx.x) / 32;
	if (thread_num == 0)
	{
		buf[warp_num] = warp_sum_a;
	}
	__syncthreads();
	sum_a = buf[0];
	for (int i = 1;  i < num_warps;  ++i)
	{
		sum_a += buf[i];
	}
	__syncthreads();
}

__device__ void ____cuReduceSum(int* buf, int thread_sum_a, int thread_sum_b, int& sum_a, int& sum_b)
{
	int thread_num = (threadIdx.z*16+threadIdx.x) & 31;
#ifdef OLDCUDA
	int warp_sum_a = __shfl(thread_sum_a,thread_num+1) + thread_sum_a;
	warp_sum_a += __shfl(warp_sum_a,thread_num+2);
	warp_sum_a += __shfl(warp_sum_a,thread_num+4);
	warp_sum_a += __shfl(warp_sum_a,thread_num+8);
	warp_sum_a += __shfl(warp_sum_a,thread_num+16);
	int warp_sum_b = __shfl(thread_sum_b,thread_num+1) + thread_sum_b;
	warp_sum_b += __shfl(warp_sum_b,thread_num+2);
	warp_sum_b += __shfl(warp_sum_b,thread_num+4);
	warp_sum_b += __shfl(warp_sum_b,thread_num+8);
	warp_sum_b += __shfl(warp_sum_b,thread_num+16);
#else
	int warp_sum_a = __shfl_sync(0xFFFFFFFF,thread_sum_a,thread_num+1) + thread_sum_a;
	warp_sum_a += __shfl_sync(0xFFFFFFFF,warp_sum_a,thread_num+2);
	warp_sum_a += __shfl_sync(0xFFFFFFFF,warp_sum_a,thread_num+4);
	warp_sum_a += __shfl_sync(0xFFFFFFFF,warp_sum_a,thread_num+8);
	warp_sum_a += __shfl_sync(0xFFFFFFFF,warp_sum_a,thread_num+16);
	int warp_sum_b = __shfl_sync(0xFFFFFFFF,thread_sum_b,thread_num+1) + thread_sum_b;
	warp_sum_b += __shfl_sync(0xFFFFFFFF,warp_sum_b,thread_num+2);
	warp_sum_b += __shfl_sync(0xFFFFFFFF,warp_sum_b,thread_num+4);
	warp_sum_b += __shfl_sync(0xFFFFFFFF,warp_sum_b,thread_num+8);
	warp_sum_b += __shfl_sync(0xFFFFFFFF,warp_sum_b,thread_num+16);
#endif
	int num_warps = (blockDim.z*blockDim.x) / 32;
	int warp_num = (threadIdx.z*16+threadIdx.x) / 32;
	if (thread_num == 0)
	{
		buf[warp_num] = warp_sum_a;
		buf[num_warps+warp_num] = warp_sum_b;
	}
	__syncthreads();
	sum_a = buf[0];
	sum_b = buf[num_warps];
	for (int i = 1;  i < num_warps;  ++i)
	{
		sum_a += buf[i];
		sum_b += buf[num_warps+i];
	}
	__syncthreads();
}

__device__ void ____cuReduceSum(float* buf, float a, float b, float& sum_a, float& sum_b)
{
	int thread_num = (threadIdx.z*16+threadIdx.x) & 31;
#ifdef OLDCUDA
	float warp_sum_a = __shfl(a,thread_num+1) + a;
	warp_sum_a += __shfl(warp_sum_a,thread_num+2);
	warp_sum_a += __shfl(warp_sum_a,thread_num+4);
	warp_sum_a += __shfl(warp_sum_a,thread_num+8);
	warp_sum_a += __shfl(warp_sum_a,thread_num+16);
	float warp_sum_b = __shfl(b,thread_num+1) + b;
	warp_sum_b += __shfl(warp_sum_b,thread_num+2);
	warp_sum_b += __shfl(warp_sum_b,thread_num+4);
	warp_sum_b += __shfl(warp_sum_b,thread_num+8);
	warp_sum_b += __shfl(warp_sum_b,thread_num+16);
#else
	float warp_sum_a = __shfl_sync(0xFFFFFFFF,a,thread_num+1) + a;
	warp_sum_a += __shfl_sync(0xFFFFFFFF,warp_sum_a,thread_num+2);
	warp_sum_a += __shfl_sync(0xFFFFFFFF,warp_sum_a,thread_num+4);
	warp_sum_a += __shfl_sync(0xFFFFFFFF,warp_sum_a,thread_num+8);
	warp_sum_a += __shfl_sync(0xFFFFFFFF,warp_sum_a,thread_num+16);
	float warp_sum_b = __shfl_sync(0xFFFFFFFF,b,thread_num+1) + b;
	warp_sum_b += __shfl_sync(0xFFFFFFFF,warp_sum_b,thread_num+2);
	warp_sum_b += __shfl_sync(0xFFFFFFFF,warp_sum_b,thread_num+4);
	warp_sum_b += __shfl_sync(0xFFFFFFFF,warp_sum_b,thread_num+8);
	warp_sum_b += __shfl_sync(0xFFFFFFFF,warp_sum_b,thread_num+16);
#endif
	int num_warps = (blockDim.z*blockDim.x) / 32;
	int warp_num = (threadIdx.z*16+threadIdx.x) / 32;
	if (thread_num == 0)
	{
		buf[warp_num] = warp_sum_a;
		buf[num_warps+warp_num] = warp_sum_b;
	}
	__syncthreads();
	sum_a = buf[0];
	sum_b = buf[num_warps];
	for (int i = 1;  i < num_warps;  ++i)
	{
		sum_a += buf[i];
		sum_b += buf[num_warps+i];
	}
	__syncthreads();
}

__device__ void ___cuComputeMeanRMS(
	float* buf,
	float* val,
	float& mean,
	float& rms
	)
{
	float thread_sum = 0.0f, thread_sum2 = 0.0f;
	for (int i = 0;  i < 16;  ++i)
	{
		thread_sum += val[i];
		thread_sum2 += val[i] * val[i];
	}
	float sum, sum2;
	____cuReduceSum(buf,thread_sum,thread_sum2,sum,sum2);
	int nn = blockDim.z*blockDim.x*16;
	float fn = (float)nn;
	float fn3 = fn*fn*fn;
	mean = sum / fn;
	rms = sqrtf(sum2/fn - sum*sum/fn3);
}

__device__ int ____cuConflict_Free_Index_Int(int index)
{
	return index + (((index >> 6) + (index >> 11)) << 1);
}
//#define ____cuConflict_Free_Index_Int(index) (index+(((index>>6)+(index>>11))<<1))

__device__ void ___cuPrescanSum(short* temp, int thrid, int n)
{
	int n_2 = n >> 1;
	int offset = 1;
	for (int d = n_2;  d > 1;  d >>= 1)
	{
		__syncthreads();
		for (int thid = thrid;  thid < d;  thid += 256)
		{
			int ai = ____cuConflict_Free_Index_Int(offset*(2*thid+1)-1);
			int bi = ____cuConflict_Free_Index_Int(offset*(2*thid+2)-1);
			temp[bi] += temp[ai];
		}
		offset <<= 1;
	}
	if (thrid == 0) temp[____cuConflict_Free_Index_Int(n-int(1))] = 0;
	for (int d = 1;  d <= n_2;  d <<= 1)
	{
		__syncthreads();
		for (int thid = thrid;  thid < d;  thid += 256)
		{
			int ai = ____cuConflict_Free_Index_Int(offset*(2*thid+1)-1);
			int bi = ____cuConflict_Free_Index_Int(offset*(2*thid+2)-1);
			int t = temp[ai];
			temp[ai]  = temp[bi];
			temp[bi] += t;
		}
		offset >>= 1;
	}
	__syncthreads();
}
__device__ void ___cuPrescanMax(short* temp, int thrid, int n)
{
	int n_2 = n >> 1;
	int offset = 1;
	for (int d = n_2;  d > 1;  d >>= 1)
	{
		__syncthreads();
		for (int thid = thrid;  thid < d;  thid += 256)
		{
			int ai = ____cuConflict_Free_Index_Int(offset*(2*thid+1)-1);
			int bi = ____cuConflict_Free_Index_Int(offset*(2*thid+2)-1);
			temp[bi] = max(temp[ai],temp[bi]);
		}
		offset <<= 1;
	}
	if (thrid == 0) temp[____cuConflict_Free_Index_Int(n-int(1))] = 0;
	for (int d = 1;  d <= n_2;  d <<= 1)
	{
		__syncthreads();
		for (int thid = thrid;  thid < d;  thid += 256)
		{
			int ai = ____cuConflict_Free_Index_Int(offset*(2*thid+1)-1);
			int bi = ____cuConflict_Free_Index_Int(offset*(2*thid+2)-1);
			int t = temp[ai];
			temp[ai]  = temp[bi];
			temp[bi] = max(t,temp[bi]);
		}
		offset >>= 1;
	}
	__syncthreads();
}

__device__ void ___cuComputeEscapePayloadLen(float val, int rle, int& escape_length, int& payload_length)
{
	escape_length = (rle >= 0);
	payload_length = (val <= (float)VLESC2 || val >= (float)RLESC3)*2 + (val < -32768.0f || val > 32767.0f) + (val < -8388608.0f || val > 8388607.0f) + (rle > 1) + (rle > 255);
}

__device__ char ___cuComputeEscapeCode(float val, int rle)
{
	if (rle > 255) return (char)RLESC3;
	else if (rle > 1) return (char)RLESC1;
	else 
	{
		if (val > (float)VLESC2 && val < (float)RLESC3)
			return (char)val;
		else if (val >= -32768.0f && val <= 32767.0f)
			return VLESC2;
		else if (val >= -8388608.0f && val <= 8388607.0f)
			return VLESC3;
		else
			return VLESC4;
	}
}

__device__ int ___cuComputePayloadLen(float val, int rle)
{
	return (val <= (float)VLESC2 || val >= (float)RLESC3)*2 + (val < -32768.0f || val > 32767.0f) + (val < -8388608.0f || val > 8388607.0f) + (rle > 1) + (rle > 255);
}

__device__ void __cuComputeRunLength4096(
	short* temp,
	int flags
	)
{
	int thrid = threadIdx.z * 16 + threadIdx.x;
	// lower 16 bits : flag that indicates if current rle == 1
	for (int i = 0;  i < 16;  ++i) temp[____cuConflict_Free_Index_Int(thrid+256*i)] = ((flags >> i) & 1);
	__syncthreads();
	// upper 16 bits : flag that indicates if current rle == 1 and next rle == 0.
	//                 this means current index is an rle output.
	for (int i = 0;  i < 16;  ++i) flags |= ((thrid == 255 && i == 15) || (temp[____cuConflict_Free_Index_Int(thrid+1+256*i)] == 0)) << (i+16);
	flags = flags & ((flags << 16) | 65535);
	___cuPrescanSum(temp,thrid,4096);
	if (thrid == 255) temp[____cuConflict_Free_Index_Int(4096)] = temp[____cuConflict_Free_Index_Int(4095)]+((flags>>15)&1);
	__syncthreads();
	int rle[16];
	for (int i = 0;  i < 16;  ++i) rle[i] = (flags & (1<<(16+i))) ? temp[____cuConflict_Free_Index_Int(thrid+1+256*i)] : 0;
	__syncthreads();
	for (int i = 0;  i < 16;  ++i) temp[____cuConflict_Free_Index_Int(thrid+256*i)] = rle[i];
	___cuPrescanMax(temp,thrid,4096);
	for (int i = 0;  i < 16;  ++i)
	{
		rle[i] = rle[i] - (rle[i] > 0 ? temp[____cuConflict_Free_Index_Int(thrid+256*i)] : (flags>>i)&1);
		temp[____cuConflict_Free_Index_Int(thrid+256*i)] = rle[i];
	}
	__syncthreads();
}

/*
 * This method returns the total length of the compressed block in number of bytes.
 */
__device__ void __cuComputeCompressedSize4096(
	short* temp,
	float* val,
	int& escape_length,
        int& payload_length
	)
{
	int thrid = threadIdx.z*16+threadIdx.x;
	int thread_escape_length = 0, tread_payload_length = 0;
	for (int i = 0;  i < 16;  ++i)
	{
		int rle = temp[____cuConflict_Free_Index_Int(thrid+256*i)];
		int esc_len, pay_len;
		___cuComputeEscapePayloadLen(val[i],rle,esc_len,pay_len);
		thread_escape_length += esc_len;
		tread_payload_length += pay_len;
	}
	____cuReduceSum((int*)(temp+4230),thread_escape_length,tread_payload_length,escape_length,payload_length);
}

__device__ int __cuCountNaNs(short* temp, float* val)
{
	int thread_NaN_count = 0;
	for (int i = 0;  i < 16;  ++i) if (!isfinite(val[i])) ++thread_NaN_count;
	int total_NaN_count;
	____cuReduceSum((int*)temp,thread_NaN_count,total_NaN_count);
	return total_NaN_count;
}

__device__ void _cuNormalize(
	short* temp,
	float scale,
	float* val,
	float& mean,
	float& multiplier
	)
{
	short NaN_count = 0;
	__cuCountNaNs(temp,val);
	if (NaN_count > 0)
	{
		mean = 0.0f;
		multiplier = -1.0f;
	}
	else
	{
		___cuComputeMeanRMS((float*)temp,val,mean,multiplier);
		multiplier = multiplier > 0.0f ? 1.0f / (multiplier*scale) : 1.0f;
		for (int i = 0;  i < 16;  ++i) val[i] = (val[i] - mean) * multiplier;
	}
}

/*
 * Normalize 4096-element block using a pre-computed global RMS.
 * Falls back to multiplier=1 when global_rms==0.
 */
__device__ void _cuNormalizeGlobal(
	short* temp,
	float scale,
	float* val,
	float global_mean,
	float global_rms,
	float& mean,
	float& multiplier
	)
{
	if (__cuCountNaNs(temp, val) > 0)
	{
		mean = 0.0f;
		multiplier = -1.0f;
	}
	else
	{
		mean = 0.0f; // modify to use global_mean if needed
		multiplier = global_rms > 0.0f ? 1.0f / (global_rms * scale) : 1.0f;
		for (int i = 0;  i < 16;  ++i) val[i] = val[i] * multiplier;
	}
}

__device__ void _cuEncode(
	short* temp,
	float* val,
	char* dest,
	int escape_length
	)
{
	int thrid = threadIdx.z*16+threadIdx.x;
	// compute byte offset for each escape code
	int rle[16];
	for (int i = 0;  i < 16;  ++i)
	{
		rle[i] = temp[____cuConflict_Free_Index_Int(thrid+i*256)];
		temp[____cuConflict_Free_Index_Int(thrid+i*256)] = (rle[i] >= 0);
		int esc_code = ___cuComputeEscapeCode(val[i],rle[i]);
		int pay_leng = ___cuComputePayloadLen(val[i],rle[i]);
		rle[i] = (rle[i] << 16) | (pay_leng << 8) | (esc_code & 0xFF);
	}
	___cuPrescanSum(temp,thrid,4096);
	// write out escape codes
	for (int i = 0;  i < 16;  ++i) if ((rle[i]>>16) >= 0) dest[temp[____cuConflict_Free_Index_Int(thrid+i*256)]] = ((rle[i]<<24)>>24);
	for (int i = 0;  i < 16;  ++i) temp[____cuConflict_Free_Index_Int(thrid+i*256)] = (rle[i] >> 8) & 0xFF;
	for (int i = 0;  i < 16;  ++i) rle[i] = rle[i] >> 16;
	___cuPrescanSum(temp,thrid,4096);
	__syncthreads();
	// write out payloads
	for (int i = 0;  i < 16;  ++i) 
	{
		unsigned char* pay_p = (unsigned char*)(dest + escape_length + temp[____cuConflict_Free_Index_Int(thrid+i*256)]);
		if (rle[i] > 255)
		{
			pay_p[0] = (unsigned char)(rle[i]&0xFF);
			pay_p[1] = (unsigned char)((rle[i]>>8)&0xFF);
		}
		else if (rle[i] > 1)
		{
			pay_p[0] = (unsigned char)rle[i];
		}
		else if (!(val[i] > (float)VLESC2 && val[i] < (float)RLESC3))
		{
			if (val[i] >= -32768.0f && val[i] <= 32767.0f)
			{
				// VLESC2
				int ival = (int)val[i];
				pay_p[0] = (unsigned char)(ival & 0xFF);
				pay_p[1] = (unsigned char)((ival >> 8) & 0xFF);
			}
			else if (val[i] >= -8388608.0f && val[i] <= 8388607.0f)
			{
				// VLESC3
				int ival = (int)val[i];
				pay_p[0] = (unsigned char)(ival & 0xFF);
				pay_p[1] = (unsigned char)((ival >> 8) & 0xFF);
				pay_p[2] = (unsigned char)((ival >> 16) & 0xFF);
			}
			else
			{
				// VLESC4
				unsigned int fval = *((unsigned int*)(val+i));
				pay_p[0] = (unsigned char)(fval & 0xFF);
				pay_p[1] = (unsigned char)((fval >> 8) & 0xFF);
				pay_p[2] = (unsigned char)((fval >> 16) & 0xFF);
				pay_p[3] = (unsigned char)((fval >> 24) & 0xFF);
			}
		}
		__syncthreads();  // make threads converge
	}
}

__device__ short ___cuComputePayloadSize(char esc)
{
	return (esc == RLESC1) + (esc == RLESC3 || esc == VLESC2)*2 + (esc == VLESC3)*3 + (esc == VLESC4)*4;
}

__device__ short ___cuComputeNumCells(char esc, char* payload)
{
	if (esc == RLESC1 || esc == RLESC3)
	{
		unsigned char rle0 = ((unsigned char*)payload)[0];
		unsigned char rle1 = (esc == RLESC3) ? ((unsigned char*)payload)[1] : 0;
		return (short)rle0 | ((short)rle1 << 8);
	}
	else
		return 1;
}

__device__ void __cuDecode(
	short* temp,
	char* c_blk,
	int escape_length,
	float* val
	)
{
	int thrid = threadIdx.z * 16 + threadIdx.x;

	// compute payload offsets
	int maxi = 0;
	for (maxi = 256;  maxi < 4096 && maxi < escape_length;  maxi<<=1);
	maxi = maxi >> 8;
	int pay_off_esc[16];
	for (int i = 0;  i < maxi;  ++i)
	{
		int esc = (thrid+i*256) < escape_length ? (int)c_blk[thrid+i*256] : 0;
		temp[____cuConflict_Free_Index_Int(thrid+i*256)] = ___cuComputePayloadSize((char)esc);
		pay_off_esc[i] = esc;
	}
	___cuPrescanSum(temp,thrid,maxi<<8);
	/*
	for (int j = 0;  j < escape_length;  ++j)
	{
		for (int i = 0;  i < 16;  ++i)
		{
			if (thrid+i*256 == j)
			{
				printf("%04d :: Esc=%4d, Payload_Offset=%d\n",j,pay_off_esc[i],temp[____cuConflict_Free_Index_Int(thrid+i*256)]);
			}
		}
		__syncthreads();
	}
	__syncthreads();
	*/
	// at this point, payload offset for each escape code is in temp(thrid+i*256)

	// compute cell index range for each escape code
	for (int i = 0;  i < maxi;  ++i)
	{
		int pay_off = temp[____cuConflict_Free_Index_Int(thrid+i*256)] + escape_length;
		temp[____cuConflict_Free_Index_Int(thrid+i*256)] = ___cuComputeNumCells((char)pay_off_esc[i],c_blk+pay_off);
		pay_off_esc[i] = (pay_off_esc[i]) << 16 | (pay_off & 0xFFFF);
	}
	if (thrid == 255) temp[____cuConflict_Free_Index_Int(maxi<<8)] = temp[____cuConflict_Free_Index_Int((maxi<<8)-1)];
	___cuPrescanSum(temp,thrid,maxi<<8);
	if (thrid == 255) temp[____cuConflict_Free_Index_Int(maxi<<8)] += temp[____cuConflict_Free_Index_Int((maxi<<8)-1)];
	__syncthreads();
	int cell_index_range[16];
	for (int i = 0;  i < maxi;  ++i)
	{
		int lo_range = (int)temp[____cuConflict_Free_Index_Int(thrid+i*256)];
		int hi_range = (int)temp[____cuConflict_Free_Index_Int(thrid+i*256+1)] - 1;
		// both lo_range and hi_range is guaranteed to be >= 0.
		cell_index_range[i] = lo_range | (hi_range << 16);
	}
	__syncthreads();
	/*
	for (int j = 0;  j < escape_length;  ++j)
	{
		for (int i = 0;  i < 16;  ++i)
		{
			if (thrid+i*256 == j)
			{
				int escape_code = pay_off_esc[i] >> 16;
				int payload_offset = pay_off_esc[i] & 0xFFFF;
				int lo_range = cell_index_range[i] & 0xFFFF;
				int hi_range = cell_index_range[i] >> 16;
				if (lo_range < 2056 && hi_range > 2040) printf("%04d :: Esc=%4d, Payload_Offset=%4d, Cell_Indices=[%d,%d]\n",j,escape_code,payload_offset,lo_range,hi_range);
				for (int k = lo_range;  k <= hi_range;  ++k)
				{
					if (k >= 2048 && k <= 2052)
					{
						float dv = 0.0f;
						unsigned char* p_payload = (unsigned char*)(c_blk + payload_offset);
						switch (escape_code)
						{
							case RLESC1:
							case RLESC3:
								dv = 0.0f;
								break;
							case VLESC2:
								{
									short pay0 = (short)(((unsigned char*)p_payload)[0]);
									short pay1 = (short)(((unsigned char*)p_payload)[1]);
									short sval = pay0 | (pay1 << 8);
									dv = (float)sval;
								}
								break;
							case VLESC3:
								{
									int pay0 = (int)(((unsigned char*)p_payload)[0]);
									int pay1 = (int)(((unsigned char*)p_payload)[1]);
									int pay2 = (int)(((unsigned char*)p_payload)[2]);
									int ival = pay0 | (pay1 << 8) | ((pay2 << 24) >> 8);
									dv = (float)ival;
								}
								break;
							case VLESC4:
								{
									int pay0 = (int)(((unsigned char*)p_payload)[0]);
									int pay1 = (int)(((unsigned char*)p_payload)[1]);
									int pay2 = (int)(((unsigned char*)p_payload)[2]);
									int pay3 = (int)(((unsigned char*)p_payload)[3]);
									int ival = pay0 | (pay1 << 8) | (pay2 << 16) | (pay3 << 24);
									dv = *((float*)&ival);
								}
								break;
							default:
								dv = (float)escape_code;
								break;
						}
						printf("...%04d :: v = %e\n",k,dv);
					}
				}
			}
		}
		__syncthreads();
	}
	__syncthreads();
	*/
	// at this point, cell index range for each escape code is in cell_index_range; bits 31-16 contain high range, bits 15-0 low range.
	// at this point, pay_off_esc contains escape code (bits 23-16) and payload offset (bits 15-0).
	
	// write out decoded values
	for (int i = 0;  i < 16;  i+=8)
	{
		int first_cell_out = i*256;
		for (int j = 0;  j < i+8 && j < maxi;  ++j)
		{
			int first_cell = cell_index_range[j] & 0xFFFF;
			int last_cell = cell_index_range[j] >> 16;
			first_cell = first_cell > first_cell_out ? first_cell : first_cell_out;
			last_cell = last_cell < first_cell_out+2047 ? last_cell : first_cell_out+2047;
			if (first_cell <= last_cell)
			{
				// VLESC2,3,4
				int escape_code = pay_off_esc[j] >> 16;
				unsigned char* p_payload = (unsigned char*)(c_blk + (pay_off_esc[j] & 0xFFFF));
				float fval = (escape_code == RLESC1 || escape_code == RLESC3) ? 0.0f : (float)escape_code;
				if (escape_code == VLESC2 || escape_code == VLESC3 || escape_code == VLESC4)
				{
					int pay0 = (int)(((unsigned char*)p_payload)[0]);
					int pay1 = (int)(((unsigned char*)p_payload)[1]);
					if (escape_code == VLESC2)
					{
						fval = (float)(pay0 | ((pay1 << 24) >> 16));
					}
					else if (escape_code == VLESC3 || escape_code == VLESC4)
					{
						int pay2 = (int)(((unsigned char*)p_payload)[2]);
						if (escape_code == VLESC3)
						{
							fval = (float)(pay0 | (pay1 << 8) | ((pay2 << 24) >> 8));
						}
						else
						{
							int pay3 = (int)(((unsigned char*)p_payload)[3]);
							*((int*)&fval) = pay0 | (pay1 << 8) | (pay2 << 16) | (pay3 << 24);
						}
					}
				}
				((float*)temp)[first_cell-first_cell_out] = fval;
				for (int k = first_cell+1;  k <= last_cell;  ++k) ((float*)temp)[k-first_cell_out] = 0.0f;
			}
			//if (first_cell < 0 || last_cell >= 4096)
			//{
			//	printf("__cuDecode :: invalid cell index range [%d,%d]\n",first_cell,last_cell);
			//}
			__syncthreads();
		}
		__syncthreads();
		for (int k = i;  k < i+8;  ++k) val[k] = ((float*)temp)[thrid+(k-i)*256];
		__syncthreads();
	}
}
