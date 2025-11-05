#include <assert.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#define SIMDE_ENABLE_NATIVE_ALIASES
#include "simde/x86/avx512.h"  // SSE intrinsics

#include "Run_Length_Escape_Codes.hxx"

// un-comment if you want debug printouts during encoding
//#define DEBUG_ENCODE
//#define DEBUG_DECODE

// un-comment if you want individual byte mem read and writes
//#define BYTEIO

#define TMJ_AVX_RLE

inline void EncodeRLE_Slow(int& rle, char* dst, int& bytepos)
{
	if (rle > 0)
	{
		if (rle == 1)
		{
#ifdef DEBUG_ENCODE
			printf("BYTE encode %d at index %d\n",(char)0,bytepos);
#endif
			dst[bytepos++] = (char)0;
		}
		/*
		else if (rle == 2)
		{
			dst[bytepos++] = (char)0;
			dst[bytepos++] = (char)0;
		}
		*/
		else if (rle < 256)
		{
#ifdef DEBUG_ENCODE
			printf("RLESC1 rle = %d at index %d\n",rle,bytepos);
#endif
#ifndef BYTEIO
			rle = (RLESC1 & 0xFF) | ((rle & 0xFF) << 8);
			*((short*)(dst+bytepos)) = rle;
			bytepos += 2;
#else
			dst[bytepos++] = (char)RLESC1;
			dst[bytepos++] = (unsigned char)rle;
#endif
		}
		else
		{
#ifdef DEBUG_ENCODE
			printf("RLESC3 rle = %d at index %d\n",rle,bytepos);
#endif
#ifndef BYTEIO
			rle = (RLESC3 & 0xFF) | ((rle & 0xFFFFFF) << 8);
			*((int*)(dst+bytepos)) = rle;
			bytepos += 4;
#else
			dst[bytepos++] = (char)RLESC3;
			dst[bytepos++] = rle & 0xFF;
			dst[bytepos++] = (rle >> 8) & 0xFF;
			dst[bytepos++] = (rle >> 16) & 0xFF;
#endif
		}
		rle = 0;
	}
}

inline void Encode_One_Float(float fval, int ival, int& rle, char* dst, int& bytepos)
{
	if (ival == 0)
	{
		++rle;
	}
	else
	{
		EncodeRLE_Slow(rle,dst,bytepos);
		if (ival > VLESC2 && ival < RLESC3)
		{
#ifdef DEBUG_ENCODE
			printf("BYTE encode %d at index %d\n",(char)ival,bytepos);
#endif
			dst[bytepos++] = (char)ival;
		}
		else if (ival >= -32768 && ival <= 32767)
		{
#ifdef DEBUG_ENCODE
			printf("VLESC2 encode %d at index %d\n",ival,bytepos);
#endif
#ifndef BYTEIO
			ival = (VLESC2 & 0xFF) | ((ival & 0xFFFF) << 8);
			*((int*)(dst+bytepos)) = ival;
			bytepos += 3;
#else
			dst[bytepos++] = (char)VLESC2;
			dst[bytepos++] = ival & 0xFF;
			dst[bytepos++] = (ival >> 8) & 0xFF;
#endif
		}
		else if (ival >= -8388608 && ival <= 8388607)
		{
#ifdef DEBUG_ENCODE
			printf("VLESC3 encode %d at index %d\n",ival,bytepos);
#endif
#ifndef BYTEIO
			ival = (VLESC3 & 0xFF) | ((ival & 0xFFFFFF) << 8);
			*((int*)(dst+bytepos)) = ival;
			bytepos += 4;
#else
			dst[bytepos++] = (char)VLESC3;
			dst[bytepos++] = ival & 0xFF;
			dst[bytepos++] = (ival >> 8) & 0xFF;
			dst[bytepos++] = (ival >> 16) & 0xFF;
#endif
		}
		else
		{
#ifdef DEBUG_ENCODE
			printf("VLESC4 encode %e at index %d\n",fval,bytepos);
#endif
			dst[bytepos++] = (char)VLESC4;
#ifndef BYTEIO
			*((float*)(dst+bytepos)) = fval;
			bytepos += 4;
#else
			unsigned int ifval = *((int*)&fval);
			dst[bytepos++] = ifval & 0xFF;
			dst[bytepos++] = (ifval >> 8) & 0xFF;
			dst[bytepos++] = (ifval >> 16) & 0xFF;
			dst[bytepos++] = (ifval >> 24) & 0xFF;
#endif
		}
	}
}

inline void Encode_One_Word(bool is_zero, int esc, int payload, int nbytes, int& rle, char* dst, int& bytepos, int ival)
{
	if (is_zero)
	{
		//assert(ival == 0);
		++rle;
	}
	else
	{
		EncodeRLE_Slow(rle,dst,bytepos);
		long rval = (long)esc | ((long)payload << 8);
		*((long*)(dst+bytepos)) = rval;
		//dst[bytepos] = esc;
		//*((int*)(dst+bytepos+1)) = payload;
		bytepos += nbytes;
		//assert(!(nbytes == 1) || (ival > VLESC2 && ival < RLESC3));
		//assert(!(nbytes == 3) || (ival >= -32768 && ival <= 32767));
		//assert(!(nbytes == 4) || (ival >= -8388608 && ival <= 8388607));
	}
}

inline void Encode_One_Word(int i, int zeros, int* p_esc, int* p_payload, int* p_nbytes, int& rle, char* dst, int& bytepos)
{
	if (zeros & (1 << i))
	{
		++rle;
	}
	else
	{
		EncodeRLE_Slow(rle,dst,bytepos);
		long rval = (long)p_esc[i] | ((long)p_payload[i] << 8);
		*((long*)(dst+bytepos)) = rval;
		bytepos += p_nbytes[i];
	}
}

inline int count_true(__m256 predicate)
{
	__m128i sum = _mm_hadd_epi32(_mm256_castsi256_si128(_mm256_castps_si256(predicate)),_mm256_extractf128_si256(_mm256_castps_si256(predicate),1));
	sum = _mm_hadd_epi32(sum,sum);
	sum = _mm_hadd_epi32(sum,sum);
	return -_mm_extract_epi32(sum,0);
}

/*
 * Run length encode a quantized block. 
 * The encoded block is usually smaller, but can be up to 25% larger after encoding.
 * Please ensure that work block (compressed ptr) is at least 5*sizeof(float)*num/4 bytes.
 */
void Run_Length_Encode_Slow(float scale, float* vals, int num, unsigned long* compressed, int& bytepos)
{
	int rle = 0;
	char* dst = (char*)compressed;
#ifdef TMJ_AVX_RLE
	__m256 _mm_scale = _mm256_set1_ps(scale);
	__m256 _mm_byte_lo = _mm256_cvtepi32_ps(_mm256_set1_epi32(VLESC2));
	__m256 _mm_byte_hi = _mm256_cvtepi32_ps(_mm256_set1_epi32(RLESC3));
	__m256 _mm_short_lo = _mm256_cvtepi32_ps(_mm256_set1_epi32(-32768));
	__m256 _mm_short_hi = _mm256_cvtepi32_ps(_mm256_set1_epi32(32767));
	__m256 _mm_i3_lo = _mm256_cvtepi32_ps(_mm256_set1_epi32(-8388608));
	__m256 _mm_i3_hi = _mm256_cvtepi32_ps(_mm256_set1_epi32(8388607));
	for (int i = 0;  i < num;  i+=8)
	{
		__m256 fvals = _mm256_mul_ps(_mm_scale,_mm256_load_ps(vals+i));
		__m256i ivals = _mm256_cvttps_epi32(fvals);
		__m256 fivals = _mm256_cvtepi32_ps(ivals);
		__m256 is_zero = _mm256_cmp_ps(fivals,_mm256_setzero_ps(),0);  // .EQ_OQ.
		int zeros = _mm256_movemask_ps(is_zero);
		if (zeros == 255)
		{
			// all zeros
			rle += 8;
		}
		else
		{
			__m256 is_byte = _mm256_and_ps(_mm256_cmp_ps(fivals,_mm_byte_lo,30),_mm256_cmp_ps(fivals,_mm_byte_hi,17));  // .GT_OQ. | .LT_OQ.
			if (zeros == 0 && _mm256_movemask_ps(is_byte) == 255)
			{
				// all bytes
				EncodeRLE_Slow(rle,dst,bytepos);
				__m128i loval = _mm_shuffle_epi8(_mm256_castsi256_si128(ivals),_mm_setr_epi8(0,4,8,12,0,4,8,12,128,128,128,128,128,128,128,128));
				__m128i hival = _mm_shuffle_epi8(_mm256_extractf128_si256(ivals,1),_mm_setr_epi8(0,4,8,12,0,4,8,12,128,128,128,128,128,128,128,128));
				__m128 both = _mm_blend_ps(_mm_castsi128_ps(loval),_mm_castsi128_ps(hival),10);
				long all_8_bytes = _mm_extract_epi64(_mm_castps_si128(both),0);
				*((long*)(dst+bytepos)) = all_8_bytes;
				bytepos += 8;
			}
			else
			{
				int num_bytes = count_true(is_byte);
				__m256 is_short = _mm256_and_ps(_mm256_cmp_ps(fivals,_mm_short_lo,29),_mm256_cmp_ps(fivals,_mm_short_hi,18));  // .GE_OQ. | .LE_OQ.
				if (zeros == 0 && _mm256_movemask_ps(is_short) == 255 && (num_bytes+(8-num_bytes)*3) > 17)
				{
					// all shorts
					EncodeRLE_Slow(rle,dst,bytepos);
					__m128i loval = _mm_shuffle_epi8(_mm256_castsi256_si128(ivals),_mm_setr_epi8(0,1,4,5,8,9,12,13,0,1,4,5,8,9,12,13));
					__m128i hival = _mm_shuffle_epi8(_mm256_extractf128_si256(ivals,1),_mm_setr_epi8(0,1,4,5,8,9,12,13,0,1,4,5,8,9,12,13));
					__m128 both = _mm_blend_ps(_mm_castsi128_ps(loval),_mm_castsi128_ps(hival),12);
					dst[bytepos] = (char)VLESC2_8x;
					_mm_storeu_ps((float*)(dst+bytepos+1),both);
					bytepos += 17;
				}
				else
				{
					int num_shorts = count_true(is_short);
					__m256 is_i3 = _mm256_and_ps(_mm256_cmp_ps(fivals,_mm_i3_lo,29),_mm256_cmp_ps(fivals,_mm_i3_hi,18));
					if (zeros == 0 && _mm256_movemask_ps(is_i3) == 255 && (num_bytes+(num_shorts-num_bytes)*3+(8-num_shorts)*4) > 25)
					{
						// all int3s
						EncodeRLE_Slow(rle,dst,bytepos);
						__m128i loval = _mm_shuffle_epi8(_mm256_castsi256_si128(ivals),_mm_setr_epi8(128,0,1,2,4,5,6,8,9,10,12,13,14,128,128,128));
						loval = _mm_insert_epi8(loval,VLESC3_8x,0);
						_mm_storeu_ps((float*)(dst+bytepos),_mm_castsi128_ps(loval));
						__m128i hival = _mm_shuffle_epi8(_mm256_extractf128_si256(ivals,1),_mm_setr_epi8(0,1,2,4,5,6,8,9,10,12,13,14,128,128,128,128));
						_mm_storeu_ps((float*)(dst+bytepos+13),_mm_castsi128_ps(hival));
						bytepos += 25;
					}
					else
					{
						is_i3 = _mm256_andnot_ps(is_short,is_i3);
						is_short = _mm256_andnot_ps(is_byte,is_short);
						is_byte = _mm256_andnot_ps(is_zero,is_byte);
						__m256 is_not_float = _mm256_or_ps(is_zero,_mm256_or_ps(is_byte,_mm256_or_ps(is_short,is_i3)));

						__m256 esc = _mm256_and_ps(is_byte,_mm256_and_ps(_mm256_castsi256_ps(_mm256_set1_epi32(0xFF)),_mm256_castsi256_ps(ivals)));
						esc = _mm256_or_ps(esc,_mm256_and_ps(is_short,_mm256_castsi256_ps(_mm256_set1_epi32(VLESC2&0xFF))));
						esc = _mm256_or_ps(esc,_mm256_and_ps(is_i3,_mm256_castsi256_ps(_mm256_set1_epi32(VLESC3&0xFF))));
						esc = _mm256_or_ps(esc,_mm256_andnot_ps(is_not_float,_mm256_castsi256_ps(_mm256_set1_epi32(VLESC4&0xFF))));

						__m256 payload = _mm256_and_ps(_mm256_or_ps(is_short,is_i3), _mm256_castsi256_ps(ivals));
						payload = _mm256_or_ps(payload, _mm256_andnot_ps(is_not_float,fvals));

						__m256 nbytes = _mm256_and_ps(is_byte,_mm256_castsi256_ps(_mm256_set1_epi32(1)));
						nbytes = _mm256_or_ps(nbytes,_mm256_and_ps(is_short,_mm256_castsi256_ps(_mm256_set1_epi32(3))));
						nbytes = _mm256_or_ps(nbytes,_mm256_and_ps(is_i3,_mm256_castsi256_ps(_mm256_set1_epi32(4))));
						nbytes = _mm256_or_ps(nbytes,_mm256_andnot_ps(is_not_float,_mm256_castsi256_ps(_mm256_set1_epi32(5))));

						int* p_ival = (int*)(&ivals);
						int* p_esc = (int*)(&esc);
						int* p_payload = (int*)(&payload);
						int* p_nbytes = (int*)(&nbytes);
						Encode_One_Word(0,zeros,p_esc,p_payload,p_nbytes,rle,dst,bytepos);
						Encode_One_Word(1,zeros,p_esc,p_payload,p_nbytes,rle,dst,bytepos);
						Encode_One_Word(2,zeros,p_esc,p_payload,p_nbytes,rle,dst,bytepos);
						Encode_One_Word(3,zeros,p_esc,p_payload,p_nbytes,rle,dst,bytepos);
						Encode_One_Word(4,zeros,p_esc,p_payload,p_nbytes,rle,dst,bytepos);
						Encode_One_Word(5,zeros,p_esc,p_payload,p_nbytes,rle,dst,bytepos);
						Encode_One_Word(6,zeros,p_esc,p_payload,p_nbytes,rle,dst,bytepos);
						Encode_One_Word(7,zeros,p_esc,p_payload,p_nbytes,rle,dst,bytepos);
					}
				}
			}
		}
	}
	EncodeRLE_Slow(rle,dst,bytepos);
#else
	__m256 _mm_scale = _mm256_set1_ps(scale);
	for (int i = 0;  i < num;  i+=8)
	{
		__m256 fvals = _mm256_mul_ps(_mm_scale,_mm256_load_ps(vals+i));
		__m256i ivals = _mm256_cvttps_epi32(fvals);
		float* fval = (float*)(&fvals);
		int* ival = (int*)(&ivals);
		Encode_One_Float(fval[0],ival[0],rle,dst,bytepos);
		Encode_One_Float(fval[1],ival[1],rle,dst,bytepos);
		Encode_One_Float(fval[2],ival[2],rle,dst,bytepos);
		Encode_One_Float(fval[3],ival[3],rle,dst,bytepos);
		Encode_One_Float(fval[4],ival[4],rle,dst,bytepos);
		Encode_One_Float(fval[5],ival[5],rle,dst,bytepos);
		Encode_One_Float(fval[6],ival[6],rle,dst,bytepos);
		Encode_One_Float(fval[7],ival[7],rle,dst,bytepos);
	}
	EncodeRLE_Slow(rle,dst,bytepos);
#endif
/*
	for (int i = 0;  i < num;  ++i)
	{
		float fval = scale*vals[i];
		//float fval = vals[i];
		int ival = (int)fval;
		if (ival == 0)
		{
			++rle;
		}
		else
		{
			EncodeRLE_Slow(rle,dst,bytepos);
			if (ival > VLESC2 && ival < RLESC3)
			{
#ifdef DEBUG_ENCODE
				printf("BYTE encode %d at index %d\n",(char)ival,bytepos);
#endif
				dst[bytepos++] = (char)ival;
			}
			else if (ival >= -32768 && ival <= 32767)
			{
#ifdef DEBUG_ENCODE
				printf("VLESC2 encode %d at index %d\n",ival,bytepos);
#endif
#ifndef BYTEIO
				ival = (VLESC2 & 0xFF) | ((ival & 0xFFFF) << 8);
				*((int*)(dst+bytepos)) = ival;
				bytepos += 3;
#else
				dst[bytepos++] = (char)VLESC2;
				dst[bytepos++] = ival & 0xFF;
				dst[bytepos++] = (ival >> 8) & 0xFF;
#endif
			}
			else if (ival >= -8388608 && ival <= 8388607)
			{
#ifdef DEBUG_ENCODE
				printf("VLESC3 encode %d at index %d\n",ival,bytepos);
#endif
#ifndef BYTEIO
				ival = (VLESC3 & 0xFF) | ((ival & 0xFFFFFF) << 8);
				*((int*)(dst+bytepos)) = ival;
				bytepos += 4;
#else
				dst[bytepos++] = (char)VLESC3;
				dst[bytepos++] = ival & 0xFF;
				dst[bytepos++] = (ival >> 8) & 0xFF;
				dst[bytepos++] = (ival >> 16) & 0xFF;
#endif
			}
			else
			{
#ifdef DEBUG_ENCODE
                                printf("VLESC4 encode %e at index %d\n",fval,bytepos);
#endif
                                dst[bytepos++] = (char)VLESC4;
#ifndef BYTEIO
				*((float*)(dst+bytepos)) = fval;
				bytepos += 4;
#else
				unsigned int ifval = *((int*)&fval);
                                dst[bytepos++] = ifval & 0xFF;
                                dst[bytepos++] = (ifval >> 8) & 0xFF;
                                dst[bytepos++] = (ifval >> 16) & 0xFF;
                                dst[bytepos++] = (ifval >> 24) & 0xFF;
#endif
			}
		}
	}
	EncodeRLE_Slow(rle,dst,bytepos);
	*/
}

int Run_Length_Decode_Slow(float scale, float* vals, int num_expected_vals, unsigned long* compressed)
{
	int num = 0;
	char* p = (char*)compressed;
	float scalefac = 1.0f / scale;
	for (;  num < num_expected_vals;  ++p)
	{
#ifndef __INTEL_COMPILER
		int val0 = ((int*)p)[0];
		int val1 = ((int*)p)[1];
		__m128i eight_bytes = _mm_setr_epi32(val0,val1,0,0);
#else
		__m128i eight_bytes = _mm_loadu_si64(p);
#endif
		__m128i is_bytes = _mm_and_si128(_mm_cmpgt_epi8(eight_bytes,_mm_set1_epi32(VLESC2)),_mm_cmplt_epi8(eight_bytes,_mm_set1_epi32(RLESC3)));
		if (num < (num_expected_vals-8) && _mm_movemask_epi8(is_bytes) == 65535)
		{
			// 8 byte values
			__m128i first_4_bytes = _mm_cvtepi8_epi32(eight_bytes);
			__m128i next_4_bytes = _mm_cvtepi8_epi32(_mm_shuffle_epi32(eight_bytes,29));
			_mm_storeu_ps((float*)(vals+num),_mm_mul_ps(_mm_set1_ps(scalefac),_mm_cvtepi32_ps(first_4_bytes)));
			_mm_storeu_ps((float*)(vals+num+4),_mm_mul_ps(_mm_set1_ps(scalefac),_mm_cvtepi32_ps(next_4_bytes)));
			num += 8;
			p += 8;
		}
		else
		{
			int ival = (signed char)*p;
			if (ival > VLESC2 && ival < RLESC3)
			{
#ifdef DEBUG_DECODE
				printf("  BYTE=%d, num=%d\n",ival,num);
				assert(num < num_expected_vals);
#endif
				vals[num++] = (float)ival * scalefac;
			}
			else if (ival == RLESC1)
			{
				int rle = ((unsigned char*)p)[1];
#ifdef DEBUG_DECODE
				printf("  RLESC1 rle=%d, num=%d, num_expected_vals=%d\n",rle,num,num_expected_vals);
				assert(num+rle <= num_expected_vals);
#endif
				for (int j = 0;  j < rle;  ++j) vals[num+j] = 0.0f;
				num += rle;
				p += 1;
			}
			else if (ival == RLESC3)
			{
				//int rle = (((unsigned char*)p)[3] << 16) | (((unsigned char*)p)[2] << 8) | ((unsigned char*)p)[1];
				int rle = *((unsigned int*)p) >> 8;
#ifdef DEBUG_DECODE
				printf("  RLESC3 rle=%d, num=%d, num_expected_vals=%d\n",rle,num,num_expected_vals);
				assert(num+rle <= num_expected_vals);
#endif
				for (int j = 0;  j < rle;  ++j) vals[num+j] = 0.0f;
				num += rle;
				p += 3;
			}
			else if (ival == VLESC2)
			{
				//int quant = (p[2] << 8) | ((unsigned char*)p)[1];
				short quant = *((short*)(p+1));
#ifdef DEBUG_DECODE
				printf("  VLESC2 quant=%d, num=%d\n",quant,num);
				assert(num < num_expected_vals);
#endif
				vals[num++] = (float)quant * scalefac;
				p += 2;
			}
			else if (ival == VLESC3)
			{
				//int quant = (p[3] << 16) | (((unsigned char*)p)[2] << 8) | ((unsigned char*)p)[1];
				int quant = *((int*)p) >> 8;
#ifdef DEBUG_DECODE
				printf("  VLESC3 quant=%d, num=%d\n",quant,num);
				assert(num < num_expected_vals);
#endif
				vals[num++] = (float)quant * scalefac;
				p += 3;
			}
			else if (ival == VLESC2_8x)
			{
#ifdef DEBUG_DECODE
				printf("  VLESC2_8x num=%d\n",num);
				assert(num < num_expected_vals);
#endif
				// 8 x shorts
				__m128i ivals = _mm_loadu_si128((__m128i*)(p+1));
				__m128i loval = _mm_cvtepi16_epi32(ivals);
				__m128i hival = _mm_cvtepi16_epi32(_mm_alignr_epi8(ivals,ivals,8));
				__m256i both = _mm256_insertf128_si256(_mm256_castsi128_si256(loval),hival,1);
				__m256 fvals = _mm256_cvtepi32_ps(both);
				fvals = _mm256_mul_ps(_mm256_set1_ps(scalefac),fvals);
				_mm256_storeu_ps(vals+num,fvals);
				num += 8;
				p += 16;
			}
			else if (ival == VLESC3_8x)
			{
#ifdef DEBUG_DECODE
				printf("  VLESC3_8x num=%d\n",num);
				assert(num < num_expected_vals);
#endif
				__m128i loval = _mm_loadu_si128((__m128i*)(p+1));
				loval = _mm_shuffle_epi8(loval,_mm_setr_epi8(128,0,1,2,128,3,4,5,128,6,7,8,128,9,10,11));
				loval = _mm_srai_epi32(loval,8);
				__m128i hival = _mm_loadu_si128((__m128i*)(p+13));
				hival = _mm_shuffle_epi8(hival,_mm_setr_epi8(128,0,1,2,128,3,4,5,128,6,7,8,128,9,10,11));
				hival = _mm_srai_epi32(hival,8);
				__m256i both = _mm256_insertf128_si256(_mm256_castsi128_si256(loval),hival,1);
				__m256 fvals = _mm256_cvtepi32_ps(both);
				fvals = _mm256_mul_ps(_mm256_set1_ps(scalefac),fvals);
				_mm256_storeu_ps(vals+num,fvals);
				num += 8;
				p += 24;
			}
			else if (ival == VLESC4)
			{
#ifdef DEBUG_DECODE
				printf("  VLESC4 = %e\n",*((float*)(p+1)));
				assert(num < num_expected_vals);
#endif
				vals[num++] = *((float*)(p+1)) * scalefac;
				p += 4;
			}
			/*
			else
			{
#ifdef DEBUG_DECODE
				printf("  BYTE=%d, num=%d\n",ival,num);
				assert(num < num_expected_vals);
#endif
				vals[num++] = (float)ival * scalefac;
			}
			*/
		}
	}
	return num;
}

bool Run_Length_Encode_Compare(unsigned long* compressed, int bytepos, unsigned long* compressed2, int bytepos2)
{
	bool retval = false;
	printf("bytepos = %d, bytepos2 = %d\n",bytepos,bytepos2);
	if (bytepos == bytepos2)
	{
		retval = true;
		char* p = (char*)compressed;
		char* p2 = (char*)compressed2;
		for (int i = 0;  i < bytepos;  ++i)
		{
#ifdef DEBUG_ENCODE
			printf("p[%4d] = %4d, p2[%4d] = %4d\n",i,p[i],i,p2[i]);
#endif
			if (p[i] != p2[i])
			{
#ifndef DEBUG_ENCODE
				printf("p[%4d] = %4d, p2[%4d] = %4d\n",i,p[i],i,p2[i]);
#endif
				printf("Arrays differ at byte %d\n",i);
				retval = false;
			}
		}
	}
	if (retval) printf("Arrays are identical\n"); else printf("Arrays differ\n");
	return retval;
}

