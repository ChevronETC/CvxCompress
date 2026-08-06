#ifndef CVX_WAVELET_RUN_LENGTH_ENCODE_SLOW_HXX
#define CVX_WAVELET_RUN_LENGTH_ENCODE_SLOW_HXX

/*!
 * Perform quantization and run length encoding. 
 * This implementation is slow, but correct. 
 * It is used for module tests of fast implementation.
 *
 * Arguments:
 * scale      - input samples are multiplied by scale before quantization.
 * vals       - input samples.
 * num        - length of vals array.
 * compressed - compressed values are written to this block.
 * bytepos    - on output, contains number of bytes written to compressed.
 * error      - non zero if an error occured.
 *
 */
void 
Run_Length_Encode_Slow(
	float scale,
	float* vals, 
	int num, 
	unsigned long* compressed, 
	int& bytepos
	);

/*!
 * Decode array of floats that was encoded with Run_Length_Encode_Slow.
 *
 * Returns the number of values decoded, or -1 if the stream encoded a run
 * longer than num_expected_vals. In that case the write is clamped to the
 * caller's array and its contents should not be trusted.
 */
int 
Run_Length_Decode_Slow(
		float scale, 
		float* vals, 
		int num_expected_vals,
		unsigned long* compressed
		);

/*!
 * Compare two run length encoded blocks.
 * Used in module tests, to compare results from fast and slow run length encoder.
 *
 * Arguments:
 * compressed  - 1st run length encoded block.
 * bytepos     - number of bytes in 1st run length encoded block.
 * compressed2 - 2nd run length encoded block.
 * bytepos2    - number of bytes in 2nd run length encoded block.
 *
 */
bool 
Run_Length_Encode_Compare(
	unsigned long* compressed, 
	int bytepos, 
	unsigned long* compressed2, 
	int bytepos2
	);

#endif
