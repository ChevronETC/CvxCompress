#ifndef CVX_CVXCOMPRESS_H
#define CVX_CVXCOMPRESS_H

float cvx_compress(
	float         scale,
	float        *vol,
	int           nx,
	int           ny,
	int           nz,
	int           bx,
	int           by,
	int           bz,
	unsigned int *compressed,
	long         *compressed_length);

float*  cvx_decompress_outofplace(
	int           *nx,
	int           *ny,
	int           *nz,
	unsigned int  *compressed,
	long           compressed_length);

void  cvx_decompress_inplace(
	float         *vol,
	int            nx,
	int            ny,
	int            nz,
	unsigned int  *compressed,
	long           compressed_length);

#endif
