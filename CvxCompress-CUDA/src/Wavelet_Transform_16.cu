// Antonini 7-9 forward coefficients
#define al0  8.526986790094000e-001f
#define al1  3.774028556126500e-001f
#define al2 -1.106244044184200e-001f
#define al3 -2.384946501938001e-002f
#define al4  3.782845550699501e-002f

#define ah0  7.884856164056601e-001f
#define ah1 -4.180922732222101e-001f
#define ah2 -4.068941760955800e-002f
#define ah3  6.453888262893799e-002f

// Antonini 7-9 inverse coefficients
#define sl0  7.884856164056601e-001f
#define sl1  4.180922732222101e-001f
#define sl2 -4.068941760955800e-002f
#define sl3 -6.453888262893799e-002f

#define sh0  8.526986790094000e-001f
#define sh1 -3.774028556126500e-001f
#define sh2 -1.106244044184200e-001f
#define sh3  2.384946501938001e-002f
#define sh4  3.782845550699501e-002f

__device__ void __cuDs79_16(
	float& v00,
	float& v01,
	float& v02,
	float& v03,
	float& v04,
	float& v05,
	float& v06,
	float& v07,
	float& v08,
	float& v09,
	float& v0A,
	float& v0B,
	float& v0C,
	float& v0D,
	float& v0E,
	float& v0F
	)
{
	// 16 pt
	float w0F = ah3*(v0C+v0C) + ah2*(v0D+v0D) + ah1*(v0E+v0E) + ah0*v0F;
	float w0E = ah3*(v0A+v0E) + ah2*(v0B+v0F) + ah1*(v0C+v0E) + ah0*v0D;
	float w0D = ah3*(v08+v0E) + ah2*(v09+v0D) + ah1*(v0A+v0C) + ah0*v0B;
	float w0C = ah3*(v06+v0C) + ah2*(v07+v0B) + ah1*(v08+v0A) + ah0*v09;
	float w0B = ah3*(v04+v0A) + ah2*(v05+v09) + ah1*(v06+v08) + ah0*v07;
	float w0A = ah3*(v02+v08) + ah2*(v03+v07) + ah1*(v04+v06) + ah0*v05;
	float w09 = ah3*(v00+v06) + ah2*(v01+v05) + ah1*(v02+v04) + ah0*v03;
	float w08 = ah3*(v02+v04) + ah2*(v01+v03) + ah1*(v00+v02) + ah0*v01;
	float w07 = al4*(v0A+v0C) + al3*(v0B+v0D) + al2*(v0C+v0E) + al1*(v0D+v0F) + al0*v0E;
	float w06 = al4*(v08+v0E) + al3*(v09+v0F) + al2*(v0A+v0E) + al1*(v0B+v0D) + al0*v0C;
	float w05 = al4*(v06+v0E) + al3*(v07+v0D) + al2*(v08+v0C) + al1*(v09+v0B) + al0*v0A;
	float w04 = al4*(v04+v0C) + al3*(v05+v0B) + al2*(v06+v0A) + al1*(v07+v09) + al0*v08;
	float w03 = al4*(v02+v0A) + al3*(v03+v09) + al2*(v04+v08) + al1*(v05+v07) + al0*v06;
	float w02 = al4*(v00+v08) + al3*(v01+v07) + al2*(v02+v06) + al1*(v03+v05) + al0*v04;
	float w01 = al4*(v02+v06) + al3*(v01+v05) + al2*(v00+v04) + al1*(v01+v03) + al0*v02;
	float w00 = al4*(v04+v04) + al3*(v03+v03) + al2*(v02+v02) + al1*(v01+v01) + al0*v00;
	v0F = w0F;
	v0E = w0E;
	v0D = w0D;
	v0C = w0C;
	v0B = w0B;
	v0A = w0A;
	v09 = w09;
	v08 = w08;
	// 8 pt
	v07 = ah3*(w04+w04) + ah2*(w05+w05) + ah1*(w06+w06) + ah0*w07;
	v06 = ah3*(w02+w06) + ah2*(w03+w07) + ah1*(w04+w06) + ah0*w05;
	v05 = ah3*(w00+w06) + ah2*(w01+w05) + ah1*(w02+w04) + ah0*w03;
	v04 = ah3*(w02+w04) + ah2*(w01+w03) + ah1*(w00+w02) + ah0*w01;
	v03 = al4*(w02+w04) + al3*(w03+w05) + al2*(w04+w06) + al1*(w05+w07) + al0*w06;
	v02 = al4*(w00+w06) + al3*(w01+w07) + al2*(w02+w06) + al1*(w03+w05) + al0*w04;
	v01 = al4*(w02+w06) + al3*(w01+w05) + al2*(w00+w04) + al1*(w01+w03) + al0*w02;
	v00 = al4*(w04+w04) + al3*(w03+w03) + al2*(w02+w02) + al1*(w01+w01) + al0*w00;
	// 4 pt
	w03 = ah3*(v00+v00) + ah2*(v01+v01) + ah1*(v02+v02) + ah0*v03;
	w02 = ah3*(v02+v02) + ah2*(v01+v03) + ah1*(v00+v02) + ah0*v01;
	w01 = al4*(v02+v00) + al3*(v01+v01) + al2*(v00+v02) + al1*(v01+v03) + al0*v02;
	w00 = al4*(v02+v02) + al3*(v03+v03) + al2*(v02+v02) + al1*(v01+v01) + al0*v00;
	v03 = w03;
	v02 = w02;
	// 2 pt
	v01 = ah3*(w00+w00) + ah2*(w01+w01) + ah1*(w00+w00) + ah0*w01;
	v00 = al4*(w00+w00) + al3*(w01+w01) + al2*(w00+w00) + al1*(w01+w01) + al0*w00;
}

__device__ void __cuUs79_16(
	float& v00,
	float& v01,
	float& v02,
	float& v03,
	float& v04,
	float& v05,
	float& v06,
	float& v07,
	float& v08,
	float& v09,
	float& v0A,
	float& v0B,
	float& v0C,
	float& v0D,
	float& v0E,
	float& v0F
	)
{
	// 2 pt
	float w01 = sh4*(v01+v01) + sh2*(v01+v01) + sh0*v01 + sl3*(v00+v00) + sl1*(v00+v00);
	float w00 = sh3*(v01+v01) + sh1*(v01+v01) + sl2*(v00+v00) + sl0*v00;
	v01 = w01;
	v00 = w00;
	// 4 pt
	float w03 = sh4*(v02+v02) + sh2*(v02+v02) + sh0*v03 + sl3*(v00+v00) + sl1*(v01+v01);
	float w02 = sh3*(v02+v02) + sh1*(v02+v03) + sl2*(v00+v01) + sl0*v01;
	w01 = sh4*(v03+v02) + sh2*(v02+v03) + sh0*v02 + sl3*(v01+v01) + sl1*(v00+v01);
	w00 = sh3*(v03+v03) + sh1*(v02+v02) + sl2*(v01+v01) + sl0*v00;
	v03 = w03;
	v02 = w02;
	v01 = w01;
	v00 = w00;
	// 8 pt
	float w07 = sh4*(v05+v05) + sh2*(v06+v06) + sh0*v07 + sl3*(v02+v02) + sl1*(v03+v03);
	float w06 = sh3*(v05+v06) + sh1*(v06+v07) + sl2*(v02+v03) + sl0*v03;
	float w05 = sh4*(v04+v06) + sh2*(v05+v07) + sh0*v06 + sl3*(v01+v03) + sl1*(v02+v03);
	float w04 = sh3*(v04+v07) + sh1*(v05+v06) + sl2*(v01+v03) + sl0*v02;
	w03 = sh4*(v04+v07) + sh2*(v04+v06) + sh0*v05 + sl3*(v00+v03) + sl1*(v01+v02);
	w02 = sh3*(v04+v06) + sh1*(v04+v05) + sl2*(v00+v02) + sl0*v01;
	w01 = sh4*(v05+v06) + sh2*(v04+v05) + sh0*v04 + sl3*(v01+v02) + sl1*(v00+v01);
	w00 = sh3*(v05+v05) + sh1*(v04+v04) + sl2*(v01+v01) + sl0*v00;
	v07 = w07;
	v06 = w06;
	v05 = w05;
	v04 = w04;
	v03 = w03;
	v02 = w02;
	v01 = w01;
	v00 = w00;
	// 16 pt
	float w0F = sh4*(v0D+v0D) + sh2*(v0E+v0E) + sh0*v0F + sl3*(v06+v06) + sl1*(v07+v07);
	float w0E = sh3*(v0D+v0E) + sh1*(v0E+v0F) + sl2*(v06+v07) + sl0*v07;
	float w0D = sh4*(v0C+v0E) + sh2*(v0D+v0F) + sh0*v0E + sl3*(v05+v07) + sl1*(v06+v07);
	float w0C = sh3*(v0C+v0F) + sh1*(v0D+v0E) + sl2*(v05+v07) + sl0*v06;
	float w0B = sh4*(v0B+v0F) + sh2*(v0C+v0E) + sh0*v0D + sl3*(v04+v07) + sl1*(v05+v06);
	float w0A = sh3*(v0B+v0E) + sh1*(v0C+v0D) + sl2*(v04+v06) + sl0*v05;
	float w09 = sh4*(v0A+v0E) + sh2*(v0B+v0D) + sh0*v0C + sl3*(v03+v06) + sl1*(v04+v05);
	float w08 = sh3*(v0A+v0D) + sh1*(v0B+v0C) + sl2*(v03+v05) + sl0*v04;
	w07 = sh4*(v09+v0D) + sh2*(v0A+v0C) + sh0*v0B + sl3*(v02+v05) + sl1*(v03+v04);
	w06 = sh3*(v09+v0C) + sh1*(v0A+v0B) + sl2*(v02+v04) + sl0*v03;
	w05 = sh4*(v08+v0C) + sh2*(v09+v0B) + sh0*v0A + sl3*(v01+v04) + sl1*(v02+v03);
	w04 = sh3*(v08+v0B) + sh1*(v09+v0A) + sl2*(v01+v03) + sl0*v02;
	w03 = sh4*(v08+v0B) + sh2*(v08+v0A) + sh0*v09 + sl3*(v00+v03) + sl1*(v01+v02);
	w02 = sh3*(v08+v0A) + sh1*(v08+v09) + sl2*(v00+v02) + sl0*v01;
	w01 = sh4*(v09+v0A) + sh2*(v08+v09) + sh0*v08 + sl3*(v01+v02) + sl1*(v00+v01);
	w00 = sh3*(v09+v09) + sh1*(v08+v08) + sl2*(v01+v01) + sl0*v00;
	v0F = w0F;
	v0E = w0E;
	v0D = w0D;
	v0C = w0C;
	v0B = w0B;
	v0A = w0A;
	v09 = w09;
	v08 = w08;
	v07 = w07;
	v06 = w06;
	v05 = w05;
	v04 = w04;
	v03 = w03;
	v02 = w02;
	v01 = w01;
	v00 = w00;
}

// Transpose fast (1) and slow (3) axes.
__device__ void __cuTranspose13(
	float* buf,
	float& v00,
	float& v01,
	float& v02,
	float& v03,
	float& v04,
	float& v05,
	float& v06,
	float& v07,
	float& v08,
	float& v09,
	float& v0A,
	float& v0B,
	float& v0C,
	float& v0D,
	float& v0E,
	float& v0F
	)
{
	for (int thr_group = 0;  thr_group < 16;  thr_group += 4)
	{
		int thr_z = threadIdx.z - thr_group;
		if (thr_z >= 0 && thr_z < 4)
		{
			buf[thr_z*272+threadIdx.x] = v00;
			buf[17+thr_z*272+threadIdx.x] = v01;
			buf[34+thr_z*272+threadIdx.x] = v02;
			buf[51+thr_z*272+threadIdx.x] = v03;
			buf[68+thr_z*272+threadIdx.x] = v04;
			buf[85+thr_z*272+threadIdx.x] = v05;
			buf[102+thr_z*272+threadIdx.x] = v06;
			buf[119+thr_z*272+threadIdx.x] = v07;
			buf[136+thr_z*272+threadIdx.x] = v08;
			buf[153+thr_z*272+threadIdx.x] = v09;
			buf[170+thr_z*272+threadIdx.x] = v0A;
			buf[187+thr_z*272+threadIdx.x] = v0B;
			buf[204+thr_z*272+threadIdx.x] = v0C;
			buf[221+thr_z*272+threadIdx.x] = v0D;
			buf[238+thr_z*272+threadIdx.x] = v0E;
			buf[255+thr_z*272+threadIdx.x] = v0F;
		}
		__syncthreads();
		if (thr_z >= 0 && thr_z < 4)
		{
			v00 = buf[thr_z*272+threadIdx.x*17];
			v01 = buf[thr_z*272+threadIdx.x*17+1];
			v02 = buf[thr_z*272+threadIdx.x*17+2];
			v03 = buf[thr_z*272+threadIdx.x*17+3];
			v04 = buf[thr_z*272+threadIdx.x*17+4];
			v05 = buf[thr_z*272+threadIdx.x*17+5];
			v06 = buf[thr_z*272+threadIdx.x*17+6];
			v07 = buf[thr_z*272+threadIdx.x*17+7];
			v08 = buf[thr_z*272+threadIdx.x*17+8];
			v09 = buf[thr_z*272+threadIdx.x*17+9];
			v0A = buf[thr_z*272+threadIdx.x*17+10];
			v0B = buf[thr_z*272+threadIdx.x*17+11];
			v0C = buf[thr_z*272+threadIdx.x*17+12];
			v0D = buf[thr_z*272+threadIdx.x*17+13];
			v0E = buf[thr_z*272+threadIdx.x*17+14];
			v0F = buf[thr_z*272+threadIdx.x*17+15];
		}
		__syncthreads();
	}
}

// Transpose medium (2) and slow (3) axis
__device__ void __cuTranspose23(
	float* buf,
	float& v00,
	float& v01,
	float& v02,
	float& v03,
	float& v04,
	float& v05,
	float& v06,
	float& v07,
	float& v08,
	float& v09,
	float& v0A,
	float& v0B,
	float& v0C,
	float& v0D,
	float& v0E,
	float& v0F
	)
{
	float w00,w01,w02,w03,w04,w05,w06,w07,w08,w09,w0A,w0B,w0C,w0D,w0E,w0F;
	buf[0+threadIdx.z*16+threadIdx.x] = v00;
	buf[256+threadIdx.z*16+threadIdx.x] = v01;
	buf[512+threadIdx.z*16+threadIdx.x] = v02;
	buf[768+threadIdx.z*16+threadIdx.x] = v03;
	buf[1024+threadIdx.z*16+threadIdx.x] = v04;
	buf[1280+threadIdx.z*16+threadIdx.x] = v05;
	buf[1536+threadIdx.z*16+threadIdx.x] = v06;
	buf[1792+threadIdx.z*16+threadIdx.x] = v07;
	__syncthreads();
	if (threadIdx.z < 8)
	{
		w00 = buf[threadIdx.x+threadIdx.z*256+0];
		w01 = buf[threadIdx.x+threadIdx.z*256+16];
		w02 = buf[threadIdx.x+threadIdx.z*256+32];
		w03 = buf[threadIdx.x+threadIdx.z*256+48];
		w04 = buf[threadIdx.x+threadIdx.z*256+64];
		w05 = buf[threadIdx.x+threadIdx.z*256+80];
		w06 = buf[threadIdx.x+threadIdx.z*256+96];
		w07 = buf[threadIdx.x+threadIdx.z*256+112];
		w08 = buf[threadIdx.x+threadIdx.z*256+128];
		w09 = buf[threadIdx.x+threadIdx.z*256+144];
		w0A = buf[threadIdx.x+threadIdx.z*256+160];
		w0B = buf[threadIdx.x+threadIdx.z*256+176];
		w0C = buf[threadIdx.x+threadIdx.z*256+192];
		w0D = buf[threadIdx.x+threadIdx.z*256+208];
		w0E = buf[threadIdx.x+threadIdx.z*256+224];
		w0F = buf[threadIdx.x+threadIdx.z*256+240];
	}
	__syncthreads();
	buf[0+threadIdx.z*16+threadIdx.x] = v08;
	buf[256+threadIdx.z*16+threadIdx.x] = v09;
	buf[512+threadIdx.z*16+threadIdx.x] = v0A;
	buf[768+threadIdx.z*16+threadIdx.x] = v0B;
	buf[1024+threadIdx.z*16+threadIdx.x] = v0C;
	buf[1280+threadIdx.z*16+threadIdx.x] = v0D;
	buf[1536+threadIdx.z*16+threadIdx.x] = v0E;
	buf[1792+threadIdx.z*16+threadIdx.x] = v0F;
	__syncthreads();
	if (threadIdx.z >= 8)
	{
		w00 = buf[threadIdx.x+(threadIdx.z-8)*256+0];
		w01 = buf[threadIdx.x+(threadIdx.z-8)*256+16];
		w02 = buf[threadIdx.x+(threadIdx.z-8)*256+32];
		w03 = buf[threadIdx.x+(threadIdx.z-8)*256+48];
		w04 = buf[threadIdx.x+(threadIdx.z-8)*256+64];
		w05 = buf[threadIdx.x+(threadIdx.z-8)*256+80];
		w06 = buf[threadIdx.x+(threadIdx.z-8)*256+96];
		w07 = buf[threadIdx.x+(threadIdx.z-8)*256+112];
		w08 = buf[threadIdx.x+(threadIdx.z-8)*256+128];
		w09 = buf[threadIdx.x+(threadIdx.z-8)*256+144];
		w0A = buf[threadIdx.x+(threadIdx.z-8)*256+160];
		w0B = buf[threadIdx.x+(threadIdx.z-8)*256+176];
		w0C = buf[threadIdx.x+(threadIdx.z-8)*256+192];
		w0D = buf[threadIdx.x+(threadIdx.z-8)*256+208];
		w0E = buf[threadIdx.x+(threadIdx.z-8)*256+224];
		w0F = buf[threadIdx.x+(threadIdx.z-8)*256+240];
	}
	__syncthreads();
	v00 = w00;
	v01 = w01;
	v02 = w02;
	v03 = w03;
	v04 = w04;
	v05 = w05;
	v06 = w06;
	v07 = w07;
	v08 = w08;
	v09 = w09;
	v0A = w0A;
	v0B = w0B;
	v0C = w0C;
	v0D = w0D;
	v0E = w0E;
	v0F = w0F;
}

// buf is a float array in shared memory of at least 2048 elements.
__device__ __forceinline__ void _cuDs79XYZ_16(
	float* buf,
	float& v00,
	float& v01,
	float& v02,
	float& v03,
	float& v04,
	float& v05,
	float& v06,
	float& v07,
	float& v08,
	float& v09,
	float& v0A,
	float& v0B,
	float& v0C,
	float& v0D,
	float& v0E,
	float& v0F
	)
{
	// input is ordered X-Z-Y
	__cuTranspose13(buf,v00,v01,v02,v03,v04,v05,v06,v07,v08,v09,v0A,v0B,v0C,v0D,v0E,v0F);
	// vXX is now ordered Y-Z-X
	// X forward wavelet transform
	__cuDs79_16(v00,v01,v02,v03,v04,v05,v06,v07,v08,v09,v0A,v0B,v0C,v0D,v0E,v0F);
	__cuTranspose13(buf,v00,v01,v02,v03,v04,v05,v06,v07,v08,v09,v0A,v0B,v0C,v0D,v0E,v0F);
	// vXX is now ordered X-Z-Y again
	// Y forward wavelet transform
	__cuDs79_16(v00,v01,v02,v03,v04,v05,v06,v07,v08,v09,v0A,v0B,v0C,v0D,v0E,v0F);
	__cuTranspose23(buf,v00,v01,v02,v03,v04,v05,v06,v07,v08,v09,v0A,v0B,v0C,v0D,v0E,v0F);
	// vXX is now ordered X-Y-Z
	// Z forward wavelet transform
	__cuDs79_16(v00,v01,v02,v03,v04,v05,v06,v07,v08,v09,v0A,v0B,v0C,v0D,v0E,v0F);
	// output is ordered X-Y-Z
}

// buf is a float array in shared memory of at least 2048 elements.
__device__ __forceinline__ void _cuUs79XYZ_16(
	float* buf,
	float& v00,
	float& v01,
	float& v02,
	float& v03,
	float& v04,
	float& v05,
	float& v06,
	float& v07,
	float& v08,
	float& v09,
	float& v0A,
	float& v0B,
	float& v0C,
	float& v0D,
	float& v0E,
	float& v0F
	)
{
	// input is ordered X-Y-Z
	// Z inverse wavelet transform
	__cuUs79_16(v00,v01,v02,v03,v04,v05,v06,v07,v08,v09,v0A,v0B,v0C,v0D,v0E,v0F);
	__cuTranspose23(buf,v00,v01,v02,v03,v04,v05,v06,v07,v08,v09,v0A,v0B,v0C,v0D,v0E,v0F);
	// vXX is now ordered X-Z-Y
	// Y inverse wavelet transform
        __cuUs79_16(v00,v01,v02,v03,v04,v05,v06,v07,v08,v09,v0A,v0B,v0C,v0D,v0E,v0F);
	__cuTranspose13(buf,v00,v01,v02,v03,v04,v05,v06,v07,v08,v09,v0A,v0B,v0C,v0D,v0E,v0F);
	// vXX is now ordered Y-Z-X
	// X inverse wavelet transform
        __cuUs79_16(v00,v01,v02,v03,v04,v05,v06,v07,v08,v09,v0A,v0B,v0C,v0D,v0E,v0F);
	__cuTranspose13(buf,v00,v01,v02,v03,v04,v05,v06,v07,v08,v09,v0A,v0B,v0C,v0D,v0E,v0F);
	// output is ordered X-Z-Y
}

