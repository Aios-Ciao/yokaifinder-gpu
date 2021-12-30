#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>
#include <stdio.h>

__global__ void calc_1_4col(
	unsigned char *work4col,			// 格納先 自分の要素は求めたグローバルインデックス*8からはじまる8byte
	const unsigned char *chrmask,		// 有効文字コードかの判定用ビットマスク
	const unsigned char *lut_x31F4,		// $31F4導出用のLUT
	const unsigned char *lut_x31F5		// $31F5導出用のLUT
)
{
	unsigned int gblidx = 8 * (((blockIdx.y * gridDim.x) + blockIdx.x) * blockDim.x + threadIdx.x);

	unsigned char	col1 = blockIdx.y;
	unsigned char	col2 = blockIdx.x >> 6;
	unsigned char	col3 = blockIdx.x & 0x3F;
	unsigned char	col4 = threadIdx.x;

	unsigned char	c = 0;					// キャリーフラグ

	// 一部オペコードをインライン化
	auto bitrev = [](unsigned int v) {	unsigned int wk = __brev(v); return (wk >> 24); };
	auto adc = [&c](unsigned short vl, unsigned short vr) { unsigned short wk = vl + vr + c; c = (wk >> 8) & 0x01; return (wk & 0xFF); };
	auto ror = [&c](unsigned char v) { unsigned char wc = c; c = v & 0x01; return (unsigned char)((v >> 1) | (wc << 7)); };
	auto bitcnt = [](unsigned long int v) { return (__popc(v)); };

	unsigned char pre31f4(0), pre31f5(0), pre31f7(0), pre31f8(0), pre31fa(1), pre31fb(0);
	unsigned char xor31f4, xor31f5, wk31fa;

	const unsigned char	unvalid = chrmask[col1] | chrmask[col2] | chrmask[col3] | chrmask[col4];
	const unsigned char chr1(col1 & 0x3F), chr2(col2 & 0x3F), chr3(col3 & 0x3F), chr4(col4 & 0x3F);

	// col1～col4で与えれるならびで計算をまわす
	xor31f4 = lut_x31F4[pre31f5]; xor31f5 = lut_x31F5[pre31f5];
	pre31f5 = pre31f4 ^ xor31f5;
	pre31f4 = bitrev(chr1) ^ xor31f4;	c = (pre31f4 >= 0xE5) ? 1 : 0;
	pre31f7 = adc(chr1, pre31f7);
	pre31f8 = adc(pre31f8, pre31f5);	wk31fa = ror(pre31fa);
	pre31fa = adc(wk31fa, chr1);
	pre31fb += c + bitcnt(chr1);

	xor31f4 = lut_x31F4[pre31f5]; xor31f5 = lut_x31F5[pre31f5];
	pre31f5 = pre31f4 ^ xor31f5;
	pre31f4 = bitrev(chr2) ^ xor31f4;	c = (pre31f4 >= 0xE5) ? 1 : 0;
	pre31f7 = adc(chr2, pre31f7);
	pre31f8 = adc(pre31f8, pre31f5);	wk31fa = ror(pre31fa);
	pre31fa = adc(wk31fa, chr2);
	pre31fb += c + bitcnt(chr2);

	xor31f4 = lut_x31F4[pre31f5]; xor31f5 = lut_x31F5[pre31f5];
	pre31f5 = pre31f4 ^ xor31f5;
	pre31f4 = bitrev(chr3) ^ xor31f4;	c = (pre31f4 >= 0xE5) ? 1 : 0;
	pre31f7 = adc(chr3, pre31f7);
	pre31f8 = adc(pre31f8, pre31f5);	wk31fa = ror(pre31fa);
	pre31fa = adc(wk31fa, chr3);
	pre31fb += c + bitcnt(chr3);

	xor31f4 = lut_x31F4[pre31f5]; xor31f5 = lut_x31F5[pre31f5];
	pre31f5 = pre31f4 ^ xor31f5;
	pre31f4 = bitrev(chr4) ^ xor31f4;	c = (pre31f4 >= 0xE5) ? 1 : 0;
	pre31f7 = adc(chr4, pre31f7);
	pre31f8 = adc(pre31f8, pre31f5);	wk31fa = ror(pre31fa);
	pre31fa = adc(wk31fa, chr4);
	pre31fb += c + bitcnt(chr4);

	/*$31F4*/	work4col[gblidx + 0] = pre31f4;
	/*$31F5*/	work4col[gblidx + 1] = pre31f5;
	/*$31F7*/	work4col[gblidx + 2] = pre31f7;
	/*$31F8*/	work4col[gblidx + 3] = pre31f8;
	/*$31F9*/	work4col[gblidx + 4] = unvalid | (chr1 ^ chr2 ^ chr3 ^ chr4);
	/*$31FA*/	work4col[gblidx + 5] = pre31fa;
	/*$31FB*/	work4col[gblidx + 6] = pre31fb;
	/*carry*/	work4col[gblidx + 7] = c;
				// F6は実質固定値なのでつめて次の桁に渡すキャリーを格納する
}

// これまでの桁の値を使って追加で4桁分の計算値を求める
__global__ void calc_4col(
	unsigned char		*work8col,		// 格納先 自分の要素は求めたグローバルインデックス*8からはじまる8byte
	const unsigned char	*work4col,		// これまでに計算してきた演算結果
	const unsigned char *chrmask,		// 有効文字コードかの判定用ビットマスク
	const unsigned char *lut_x31F4,		// $31F4導出用のLUT
	const unsigned char *lut_x31F5,		// $31F5導出用のLUT
	const unsigned int	idx4			// work4colに対応するインデックス値
)
{
	// スレッド番号から書き込み先インデックスを計算
	unsigned int gblidx = 8 * (((blockIdx.y * gridDim.x) + blockIdx.x) * blockDim.x + threadIdx.x);

	const unsigned char	col1 = blockIdx.y;
	const unsigned char	col2 = blockIdx.x >> 6;
	const unsigned char	col3 = blockIdx.x & 0x3F;
	const unsigned char	col4 = threadIdx.x;
	const unsigned int	rdidx = idx4 * 8;

	unsigned char pre31f4(work4col[rdidx + 0]);
	unsigned char pre31f5(work4col[rdidx + 1]);
	unsigned char pre31f7(work4col[rdidx + 2]);
	unsigned char pre31f8(work4col[rdidx + 3]);
	unsigned char pre31f9(work4col[rdidx + 4]);
	unsigned char pre31fa(work4col[rdidx + 5]);
	unsigned char pre31fb(work4col[rdidx + 6]);
	unsigned char	    c(work4col[rdidx + 7]);	// キャリーフラグ

	unsigned char xor31f4, xor31f5, wk31fa;

	// 一部オペコードをインライン化
	auto bitrev = [](unsigned int v) {	unsigned int wk = __brev(v); return (wk >> 24); };
	auto adc = [&c](unsigned short vl, unsigned short vr) { unsigned short wk = vl + vr + c; c = (wk >> 8) & 0x01; return (wk & 0xFF); };
	auto ror = [&c](unsigned char v) { unsigned char wc = c; c = v & 0x01; return (unsigned char)((v >> 1) | (wc << 7)); };
	auto bitcnt = [](unsigned long int v) { return (__popc(v)); };

	unsigned char	unvalid = chrmask[col1] | chrmask[col2] | chrmask[col3] | chrmask[col4];
	unsigned char chr1(col1 & 0x3F), chr2(col2 & 0x3F), chr3(col3 & 0x3F), chr4(col4 & 0x3F);

	// col1～col4で与えれるならびで計算をまわす
	xor31f4 = lut_x31F4[pre31f5]; xor31f5 = lut_x31F5[pre31f5];
	pre31f5 = pre31f4 ^ xor31f5;
	pre31f4 = bitrev(chr1) ^ xor31f4;	c = (pre31f4 >= 0xE5) ? 1 : 0;
	pre31f7 = adc(chr1, pre31f7);
	pre31f8 = adc(pre31f8, pre31f5);	wk31fa = ror(pre31fa);
	pre31fa = adc(wk31fa, chr1);
	pre31fb += c + bitcnt(chr1);

	xor31f4 = lut_x31F4[pre31f5]; xor31f5 = lut_x31F5[pre31f5];
	pre31f5 = pre31f4 ^ xor31f5;
	pre31f4 = bitrev(chr2) ^ xor31f4;	c = (pre31f4 >= 0xE5) ? 1 : 0;
	pre31f7 = adc(chr2, pre31f7);
	pre31f8 = adc(pre31f8, pre31f5);	wk31fa = ror(pre31fa);
	pre31fa = adc(wk31fa, chr2);
	pre31fb += c + bitcnt(chr2);

	xor31f4 = lut_x31F4[pre31f5]; xor31f5 = lut_x31F5[pre31f5];
	pre31f5 = pre31f4 ^ xor31f5;
	pre31f4 = bitrev(chr3) ^ xor31f4;	c = (pre31f4 >= 0xE5) ? 1 : 0;
	pre31f7 = adc(chr3, pre31f7);
	pre31f8 = adc(pre31f8, pre31f5);	wk31fa = ror(pre31fa);
	pre31fa = adc(wk31fa, chr3);
	pre31fb += c + bitcnt(chr3);

	xor31f4 = lut_x31F4[pre31f5]; xor31f5 = lut_x31F5[pre31f5];
	pre31f5 = pre31f4 ^ xor31f5;
	pre31f4 = bitrev(chr4) ^ xor31f4;	c = (pre31f4 >= 0xE5) ? 1 : 0;
	pre31f7 = adc(chr4, pre31f7);
	pre31f8 = adc(pre31f8, pre31f5);	wk31fa = ror(pre31fa);
	pre31fa = adc(wk31fa, chr4);
	pre31fb += c + bitcnt(chr4);


	/*$31F4*/	work8col[gblidx + 0] = pre31f4;
	/*$31F5*/	work8col[gblidx + 1] = pre31f5;
	/*$31F7*/	work8col[gblidx + 2] = pre31f7;
	/*$31F8*/	work8col[gblidx + 3] = pre31f8;
	/*$31F9*/	work8col[gblidx + 4] = unvalid | (pre31f9 ^ chr1 ^ chr2 ^ chr3 ^ chr4);
	/*$31FA*/	work8col[gblidx + 5] = pre31fa;
	/*$31FB*/	work8col[gblidx + 6] = pre31fb;
	/*carry*/	work8col[gblidx + 7] = c;
	// F6は実質固定値なのでつめて次の桁に渡すキャリーを格納する
}

// 残りの１桁導出と最終桁までのチェックディジット計算、有効判定
__global__ void calclast_validate(
	unsigned char		*result,	// OUT  14桁分の各データ計算結果
	unsigned char		*valid,		// OUT  パスワードとして成り立つなら1(true)でなければ0(false)
	const unsigned char	*work,		// IN	12桁目までで導出された途中値
	const unsigned char *lut_x31F4,	// IN	$31F4導出用のLUT
	const unsigned char *lut_x31F5,	// IN	$31F5導出用のLUT
	const unsigned char	*chrmask,	// IN	有効な文字コードかどうかのフラグマップ
	const unsigned int	offset		// IN	blockIdx.xのオフセット
)
{
	const unsigned int	gblidx = 8 * blockIdx.x + (offset * gridDim.x);	// ブロック番号xがworkに対するインデックス
	const unsigned char col13 = threadIdx.x;		// スレッドidx.xが13桁目の文字コード候補

	unsigned char pre31f4(work[gblidx + 0]);
	unsigned char pre31f5(work[gblidx + 1]);
	unsigned char pre31f7(work[gblidx + 2]);
	unsigned char pre31f8(work[gblidx + 3]);
	unsigned char pre31f9(work[gblidx + 4]);
	unsigned char pre31fa(work[gblidx + 5]);
	unsigned char pre31fb(work[gblidx + 6]);
	unsigned char	    c(work[gblidx + 7]);	// キャリーフラグ

	unsigned char xor31f4, xor31f5, wk31fa;

	// 一部オペコードをインライン化
	auto bitrev = [](unsigned int v) {	unsigned int wk = __brev(v); return (wk >> 24); };
	auto adc = [&c](unsigned short vl, unsigned short vr) { unsigned short wk = vl + vr + c; c = (wk >> 8) & 0x01; return (wk & 0xFF); };
	auto ror = [&c](unsigned char v) { unsigned char wc = c; c = v & 0x01; return (unsigned char)((v >> 1) | (wc << 7)); };
	auto bitcnt = [](unsigned long int v) { return (__popc(v)); };

	unsigned char wk14 = pre31f9 ^ col13;	// 13桁目を12桁目までの結果にXOR→ 31f9
	pre31f9 = (pre31f9 & 0x80) | chrmask[wk14 & 0x3F];	// 無効コード判定は引き継ぐ

	unsigned char	col14 = pre31f9 ^ 0x07;		// 下位3bit反転で14桁目

	const unsigned char chr1(col13 & 0x3F), chr2(col14 & 0x3F);


	// col13～col14で与えれるならびで計算をまわす
	xor31f4 = lut_x31F4[pre31f5]; xor31f5 = lut_x31F5[pre31f5];
	pre31f5 = pre31f4 ^ xor31f5;
	pre31f4 = bitrev(chr1) ^ xor31f4;	c = (pre31f4 >= 0xE5) ? 1 : 0;
	pre31f7 = adc(chr1, pre31f7);
	pre31f8 = adc(pre31f8, pre31f5);	wk31fa = ror(pre31fa);
	pre31fa = adc(wk31fa, chr1);
	pre31fb += c + bitcnt(chr1);

	xor31f4 = lut_x31F4[pre31f5]; xor31f5 = lut_x31F5[pre31f5];
	pre31f5 = pre31f4 ^ xor31f5;
	pre31f4 = bitrev(chr2) ^ xor31f4;	c = (pre31f4 >= 0xE5) ? 1 : 0;
	pre31f7 = adc(chr2, pre31f7);
	pre31f8 = adc(pre31f8, pre31f5);	wk31fa = ror(pre31fa);
	pre31fa = adc(wk31fa, chr2);
	pre31fb += c + bitcnt(chr2);

	bool	judge = true
		&& (pre31f4 == 0x65)
		&& (pre31f5 == 0x94)
		&& (pre31f7 == 0xAC)
		&& (pre31f8 == 0xE9)
		&& (pre31f9 == 0x07)
		&& (pre31fa == 0x33)
		&& (pre31fb == 0x25)
		;

	/*$31F4*/	result[((blockIdx.x * blockDim.x) + threadIdx.x * 8) + 0] = pre31f4;
	/*$31F5*/	result[((blockIdx.x * blockDim.x) + threadIdx.x * 8) + 1] = pre31f5;
	/*$31F7*/	result[((blockIdx.x * blockDim.x) + threadIdx.x * 8) + 2] = pre31f7;
	/*$31F8*/	result[((blockIdx.x * blockDim.x) + threadIdx.x * 8) + 3] = pre31f8;
	/*$31F9*/	result[((blockIdx.x * blockDim.x) + threadIdx.x * 8) + 4] = pre31f9;
	/*$31FA*/	result[((blockIdx.x * blockDim.x) + threadIdx.x * 8) + 5] = pre31fa;
	/*$31FB*/	result[((blockIdx.x * blockDim.x) + threadIdx.x * 8) + 6] = pre31fb;
	/*Passed*/	result[((blockIdx.x * blockDim.x) + threadIdx.x * 8) + 7] = judge;

	valid[(blockIdx.x * blockDim.x) + threadIdx.x] = judge;
}

const unsigned char lut_xor31F4[256] = {
	0x00,0x11,0x23,0x32,0x46,0x57,0x65,0x74,0x8C,0x9D,0xAF,0xBE,0xCA,0xDB,0xE9,0xF8,
	0x10,0x01,0x33,0x22,0x56,0x47,0x75,0x64,0x9C,0x8D,0xBF,0xAE,0xDA,0xCB,0xF9,0xE8,
	0x21,0x30,0x02,0x13,0x67,0x76,0x44,0x55,0xAD,0xBC,0x8E,0x9F,0xEB,0xFA,0xC8,0xD9,
	0x31,0x20,0x12,0x03,0x77,0x66,0x54,0x45,0xBD,0xAC,0x9E,0x8F,0xFB,0xEA,0xD8,0xC9,
	0x42,0x53,0x61,0x70,0x04,0x15,0x27,0x36,0xCE,0xDF,0xED,0xFC,0x88,0x99,0xAB,0xBA,
	0x52,0x43,0x71,0x60,0x14,0x05,0x37,0x26,0xDE,0xCF,0xFD,0xEC,0x98,0x89,0xBB,0xAA,
	0x63,0x72,0x40,0x51,0x25,0x34,0x06,0x17,0xEF,0xFE,0xCC,0xDD,0xA9,0xB8,0x8A,0x9B,
	0x73,0x62,0x50,0x41,0x35,0x24,0x16,0x07,0xFF,0xEE,0xDC,0xCD,0xB9,0xA8,0x9A,0x8B,
	0x84,0x95,0xA7,0xB6,0xC2,0xD3,0xE1,0xF0,0x08,0x19,0x2B,0x3A,0x4E,0x5F,0x6D,0x7C,
	0x94,0x85,0xB7,0xA6,0xD2,0xC3,0xF1,0xE0,0x18,0x09,0x3B,0x2A,0x5E,0x4F,0x7D,0x6C,
	0xA5,0xB4,0x86,0x97,0xE3,0xF2,0xC0,0xD1,0x29,0x38,0x0A,0x1B,0x6F,0x7E,0x4C,0x5D,
	0xB5,0xA4,0x96,0x87,0xF3,0xE2,0xD0,0xC1,0x39,0x28,0x1A,0x0B,0x7F,0x6E,0x5C,0x4D,
	0xC6,0xD7,0xE5,0xF4,0x80,0x91,0xA3,0xB2,0x4A,0x5B,0x69,0x78,0x0C,0x1D,0x2F,0x3E,
	0xD6,0xC7,0xF5,0xE4,0x90,0x81,0xB3,0xA2,0x5A,0x4B,0x79,0x68,0x1C,0x0D,0x3F,0x2E,
	0xE7,0xF6,0xC4,0xD5,0xA1,0xB0,0x82,0x93,0x6B,0x7A,0x48,0x59,0x2D,0x3C,0x0E,0x1F,
	0xF7,0xE6,0xD4,0xC5,0xB1,0xA0,0x92,0x83,0x7B,0x6A,0x58,0x49,0x3D,0x2C,0x1E,0x0F,
};

const unsigned char lut_xor31F5[256] = {
	0x00,0x89,0x12,0x9B,0x24,0xAD,0x36,0xBF,0x48,0xC1,0x5A,0xD3,0x6C,0xE5,0x7E,0xF7,
	0x81,0x08,0x93,0x1A,0xA5,0x2C,0xB7,0x3E,0xC9,0x40,0xDB,0x52,0xED,0x64,0xFF,0x76,
	0x02,0x8B,0x10,0x99,0x26,0xAF,0x34,0xBD,0x4A,0xC3,0x58,0xD1,0x6E,0xE7,0x7C,0xF5,
	0x83,0x0A,0x91,0x18,0xA7,0x2E,0xB5,0x3C,0xCB,0x42,0xD9,0x50,0xEF,0x66,0xFD,0x74,
	0x04,0x8D,0x16,0x9F,0x20,0xA9,0x32,0xBB,0x4C,0xC5,0x5E,0xD7,0x68,0xE1,0x7A,0xF3,
	0x85,0x0C,0x97,0x1E,0xA1,0x28,0xB3,0x3A,0xCD,0x44,0xDF,0x56,0xE9,0x60,0xFB,0x72,
	0x06,0x8F,0x14,0x9D,0x22,0xAB,0x30,0xB9,0x4E,0xC7,0x5C,0xD5,0x6A,0xE3,0x78,0xF1,
	0x87,0x0E,0x95,0x1C,0xA3,0x2A,0xB1,0x38,0xCF,0x46,0xDD,0x54,0xEB,0x62,0xF9,0x70,
	0x08,0x81,0x1A,0x93,0x2C,0xA5,0x3E,0xB7,0x40,0xC9,0x52,0xDB,0x64,0xED,0x76,0xFF,
	0x89,0x00,0x9B,0x12,0xAD,0x24,0xBF,0x36,0xC1,0x48,0xD3,0x5A,0xE5,0x6C,0xF7,0x7E,
	0x0A,0x83,0x18,0x91,0x2E,0xA7,0x3C,0xB5,0x42,0xCB,0x50,0xD9,0x66,0xEF,0x74,0xFD,
	0x8B,0x02,0x99,0x10,0xAF,0x26,0xBD,0x34,0xC3,0x4A,0xD1,0x58,0xE7,0x6E,0xF5,0x7C,
	0x0C,0x85,0x1E,0x97,0x28,0xA1,0x3A,0xB3,0x44,0xCD,0x56,0xDF,0x60,0xE9,0x72,0xFB,
	0x8D,0x04,0x9F,0x16,0xA9,0x20,0xBB,0x32,0xC5,0x4C,0xD7,0x5E,0xE1,0x68,0xF3,0x7A,
	0x0E,0x87,0x1C,0x95,0x2A,0xA3,0x38,0xB1,0x46,0xCF,0x54,0xDD,0x62,0xEB,0x70,0xF9,
	0x8F,0x06,0x9D,0x14,0xAB,0x22,0xB9,0x30,0xC7,0x4E,0xD5,0x5C,0xE3,0x6A,0xF1,0x78,
};

using namespace std;

// const char _dict[] = { "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!.-nmcﾅﾑｺ" };
const unsigned char charvalidmask[64] = { 
	0x00,0x00,0x00,0x00,0x00,0x00,0x80,0x80,	// 0x06と0x07が無効
	0x00,0x00,0x00,0x00,0x00,0x00,0x80,0x80,
	0x00,0x00,0x00,0x00,0x00,0x00,0x80,0x80,
	0x00,0x00,0x00,0x00,0x00,0x00,0x80,0x80,
	0x00,0x00,0x00,0x00,0x00,0x00,0x80,0x80,
	0x00,0x00,0x00,0x00,0x00,0x00,0x80,0x80,
	0x00,0x00,0x00,0x00,0x00,0x00,0x80,0x80,
};

cudaError_t chkthread(
	unsigned char *cpu_result,
	unsigned long	item_length
);

const unsigned long	WORKSIZE = 64 * 64 * 64 * 64;
static unsigned char result[WORKSIZE];

int main()
{
	cudaError_t cudaStatus;

	chkthread(result, WORKSIZE);

	// cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceReset failed!"); return 1; }

	return 0;
}

cudaError_t chkthread(
	unsigned char *cpu_result,
	unsigned long	item_length
)
{
	cudaError_t cudaStatus;
	unsigned char *dev_chrcode_mask = 0;	// 有効文字コード判定用マスク		有効は0x00、無効は0x80。
	unsigned char *dev_validpass = 0;		// 計算が合うならば1が入る

	unsigned char *dev_31F4_xortbl = 0;		// $31F4導出用テーブル
	unsigned char *dev_31F5_xortbl = 0;		// $31F5導出用テーブル

	unsigned char *cpu_validpass = 0;		// 有効パスワード判定結果(CPU側)
	unsigned char *cpu_calcresult = 0;		// 計算結果確認用

	unsigned char *dev_res4 = 0;		// 先頭4桁ぶんの計算結果とキャリー情報
	unsigned char *dev_res8 = 0;		// 5-8桁までの計算結果とキャリー情報
	unsigned char *dev_res12 = 0;		// 9-12桁まで計算結果とキャリー情報
	unsigned char *dev_result = 0;		// 14桁の計算結果

	unsigned long long validcnt = 0;	// チェックディジットを通ったパスワードの個数

	dim3	 block_1677m(64), grid_1677m(64 * 64, 64);	// 文字64種 4桁分の並列計算グリッド用	16.8M items

	cudaStatus = cudaSetDevice(0);		if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?"); goto Error; }

	
	// 導出用テーブル(const)
	cudaStatus = cudaMalloc((void**)&dev_chrcode_mask, 64 * sizeof(unsigned char));						if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMalloc((void**)&dev_31F4_xortbl, 256 * sizeof(unsigned char));		if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMalloc((void**)&dev_31F5_xortbl, 256 * sizeof(unsigned char));		if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }

	// 4桁分の各データ算出用領域確保
	cudaStatus = cudaMalloc((void**)&dev_res4, 64 * 64 * 64 * 64 * 8);	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMalloc((void**)&dev_res8, 64 * 64 * 64 * 64 * 8);	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMalloc((void**)&dev_res12, 64 * 64 * 64 * 64 * 8);	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMalloc((void**)&dev_result, 64 * 64 * 64 * 8);	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMalloc((void**)&dev_validpass, 64 * 64 * 64);	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }


	cudaStatus = cudaMemcpy(dev_chrcode_mask, charvalidmask, 64 * sizeof(unsigned char), cudaMemcpyHostToDevice);	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_31F4_xortbl, lut_xor31F4, 256 * sizeof(unsigned char), cudaMemcpyHostToDevice);	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_31F5_xortbl, lut_xor31F5, 256 * sizeof(unsigned char), cudaMemcpyHostToDevice);	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }

	cudaStatus = cudaMallocHost((void**)&cpu_validpass, 64 * 64 * 64 * 8 * sizeof(unsigned char));	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMallocHost((void**)&cpu_calcresult, 64 * 64 * 64 * sizeof(unsigned char));	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }

	// 先頭４桁全組みあわせの計算
	calc_1_4col <<< dim3(64 * 64, 64), dim3(64) >>> (dev_res4, dev_chrcode_mask, dev_31F4_xortbl, dev_31F5_xortbl);	cudaStatus = cudaGetLastError();			if (cudaStatus != cudaSuccess) { fprintf(stderr, "checkPassKernel launch failed: %s\n", cudaGetErrorString(cudaStatus)); goto Error; }
	cudaStatus = cudaDeviceSynchronize();		if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching calu14col!\n", cudaStatus); goto Error; }

	validcnt = 0;

	for (int xor4idx = 0; xor4idx < (64 * 64 * 64 * 64); ++xor4idx) {

		// 5-8桁の組合せを計算する
		calc_4col <<< dim3(64 * 64, 64), dim3(64) >>> (dev_res8, dev_res4, dev_chrcode_mask, dev_31F4_xortbl, dev_31F5_xortbl, xor4idx);
		cudaStatus = cudaDeviceSynchronize();		if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); goto Error; }

		for (int xor8idx = 0; xor8idx < (64 * 64 * 64 * 64); ++xor8idx) {
			// 9～12桁分の組合せを計算する
			calc_4col <<< dim3(64 * 64, 64), 64 >>> (dev_res12, dev_res8, dev_chrcode_mask, dev_31F4_xortbl, dev_31F5_xortbl, xor8idx);
			cudaStatus = cudaDeviceSynchronize();		if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); goto Error; }


			for (int xor12idx = 0; xor12idx < (64 * 64); ++xor12idx) {
				// 13,14桁目の導出とチェックディジットが通っているかの判定
				calclast_validate << < 64 * 64, 64 >> > (dev_result, dev_validpass, dev_res12, dev_31F4_xortbl, dev_31F5_xortbl, dev_chrcode_mask, xor12idx);
				cudaStatus = cudaDeviceSynchronize();		if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching validation!\n", cudaStatus); goto Error; }


				cudaStatus = cudaMemcpy(cpu_calcresult, dev_result, size_t(64 * 64 * 64 * 8) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
				cudaStatus = cudaMemcpy(cpu_validpass, dev_validpass, 64 * 64 * 64 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
				for (int chkrs = 0; chkrs < (64 * 64 * 64); ++chkrs) {
					printf("\n");

					for (int vidx = 0; vidx < 8; ++vidx) {
						printf("%02X ", cpu_calcresult[chkrs * 8 + vidx]);
					}
					validcnt += cpu_validpass[chkrs];
				}
//				printf("\n%d items\n", validcnt);


			}
		}
	}
	printf("found \n%lld items\n", validcnt);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();			if (cudaStatus != cudaSuccess) { fprintf(stderr, "checkPassKernel launch failed: %s\n", cudaGetErrorString(cudaStatus)); goto Error; }

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();		if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); goto Error; }

	// Copy output vector from GPU buffer to host memory.
//	cudaStatus = cudaMemcpy(cpu_result, dev_xor_result, 64 * 64 * 64 * 64 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
//	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }


Error:
	cudaFree(dev_chrcode_mask);
	cudaFree(dev_validpass);

	cudaFree(dev_31F4_xortbl);
	cudaFree(dev_31F5_xortbl);

	cudaFree(cpu_validpass);
	cudaFree(cpu_calcresult);

	cudaFree(dev_res4);
	cudaFree(dev_res8);
	cudaFree(dev_res12);
	cudaFree(dev_result);

	return(cudaStatus);

}
