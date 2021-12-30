#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>
#include <stdio.h>

#define PASSWORD_LEN		(14)
#define PASSWORD_LEN_MAX	(16)

// パスワードの計算カーネル
__global__ void checkPassKernel(
	unsigned char *result,			// OUTPUT	パスしたらTRUEを返す先	[x*y]
	const unsigned char*candidate,	// INPUT	パスワード候補(文字コード変換済み) [x*y][16]
	const unsigned char*refer_chk	// INPUT	判定用チェックディジット	[x*y][16]
)
{
	bool result_b(false), match_flg;
	int a(0), c(0), cb(0), chridx, strlength;
	unsigned char mem20(0), mem21(0);
	unsigned char mem50(0), mem51(0), mem52(0), mem53(1), mem54(0);

	int th_x = threadIdx.x;		// 
//	int th_y = threadIdx.y;

	auto ror = [&c](unsigned char v)
	{
		unsigned char wc = c;

		c = v & 0x01;
		return (unsigned char)((v >> 1) | (wc << 7));
	};
	auto adc = [&c](unsigned char lhs, unsigned char rhs)
	{
		unsigned short v;

		v = (unsigned short)lhs + (unsigned short)rhs + (unsigned short)c;
		c = (v > 0xFF) ? 1 : 0;

		return (v & 0xFF);
	};
	auto bitcnt = [](unsigned long int v)
	{
			v = (v & 0x55555555) + (v >> 1 & 0x55555555);
			v = (v & 0x33333333) + (v >> 2 & 0x33333333);
			v = (v & 0x0f0f0f0f) + (v >> 4 & 0x0f0f0f0f);
			v = (v & 0x00ff00ff) + (v >> 8 & 0x00ff00ff);
		return  (v & 0x0000ffff) + (v >> 16 & 0x0000ffff);
	};

	strlength = refer_chk[2];
	for (chridx = 0; chridx < strlength; ++chridx) {
		a = candidate[(th_x * PASSWORD_LEN_MAX) + chridx];

		// calc checkdigit1	(ループ展開しておく)
//		for (int bit = 7; bit >= 0; --bit) {
//			c = (a >> bit) & 1;
//			mem20 = ror(mem20);
//			mem21 = ror(mem21);
//
//			if (c) {
//				mem20 ^= 0x84;
//				mem21 ^= 0x08;
//			}
//		}
		c = ((a & 0x80) != 0);
		mem20 = ror(mem20);
		mem21 = ror(mem21);
		cb = c ? 0x84 : 0x00;
		mem20 ^= cb;
		cb = c ? 0x08 : 0x00;
		mem21 ^= cb;

		c = ((a & 0x40) != 0);
		mem20 = ror(mem20);
		mem21 = ror(mem21);
		cb = c ? 0x84 : 0x00;
		mem20 ^= cb;
		cb = c ? 0x08 : 0x00;
		mem21 ^= cb;

		c = ((a & 0x20) != 0);
		mem20 = ror(mem20);
		mem21 = ror(mem21);
		cb = c ? 0x84 : 0x00;
		mem20 ^= cb;
		cb = c ? 0x08 : 0x00;
		mem21 ^= cb;

		c = ((a & 0x10) != 0);
		mem20 = ror(mem20);
		mem21 = ror(mem21);
		cb = c ? 0x84 : 0x00;
		mem20 ^= cb;
		cb = c ? 0x08 : 0x00;
		mem21 ^= cb;

		c = ((a & 0x08) != 0);
		mem20 = ror(mem20);
		mem21 = ror(mem21);
		cb = c ? 0x84 : 0x00;
		mem20 ^= cb;
		cb = c ? 0x08 : 0x00;
		mem21 ^= cb;

		c = ((a & 0x04) != 0);
		mem20 = ror(mem20);
		mem21 = ror(mem21);
		cb = c ? 0x84 : 0x00;
		mem20 ^= cb;
		cb = c ? 0x08 : 0x00;
		mem21 ^= cb;

		c = ((a & 0x02) != 0);
		mem20 = ror(mem20);
		mem21 = ror(mem21);
		cb = c ? 0x84 : 0x00;
		mem20 ^= cb;
		cb = c ? 0x08 : 0x00;
		mem21 ^= cb;

		c = ((a & 0x01) != 0);
		mem20 = ror(mem20);
		mem21 = ror(mem21);
		cb = c ? 0x84 : 0x00;
		mem20 ^= cb;
		cb = c ? 0x08 : 0x00;
		mem21 ^= cb;

		// calc checkdigit2
		c = (mem20 >= 0xE5) ? 1 : 0;
		mem50 = adc(a, mem50);
		mem51 = adc(mem51, mem21);
		// calc checkdigit3
		mem52 ^= a;
		// calc checkdigit4
		{
			unsigned char v = ror(mem53);
			mem53 = adc(v, a);
		}
		// calc checkdigit5
		mem54 += (unsigned char)(c + bitcnt(a));
	}
	
	result_b = true;
	match_flg = (mem20 == refer_chk[0]);
	result_b = result_b && match_flg;
	match_flg = (mem21 == refer_chk[1]);
	result_b = result_b && match_flg;
	match_flg = (mem50 == refer_chk[3]);
	result_b = result_b && match_flg;
	match_flg = (mem51 == refer_chk[4]);
	result_b = result_b && match_flg;
	match_flg = (mem52 == refer_chk[5]);
	result_b = result_b && match_flg;
	match_flg = (mem53 == refer_chk[6]);
	result_b = result_b && match_flg;
	match_flg = (mem54 == refer_chk[7]);
	result_b = result_b && match_flg;

	result[th_x] = result_b;
}


// 2桁分のパスワード候補を生成する
__global__ void chrconvKernel(
	unsigned char *candidate,			// OUTPUT	生成する候補データ
	const unsigned char *basecode,		// INPUT	固定部分として参照する桁情報[16]
	const unsigned char *tbl_chrdict,	// INPUT	文字コードテーブル[256]
	const unsigned int	passlen,		// INPUT	変換対象の桁数(全体)
	const unsigned int	dictlen			// INPUT	文字種数
)
{
	int th_x = threadIdx.x;

	for (int i = 0; i < (passlen - 1); ++i) {
		candidate[(PASSWORD_LEN_MAX * th_x) + i] = tbl_chrdict[basecode[i]];
	}
}

__global__ void make_4col_fullpair(
	unsigned char *pass4col,		// 格納先 自分の要素は求めたグローバルインデックス*4からはじまる4byte
	unsigned char *chrmask,			// 有効文字コードかの判定用ビットマスク
	const int  offset				// blockIdxのzに履かせる下駄
)
{
	unsigned int	blocksize = blockDim.x * blockDim.y * blockDim.z;
	unsigned int	thidx = (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
	unsigned int	gridx = ((gridDim.x * gridDim.y) * ((blockIdx.z + (offset*gridDim.z)))) + (gridDim.x * blockIdx.y) + blockIdx.x;
	unsigned int	gblidx = gridx * blocksize + thidx;

	unsigned char	col1 = blockIdx.z + (offset * gridDim.z);
	unsigned char	col2 = blockIdx.y;
	unsigned char	col3 = blockIdx.x * (blockDim.z * blockDim.y) + threadIdx.z * blockDim.y + threadIdx.y;
	unsigned char	col4 = threadIdx.x;
	
	unsigned char unvalid;

	unvalid = chrmask[col1] | chrmask[col2] | chrmask[col3] | chrmask[col4];

	pass4col[gblidx] = unvalid | (col1 ^ col2 ^ col3 ^ col4);
}

// これまでのXOR結果と4桁ぶんのXORを求める
__global__ void calcxor8(
	unsigned char		*xor8,		// 演算結果
	const unsigned char *xor4,		// これまでの演算結果
	const unsigned int	idxG		// xor8のインデックス
)
{
	unsigned int idxH = (blockIdx.x * 256) + threadIdx.x;

	unsigned char W1 = xor4[idxG];
	unsigned char W2 = xor4[idxH];
	unsigned char enf = (W1 | W2) & 0x80;

	xor8[idxH] = enf | (W1 ^ W2);
}

// 13桁目の導出とxor 0x07結果の妥当性判断
__global__ void calcxor13_validchk(
	unsigned char		*valid14,		// 14桁目が有効なら1を格納
	const unsigned char *xor12,			// 12桁目まで導出されたXOR
	const unsigned char *chrmask		// 有効な文字コードかどうか
)
{
	unsigned char col13 = xor12[blockIdx.x] ^ threadIdx.x;	// 13桁目の導出 1677万要素の当該桁値

	col13 |= chrmask[threadIdx.x];
	col13 ^= 0x07;

	valid14[(blockIdx.x*blockDim.x) + threadIdx.x] = !(col13 & 0x80);	// 無効な文字コードか判定をかける
}

// カーネル呼び出しテスト用
__global__ void kerntest()
{
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

// 文字のビットを上下反転するテーブル
unsigned char lut_reverse_char_bit[] = {
  0b00000000, 0b10000000, 0b01000000, 0b11000000, 0b00100000, 0b10100000, 0b01100000, 0b11100000,	// 00 - 07
  0b00010000, 0b10010000, 0b01010000, 0b11010000, 0b00110000, 0b10110000, 0b01110000, 0b11110000,	// 08 - 0F
  0b00001000, 0b10001000, 0b01001000, 0b11001000, 0b00101000, 0b10101000, 0b01101000, 0b11101000,	// 10 - 17
  0b00011000, 0b10011000, 0b01011000, 0b11011000, 0b00111000, 0b10111000, 0b01111000, 0b11111000,	// 18 - 1F
  0b00000100, 0b10000100, 0b01000100, 0b11000100, 0b00100100, 0b10100100, 0b01100100, 0b11100100,	// 20 - 27
  0b00010100, 0b10010100, 0b01010100, 0b11010100, 0b00110100, 0b10110100, 0b01110100, 0b11110100,	// 28 - 2F
  0b00001100, 0b10001100, 0b01001100, 0b11001100, 0b00101100, 0b10101100, 0b01101100, 0b11101100,	// 30 - 37
  0b00011100, 0b10011100, 0b01011100, 0b11011100, 0b00111100, 0b10111100, 0b01111100, 0b11111100,	// 38 - 3F
};

using namespace std;

//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
cudaError_t chkPass(unsigned char *, string, int, unsigned char *);

const unsigned char g_chrcode[256] =
{
	0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
	0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
	0x00,0x2D,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x2B,0x00,0x00,
	0x25,0x04,0x0C,0x14,0x1C,0x24,0x05,0x0D,0x15,0x1D,0x00,0x00,0x00,0x00,0x00,0x00,
	0x00,0x00,0x08,0x10,0x18,0x20,0x28,0x30,0x01,0x09,0x11,0x19,0x21,0x29,0x31,0x02,
	0x0A,0x12,0x1A,0x22,0x2A,0x32,0x03,0x0B,0x13,0x1B,0x23,0x00,0x00,0x00,0x00,0x00,
	0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
	0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
	0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
	0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
	0x00,0x00,0x00,0x00,0x00,0x33,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
	0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x35,0x00,0x00,0x00,0x00,0x00,
	0x00,0x00,0x00,0x00,0x00,0x2C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
	0x00,0x34,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
	0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
	0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
};

//const char _dict[] = { "ABCDEFGHIJKLMNOPQRSTUVWXYZ" };
const char _dict[] = { "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!.-nmcﾅﾑｺ" };

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
	chkthread(result, WORKSIZE);

#if 0
	unsigned char	pass_columns[PASSWORD_LEN_MAX] = { 0 };
	int passed = -1;

	cudaError_t cudaStatus;

	const size_t	dictlen = sizeof(_dict);
	std::string chrtable(_dict, dictlen);
	unsigned char *chkresult = new unsigned char[dictlen * dictlen];

	const int		COLUMNS = 14;
	const int		refresh_col_n = 4;
	const int		chkcol = 8;
	unsigned char	chkkey;
	unsigned char	refresh_col;		// 進捗表示用の判定桁

	// 終了判定用桁値を退避
	chkkey = pass_columns[chkcol];
	refresh_col = pass_columns[refresh_col_n];

	while (pass_columns[chkcol] == chkkey) {

		if (refresh_col != pass_columns[refresh_col_n]) {

			for (int idx = 0; idx < PASSWORD_LEN; ++idx) {
				fprintf(stderr, "%c", _dict[pass_columns[idx]]);
			}
			fprintf(stderr, "\n");
			refresh_col = pass_columns[refresh_col_n];
		}

		cudaError_t cudaStatus = chkPass(chkresult, chrtable, 14, pass_columns);
		if (cudaStatus != cudaSuccess) { fprintf(stderr, "passcheck failed!"); return 1; }

		for (int idx = 0; idx < (chrtable.size() * chrtable.size()); ++idx) {
			if (chkresult[idx] != 0) {
				fprintf(stderr, "Passed %d\n", idx);
			}
		}

		// 終了条件に合致する状態を判定したら探索終了
		bool carry = true;
		for (int col = (2); col < (COLUMNS - 1); ++col) {
			//	for (int col = 2; col < (COLUMNS - 1); ++col) {
			pass_columns[col] += carry ? 1 : 0;
			carry = (pass_columns[col] == dictlen);
			pass_columns[col] = carry ? 0 : pass_columns[col];
			//	candidate[col] = dict[counters[col]];
		}
	}

	
	// cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceReset failed!"); return 1; }

	delete[] chkresult;
#endif
	return 0;
}

cudaError_t chkthread(
	unsigned char *cpu_result,
	unsigned long	item_length
)
{
	cudaError_t cudaStatus;
//	unsigned char *dev_result = 0;		// 判定結果
	unsigned char *dev_xor_result = 0;		// 4桁ぶんのxor結果
	unsigned char *dev_xor_result8 = 0;		// 8桁ぶんのxor結果(テンポラリ)
	unsigned char *dev_xor_result12 = 0;	// 12桁ぶんのxor結果(テンポラリ)
	unsigned char *dev_31F9_is07h = 0;		// 13桁目までxorを求めた結果に0x07でxorした値(14桁目の文字コード)が有効な文字コードか

	unsigned char *dev_chrcode_mask = 0;	// 有効文字コード判定用マスク		有効は0x00、無効は0x80。

	unsigned char *cpu_31F9_valid = 0;
	unsigned long validcnt = 0;


//	dim3	 block_530m(48, 8, 2), grid_530m(3, 48, 6);	// 文字48種 4桁分の並列計算グリッド用	 5.3M items
	dim3	 block_1677m(64, 8, 2), grid_1677m(4, 64, 4);	// 文字64種 4桁分の並列計算グリッド用	16.8M items

	cudaStatus = cudaSetDevice(0);																		if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?"); goto Error; }

	// 4桁分総組合せのXORデータ生成
	cudaStatus = cudaMalloc((void**)&dev_xor_result, 64 * 64 * 64 * 64 * sizeof(unsigned char));		if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMalloc((void**)&dev_xor_result8, 64 * 64 * 64 * 64 * sizeof(unsigned char));		if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMalloc((void**)&dev_xor_result12, 64 * 64 * 64 * 64 * sizeof(unsigned char));		if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMalloc((void**)&dev_31F9_is07h, 64 * 64 * 64 * 64 * 64 * sizeof(unsigned char));		if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }

	// 64文字種分のコード有効無効情報
	cudaStatus = cudaMalloc((void**)&dev_chrcode_mask, 64 * sizeof(unsigned char));						if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMemcpy(dev_chrcode_mask, charvalidmask, 64 * sizeof(unsigned char), cudaMemcpyHostToDevice);	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }

	cudaStatus = cudaMallocHost((void**)&cpu_31F9_valid, 64 * 64 * 64 * 64 * 64 * sizeof(unsigned char));	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }

	// 4桁分総組合せのXORデータ計算
	for (int offset = 0; offset < 16; ++offset) {
		make_4col_fullpair <<<grid_1677m, block_1677m >>> (dev_xor_result, dev_chrcode_mask, offset);
	}
	cudaStatus = cudaGetLastError();			if (cudaStatus != cudaSuccess) { fprintf(stderr, "checkPassKernel launch failed: %s\n", cudaGetErrorString(cudaStatus)); goto Error; }
	cudaStatus = cudaDeviceSynchronize();		if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); goto Error; }

	validcnt = 0;

	// 8桁ぶんの組合せを計算する
	for (int xor4idx = 0; xor4idx < (256 * 256 * 256); ++xor4idx) {

		// xor4idx番目の組合せについて次の4桁を計算する
		calcxor8 << < 256 * 256, 256 >> > (dev_xor_result8, dev_xor_result, xor4idx);
		cudaStatus = cudaDeviceSynchronize();		if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); goto Error; }

		// 12桁分の組合せを計算する
		for (int xor8idx = 0; xor8idx < (256 * 256 * 256); ++xor8idx) {
			// xor8idx番目の組合せについて次の4桁を計算する
			calcxor8 << < 256 * 256, 256 >> > (dev_xor_result12, dev_xor_result8, xor8idx);
			cudaStatus = cudaDeviceSynchronize();		if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); goto Error; }

			// 13桁目の導出と14桁目の有効な文字コードか判定
			calcxor13_validchk << < 256 * 256 * 256, 64 >> > (dev_31F9_is07h, dev_xor_result12, dev_chrcode_mask);
			cudaStatus = cudaDeviceSynchronize();		if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); goto Error; }


			cudaStatus = cudaMemcpy(cpu_31F9_valid, dev_31F9_is07h, 64 * 64 * 64 * 64 * 64 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
			for (int chkrs = 0; chkrs < (256 * 256 * 256 * 64); ++chkrs) {
				validcnt += cpu_31F9_valid[chkrs];
			}

			printf("%d items\n", validcnt);
			validcnt = 0;
		}
	}
//	kerntest <<< dim3(256*256), dim3(256) >>> ();

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();			if (cudaStatus != cudaSuccess) { fprintf(stderr, "checkPassKernel launch failed: %s\n", cudaGetErrorString(cudaStatus)); goto Error; }

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();		if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); goto Error; }

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(cpu_result, dev_xor_result, 64 * 64 * 64 * 64 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }


Error:
	cudaFree(cpu_31F9_valid);
	cudaFree(dev_31F9_is07h);
	cudaFree(dev_xor_result12);
	cudaFree(dev_xor_result8);
	cudaFree(dev_xor_result);
	cudaFree(dev_chrcode_mask);

	return(cudaStatus);

}
#if 0
// デバイスとのやり取り
cudaError_t chkPass(
	unsigned char *result_holder,		// OUTPUT 計算結果の受取先
	string	chrdic,						// 文字種のテーブル
	int		passlength,					// パスワード長さ
	unsigned char *passcolmns			// パスワードの総当たり用カウンタ[PASSWORD_LEN_MAX = 16]
)
{
	cudaError_t cudaStatus;

	unsigned char *dev_result = 0;		// 判定結果
	unsigned char *dev_candidate = 0;	// 照合対象のキーと長さ情報の配列
	unsigned char *dev_reference = 0;	// 判定対象のチェックディジット

	unsigned char *dev_passnum = 0;		// パスワード生成用カウンタ
	unsigned char *dev_lut_code = 0;	// 文字変換テーブル

	int dlen = chrdic.size();

	unsigned char reference[16] = {
		0x65, 0x94, 0x0E, 0xAC, 0xE9, 0x07, 0x33, 0x25,	// ←探すべきチェックディジット群
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
	};


	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?"); goto Error; }

	// 照合結果は辞書の文字種分
	cudaStatus = cudaMalloc((void**)&dev_result, chrdic.size() * chrdic.size() * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }

	// パスワード候補数は文字種分
	cudaStatus = cudaMalloc((void**)&dev_candidate, chrdic.size() * chrdic.size() * sizeof(unsigned char) * PASSWORD_LEN_MAX);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }

	// 文字変換テーブル
	cudaStatus = cudaMalloc((void**)&dev_lut_code, 256);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }

	// 文字テーブル
	cudaStatus = cudaMalloc((void**)&dev_passnum, PASSWORD_LEN_MAX);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }

	// 照合対象のチェックディジットは一種類
	cudaStatus = cudaMalloc((void**)&dev_reference, 16);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }

	// Copy input vectors from host memory to GPU buffers.
	// 文字コード変換テーブル
	cudaStatus = cudaMemcpy(dev_lut_code, g_chrcode, 256, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }

	// カウンタから文字コード生成
	chrconvKernel << <chrdic.size(), chrdic.size() >> >	(dev_candidate, dev_passnum, dev_lut_code, passlength, chrdic.size());
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "chrconvKernel launch failed: %s\n", cudaGetErrorString(cudaStatus)); goto Error; }

	cudaStatus = cudaMemcpy(dev_reference, reference, 16, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }

	cudaStatus = cudaDeviceSynchronize();

	// Launch a kernel on the GPU with one thread for each element.
	checkPassKernel << <chrdic.size(), chrdic.size() >> > (dev_result, dev_candidate, dev_reference);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "checkPassKernel launch failed: %s\n", cudaGetErrorString(cudaStatus)); goto Error; }

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); goto Error; }

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(result_holder, dev_result, chrdic.size() * chrdic.size(), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }

Error:
	cudaFree(dev_result);
	cudaFree(dev_candidate);
	cudaFree(dev_reference);
	cudaFree(dev_passnum);
	cudaFree(dev_lut_code);

	return(cudaStatus);
}

#endif