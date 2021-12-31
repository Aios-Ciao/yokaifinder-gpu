#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>
#include <stdio.h>

__device__ inline unsigned char bitrev(unsigned int v)
{
	unsigned int wk = __brev(v);
	return (wk >> 24) & 0xFF;
}

__device__ inline unsigned char adc(unsigned char &c, unsigned char vl, unsigned char vr)
{
	unsigned short wk = vl + vr + c;
	c = (wk & 0x0100) ? 1 : 0;
	return (unsigned char)(wk & 0xFF);
}

__device__ inline unsigned char ror(unsigned char &c, unsigned char v)
{
	unsigned char wc = c * 0x80;
	c = v & 0x01;
	return (unsigned char)((v >> 1) | wc);
}

__device__ inline unsigned char bitcnt(unsigned char v)
{
	return ((unsigned char)__popc(v));
}

__device__ unsigned char calcstep(const unsigned char chr, unsigned char *_31F4x)
{
	unsigned char	bchr, c;
	const unsigned char bmask_31f4[2] = { 0x00, 0x84 };
	const unsigned char bmask_31f5[2] = { 0x00, 0x08 };
	unsigned char wk31f4(_31F4x[0]), wk31f5(_31F4x[1]), wk31fa;

	bchr = bitrev(chr);
	c = bchr & 0x01; bchr >>= 1; wk31f4 = ror(c, wk31f4); wk31f5 = ror(c, wk31f5); wk31f4 ^= bmask_31f4[c]; wk31f5 ^= bmask_31f5[c];
	c = bchr & 0x01; bchr >>= 1; wk31f4 = ror(c, wk31f4); wk31f5 = ror(c, wk31f5); wk31f4 ^= bmask_31f4[c]; wk31f5 ^= bmask_31f5[c];
	c = bchr & 0x01; bchr >>= 1; wk31f4 = ror(c, wk31f4); wk31f5 = ror(c, wk31f5); wk31f4 ^= bmask_31f4[c]; wk31f5 ^= bmask_31f5[c];
	c = bchr & 0x01; bchr >>= 1; wk31f4 = ror(c, wk31f4); wk31f5 = ror(c, wk31f5); wk31f4 ^= bmask_31f4[c]; wk31f5 ^= bmask_31f5[c];
	c = bchr & 0x01; bchr >>= 1; wk31f4 = ror(c, wk31f4); wk31f5 = ror(c, wk31f5); wk31f4 ^= bmask_31f4[c]; wk31f5 ^= bmask_31f5[c];
	c = bchr & 0x01; bchr >>= 1; wk31f4 = ror(c, wk31f4); wk31f5 = ror(c, wk31f5); wk31f4 ^= bmask_31f4[c]; wk31f5 ^= bmask_31f5[c];
	c = bchr & 0x01; bchr >>= 1; wk31f4 = ror(c, wk31f4); wk31f5 = ror(c, wk31f5); wk31f4 ^= bmask_31f4[c]; wk31f5 ^= bmask_31f5[c];
	c = bchr & 0x01; bchr >>= 1; wk31f4 = ror(c, wk31f4); wk31f5 = ror(c, wk31f5); wk31f4 ^= bmask_31f4[c]; wk31f5 ^= bmask_31f5[c];
	_31F4x[0] = wk31f4;							// 4
	_31F4x[1] = wk31f5;							// 5
//	_31F4x[2] = 0x0E;							// 6 未操作

	c = (wk31f4 >= 0xE5) ? 1 : 0;
	_31F4x[3] = adc(c, chr, _31F4x[3]);			// 7

	_31F4x[4] = adc(c, _31F4x[4], _31F4x[1]);	// 8

	_31F4x[5] = chr ^ _31F4x[5];				// 9

	wk31fa = ror(c, _31F4x[6]);
	_31F4x[6] = adc(c, wk31fa, chr);			// A

	_31F4x[7] += c + bitcnt(chr);				// B

	return (c);
}

// <<< dim3(64 * 64, 64), dim3(64) >>>
__global__ void calc_1_4col(
	unsigned char		*work4col,		// 格納先 自分の要素は求めたグローバルインデックス*8からはじまる8byte
	unsigned char		*valid,			// 有効な文字コードで構成されたか
	const unsigned char *chrmask		// 有効文字コードかの判定用ビットマスク
)
{
	const unsigned int idx = ((blockIdx.y * gridDim.x) + blockIdx.x) * blockDim.x + threadIdx.x;
	const unsigned int gblidx = idx * 8;

	const unsigned char	col1 = blockIdx.y;
	const unsigned char	col2 = blockIdx.x >> 6;
	const unsigned char	col3 = blockIdx.x & 0x3F;
	const unsigned char	col4 = threadIdx.x;

	const unsigned char chr1(col1 & 0x3F), chr2(col2 & 0x3F), chr3(col3 & 0x3F), chr4(col4 & 0x3F);

	/*$31F4*/	work4col[gblidx + 0] = 0;
	/*$31F5*/	work4col[gblidx + 1] = 0;
	/*$31F6*/	work4col[gblidx + 2] = 0x0E;	// 固定値、不使用
	/*$31F7*/	work4col[gblidx + 3] = 0;
	/*$31F8*/	work4col[gblidx + 4] = 0;
	/*$31F9*/	work4col[gblidx + 5] = 0;
	/*$31FA*/	work4col[gblidx + 6] = 1;
	/*$31FB*/	work4col[gblidx + 7] = 0;

	calcstep(chr1, &work4col[gblidx]);
	calcstep(chr2, &work4col[gblidx]);
	calcstep(chr3, &work4col[gblidx]);
	calcstep(chr4, &work4col[gblidx]);

	valid[idx] = chrmask[chr1] | chrmask[chr2] | chrmask[chr3] | chrmask[chr4];
}

// これまでの桁の値を使って追加で4桁分の計算値を求める
__global__ void calc_4col(
	unsigned char		*work8col,		// 格納先 自分の要素は求めたグローバルインデックス*8からはじまる8byte
	unsigned char		*unvalid8,		// 無効文字が含まれていたらtrue
	const unsigned char	*work4col,		// これまでに計算してきた演算結果
	const unsigned char *chrmask,		// 有効文字コードかの判定用ビットマスク
	const unsigned int	idx4			// work4colに対応するインデックス値
)
{
	// スレッド番号から書き込み先インデックスを計算
	const unsigned int idx = ((blockIdx.y * gridDim.x) + blockIdx.x) * blockDim.x + threadIdx.x;
	const unsigned int gblidx = idx * 8;

	const unsigned char	col1 = blockIdx.y;
	const unsigned char	col2 = blockIdx.x >> 6;
	const unsigned char	col3 = blockIdx.x & 0x3F;
	const unsigned char	col4 = threadIdx.x;
	const unsigned int	rdidx = idx4 * 8;

	/*$31F4*/	work8col[gblidx + 0] = work4col[rdidx + 0];
	/*$31F5*/	work8col[gblidx + 1] = work4col[rdidx + 1];
	/*$31F6*/	work8col[gblidx + 2] = work4col[rdidx + 2];
	/*$31F7*/	work8col[gblidx + 3] = work4col[rdidx + 3];
	/*$31F8*/	work8col[gblidx + 4] = work4col[rdidx + 4];
	/*$31F9*/	work8col[gblidx + 5] = work4col[rdidx + 5];
	/*$31FA*/	work8col[gblidx + 6] = work4col[rdidx + 6];
	/*$31FB*/	work8col[gblidx + 7] = work4col[rdidx + 7];

	const unsigned char chr1(col1 & 0x3F), chr2(col2 & 0x3F), chr3(col3 & 0x3F), chr4(col4 & 0x3F);

	calcstep(chr1, &work8col[gblidx]);
	calcstep(chr2, &work8col[gblidx]);
	calcstep(chr3, &work8col[gblidx]);
	calcstep(chr4, &work8col[gblidx]);

	unvalid8[idx] = chrmask[chr1] | chrmask[chr2] | chrmask[chr3] | chrmask[chr4];
}

// 残りの１桁導出と最終桁までのチェックディジット計算、有効判定
__global__ void calclast_validate(
	unsigned char		*result,	// OUT  14桁分の各データ計算結果
	unsigned char		*valid,		// OUT  パスワードとして成り立つなら1(true)でなければ0(false)
	const unsigned char	*work,		// IN	12桁目までで導出された途中値
	const unsigned int	workidx		// IN	blockIdx.xのオフセット
)
{
	const unsigned int	wkidx = workidx * 8;
	const unsigned int	gblidx = ((blockIdx.x * blockDim.x) + threadIdx.x) * 8;
	const unsigned char col13 = blockIdx.x;			// ブロックidx.xが13桁目の文字コード候補
	const unsigned char col14 = threadIdx.x;		// スレッドidx.xが14桁目の文字コード候補

	/*$31F4*/	result[gblidx + 0] = work[wkidx + 0];
	/*$31F5*/	result[gblidx + 1] = work[wkidx + 1];
	/*$31F6*/	result[gblidx + 2] = work[wkidx + 2];
	/*$31F7*/	result[gblidx + 3] = work[wkidx + 3];
	/*$31F8*/	result[gblidx + 4] = work[wkidx + 4];
	/*$31F9*/	result[gblidx + 5] = work[wkidx + 5];
	/*$31FA*/	result[gblidx + 6] = work[wkidx + 6];
	/*$31FB*/	result[gblidx + 7] = work[wkidx + 7];

	const unsigned char chr13(col13 & 0x3F), chr14(col14 & 0x3F);

	calcstep(chr13, &result[gblidx]);
	calcstep(chr14, &result[gblidx]);

	bool	judge = true
		&& (result[gblidx + 0] == 0x65)	// $31F4
		&& (result[gblidx + 1] == 0x94)	// $31F5
		&& (result[gblidx + 2] == 0x0E)	// $31F6
		&& (result[gblidx + 3] == 0xAC)	// $31F7
		&& (result[gblidx + 4] == 0xE9)	// $31F8
		&& (result[gblidx + 5] == 0x07)	// $31F9
		&& (result[gblidx + 6] == 0x33)	// $31FA
		&& (result[gblidx + 7] == 0x25)	// $31FB
		;

	valid[(blockIdx.x * blockDim.x) + threadIdx.x] = judge;
}

using namespace std;

// const char _dict[] = { "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!.-nmcﾅﾑｺ" };
const unsigned char charvalidmask[64] = {
	0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x01,	// 0x06と0x07が無効
	0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x01,
	0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x01,
	0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x01,
	0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x01,
	0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x01,
	0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x01,
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

	unsigned char *cpu_validpass = 0;		// 有効パスワード判定結果(CPU側)
	unsigned char *cpu_calcresult = 0;		// 計算結果確認用

	unsigned char *cpu_unvalid4 = 0;		// 有効文字で構成されているか情報(先頭4桁)
	unsigned char *cpu_unvalid8 = 0;		// 有効文字で構成されているか情報(5-8桁)
	unsigned char *cpu_unvalid12 = 0;		// 有効文字で構成されているか情報(9-12桁)

	unsigned char *dev_unvalid4 = 0;		// 有効文字で構成されているか情報(先頭4桁)
	unsigned char *dev_unvalid8 = 0;		// 有効文字で構成されているか情報(5-8桁)
	unsigned char *dev_unvalid12 = 0;		// 有効文字で構成されているか情報(9-12桁)

	unsigned char *dev_res4 = 0;			// 先頭4桁ぶんの計算結果とキャリー情報
	unsigned char *dev_res8 = 0;			// 5-8桁までの計算結果とキャリー情報
	unsigned char *dev_res12 = 0;			// 9-12桁まで計算結果とキャリー情報
	unsigned char *dev_result = 0;			// 14桁の計算結果

	unsigned long long validcnt = 0;		// チェックディジットを通ったパスワードの個数

	static const char cvalid[2] = { '.', 'O' };	// 判定表示用


	cudaStatus = cudaSetDevice(0);		if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?"); goto Error; }
	
	// 導出用テーブル(const)
	cudaStatus = cudaMalloc((void**)&dev_chrcode_mask, 64 * sizeof(unsigned char));						if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }

	// 4桁分の各データ算出用領域確保
	cudaStatus = cudaMalloc((void**)&dev_res4, 64 * 64 * 64 * 64 * 8);	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMalloc((void**)&dev_res8, 64 * 64 * 64 * 64 * 8);	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMalloc((void**)&dev_res12, 64 * 64 * 64 * 64 * 8);	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMalloc((void**)&dev_result, 64 * 64 * 64 * 8);	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMalloc((void**)&dev_validpass, 64 * 64 * 64);	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }

	cudaStatus = cudaMalloc((void**)&dev_unvalid4, 64 * 64 * 64 * 64);	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMalloc((void**)&dev_unvalid8, 64 * 64 * 64 * 64);	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMalloc((void**)&dev_unvalid12, 64 * 64 * 64 * 64);	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }


	cudaStatus = cudaMemcpy(dev_chrcode_mask, charvalidmask, 64 * sizeof(unsigned char), cudaMemcpyHostToDevice);	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }

	cudaStatus = cudaMallocHost((void**)&cpu_validpass, 64 * 64 * 64 * 8 * sizeof(unsigned char));	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMallocHost((void**)&cpu_calcresult, 64 * 64 * 64 * 64 * 8 * sizeof(unsigned char));	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }

	cudaStatus = cudaMallocHost((void**)&cpu_unvalid4, 64 * 64 * 64 * 64 * sizeof(unsigned char));	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMallocHost((void**)&cpu_unvalid8, 64 * 64 * 64 * 64 * sizeof(unsigned char));	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
	cudaStatus = cudaMallocHost((void**)&cpu_unvalid12, 64 * 64 * 64 * 64 * sizeof(unsigned char));	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }

	// 先頭４桁全組みあわせの計算
	calc_1_4col <<< dim3(64 * 64, 64), dim3(64) >>> (dev_res4, dev_unvalid4, dev_chrcode_mask);	cudaStatus = cudaGetLastError();			if (cudaStatus != cudaSuccess) { fprintf(stderr, "checkPassKernel launch failed: %s\n", cudaGetErrorString(cudaStatus)); goto Error; }
	cudaStatus = cudaDeviceSynchronize();		if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching calu14col!\n", cudaStatus); goto Error; }
	cudaStatus = cudaMemcpy(cpu_unvalid4, dev_unvalid4, (64 * 64 * 64 * 64) * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	//cudaStatus = cudaMemcpy(cpu_calcresult, dev_res4, (64 * 64 * 64 * 64 * 8) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	//for (int chkrs = 0; chkrs < 16/*(64 * 64 * 64)*/; ++chkrs) {
	//	printf("\n");

	//	for (int vidx = 0; vidx < 8; ++vidx) {
	//		printf("%02x ", cpu_calcresult[chkrs * 8 + vidx]);
	//	}
	//}

	validcnt = 0;
	for (int xor4idx = 0; xor4idx < (64 * 64 * 64 * 64); ++xor4idx) {
		// 無効文字が含まれていたら検索対象除外
		if (cpu_unvalid4[xor4idx]) {
			continue;
		}
		
		// 5-8桁の組合せを計算する
		calc_4col << < dim3(64 * 64, 64), dim3(64) >> > (dev_res8, dev_unvalid8, dev_res4, dev_chrcode_mask, xor4idx);
		cudaStatus = cudaDeviceSynchronize();		if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); goto Error; }
		cudaStatus = cudaMemcpy(cpu_unvalid8, dev_unvalid8, (64 * 64 * 64 * 64) * sizeof(unsigned char), cudaMemcpyDeviceToHost);

		//cudaStatus = cudaMemcpy(cpu_calcresult, dev_res8, (64 * 64 * 64 * 64 * 8) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		//for (int chkrs = 0; chkrs < (64 * 64 * 64); ++chkrs) {
		//	printf("\n");

		//	for (int vidx = 0; vidx < 8; ++vidx) {
		//		printf("%02X ", cpu_calcresult[chkrs * 8 + vidx]);
		//	}
		//}

		for (int xor8idx = 0; xor8idx < (64 * 64 * 64 * 64); ++xor8idx) {
			if (cpu_unvalid8[xor8idx]) {
				continue;
			}
			// 9～12桁分の組合せを計算する
			calc_4col << < dim3(64 * 64, 64), 64 >> > (dev_res12, dev_unvalid12, dev_res8, dev_chrcode_mask, xor8idx);
			cudaStatus = cudaDeviceSynchronize();		if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); goto Error; }
			cudaStatus = cudaMemcpy(cpu_unvalid12, dev_unvalid12, (64 * 64 * 64 * 64) * sizeof(unsigned char), cudaMemcpyDeviceToHost);

			//cudaStatus = cudaMemcpy(cpu_calcresult, dev_res12, (64 * 64 * 64 * 64 * 8) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
			//for (int chkrs = 0; chkrs < (64 * 64 * 64); ++chkrs) {
			//	printf("\n");

			//	for (int vidx = 0; vidx < 8; ++vidx) {
			//		printf("%02X ", cpu_calcresult[chkrs * 8 + vidx]);
			//	}
			//}

			for (int xor12idx = 0; xor12idx < (64 * 64 * 64 * 64); ++xor12idx) {
				if (cpu_unvalid12[xor12idx]) {
					continue;
				}
				// 13,14桁目の導出とチェックディジットが通っているかの判定
				calclast_validate << < 64, 64 >> > (dev_result, dev_validpass, dev_res12, xor12idx);
				cudaStatus = cudaDeviceSynchronize();		if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching validation!\n", cudaStatus); goto Error; }

				cudaStatus = cudaMemcpy(cpu_validpass, dev_validpass, 64 * 64 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
				for (int chkrs = 0; chkrs < (64 * 64); ++chkrs) {
					if (!cpu_validpass[chkrs]) continue;
					cudaStatus = cudaMemcpy(cpu_calcresult, dev_result, (64 * 64 * 8) * sizeof(unsigned char), cudaMemcpyDeviceToHost);

					printf("\n");

					printf("%08X %08X %08X %04X | ", xor4idx, xor8idx, xor12idx, chkrs);
					for (int vidx = 0; vidx < 8; ++vidx) {
						printf("%02X ", cpu_calcresult[chkrs * 8 + vidx]);
					}
					printf("| %c ", cvalid[cpu_validpass[chkrs]]);
					validcnt += cpu_validpass[chkrs];
				}
		//		goto FIN;

			}
			printf("\n%lld items\n", validcnt);
		}
	}
FIN:
	printf("found %lld items\n", validcnt);

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

	cudaFree(cpu_validpass);
	cudaFree(cpu_calcresult);

	cudaFree(dev_unvalid4);
	cudaFree(dev_unvalid8);
	cudaFree(dev_unvalid12);

	cudaFree(dev_res4);
	cudaFree(dev_res8);
	cudaFree(dev_res12);
	cudaFree(dev_result);

	cudaFree(dev_unvalid4);
	cudaFree(dev_unvalid8);
	cudaFree(dev_unvalid12);

	return(cudaStatus);

}
