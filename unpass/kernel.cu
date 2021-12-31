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
//	_31F4x[2] = 0x0E;							// 6 ������

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
	unsigned char		*work4col,		// �i�[�� �����̗v�f�͋��߂��O���[�o���C���f�b�N�X*8����͂��܂�8byte
	unsigned char		*valid,			// �L���ȕ����R�[�h�ō\�����ꂽ��
	const unsigned char *chrmask		// �L�������R�[�h���̔���p�r�b�g�}�X�N
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
	/*$31F6*/	work4col[gblidx + 2] = 0x0E;	// �Œ�l�A�s�g�p
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

// ����܂ł̌��̒l���g���Ēǉ���4�����̌v�Z�l�����߂�
__global__ void calc_4col(
	unsigned char		*work8col,		// �i�[�� �����̗v�f�͋��߂��O���[�o���C���f�b�N�X*8����͂��܂�8byte
	unsigned char		*unvalid8,		// �����������܂܂�Ă�����true
	const unsigned char	*work4col,		// ����܂łɌv�Z���Ă������Z����
	const unsigned char *chrmask,		// �L�������R�[�h���̔���p�r�b�g�}�X�N
	const unsigned int	idx4			// work4col�ɑΉ�����C���f�b�N�X�l
)
{
	// �X���b�h�ԍ����珑�����ݐ�C���f�b�N�X���v�Z
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

// �c��̂P�����o�ƍŏI���܂ł̃`�F�b�N�f�B�W�b�g�v�Z�A�L������
__global__ void calclast_validate(
	unsigned char		*result,	// OUT  14�����̊e�f�[�^�v�Z����
	unsigned char		*valid,		// OUT  �p�X���[�h�Ƃ��Đ��藧�Ȃ�1(true)�łȂ����0(false)
	const unsigned char	*work,		// IN	12���ڂ܂łœ��o���ꂽ�r���l
	const unsigned int	workidx		// IN	blockIdx.x�̃I�t�Z�b�g
)
{
	const unsigned int	wkidx = workidx * 8;
	const unsigned int	gblidx = ((blockIdx.x * blockDim.x) + threadIdx.x) * 8;
	const unsigned char col13 = blockIdx.x;			// �u���b�Nidx.x��13���ڂ̕����R�[�h���
	const unsigned char col14 = threadIdx.x;		// �X���b�hidx.x��14���ڂ̕����R�[�h���

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

// const char _dict[] = { "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!.-nmc�Ѻ" };
const unsigned char charvalidmask[64] = {
	0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x01,	// 0x06��0x07������
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
	unsigned char *dev_chrcode_mask = 0;	// �L�������R�[�h����p�}�X�N		�L����0x00�A������0x80�B
	unsigned char *dev_validpass = 0;		// �v�Z�������Ȃ��1������

	unsigned char *cpu_validpass = 0;		// �L���p�X���[�h���茋��(CPU��)
	unsigned char *cpu_calcresult = 0;		// �v�Z���ʊm�F�p

	unsigned char *cpu_unvalid4 = 0;		// �L�������ō\������Ă��邩���(�擪4��)
	unsigned char *cpu_unvalid8 = 0;		// �L�������ō\������Ă��邩���(5-8��)
	unsigned char *cpu_unvalid12 = 0;		// �L�������ō\������Ă��邩���(9-12��)

	unsigned char *dev_unvalid4 = 0;		// �L�������ō\������Ă��邩���(�擪4��)
	unsigned char *dev_unvalid8 = 0;		// �L�������ō\������Ă��邩���(5-8��)
	unsigned char *dev_unvalid12 = 0;		// �L�������ō\������Ă��邩���(9-12��)

	unsigned char *dev_res4 = 0;			// �擪4���Ԃ�̌v�Z���ʂƃL�����[���
	unsigned char *dev_res8 = 0;			// 5-8���܂ł̌v�Z���ʂƃL�����[���
	unsigned char *dev_res12 = 0;			// 9-12���܂Ōv�Z���ʂƃL�����[���
	unsigned char *dev_result = 0;			// 14���̌v�Z����

	unsigned long long validcnt = 0;		// �`�F�b�N�f�B�W�b�g��ʂ����p�X���[�h�̌�

	static const char cvalid[2] = { '.', 'O' };	// ����\���p


	cudaStatus = cudaSetDevice(0);		if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?"); goto Error; }
	
	// ���o�p�e�[�u��(const)
	cudaStatus = cudaMalloc((void**)&dev_chrcode_mask, 64 * sizeof(unsigned char));						if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }

	// 4�����̊e�f�[�^�Z�o�p�̈�m��
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

	// �擪�S���S�g�݂��킹�̌v�Z
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
		// �����������܂܂�Ă����猟���Ώۏ��O
		if (cpu_unvalid4[xor4idx]) {
			continue;
		}
		
		// 5-8���̑g�������v�Z����
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
			// 9�`12�����̑g�������v�Z����
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
				// 13,14���ڂ̓��o�ƃ`�F�b�N�f�B�W�b�g���ʂ��Ă��邩�̔���
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
