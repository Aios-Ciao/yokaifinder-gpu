#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <string>
#include <forward_list>
#include <numeric>
#include <conio.h>

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

const unsigned char charvalidmask[64] = {
	0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x01,	// 0x06��0x07������
	0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x01,
	0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x01,
	0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x01,
	0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x01,
	0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x01,
	0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x01,
	0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,	// 0x38�ȏ������
};

const unsigned char chrcode[256] =
{
	0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
	0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
	0xF0,0x2D,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0x2B,0xFF,0xFF,
	0x25,0x04,0x0C,0x14,0x1C,0x24,0x05,0x0D,0x15,0x1D,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
	0xF0,0x00,0x08,0x10,0x18,0x20,0x28,0x30,0x01,0x09,0x11,0x19,0x21,0x29,0x31,0x02,
	0x0A,0x12,0x1A,0x22,0x2A,0x32,0x03,0x0B,0x13,0x1B,0x23,0xFF,0xFF,0xFF,0xFF,0xFF,
	0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
	0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
	0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
	0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
	0xFF,0xFF,0xFF,0xFF,0xFF,0x33,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
	0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0x35,0xFF,0xFF,0xFF,0xFF,0xFF,
	0xFF,0xFF,0xFF,0xFF,0xFF,0x2C,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
	0xFF,0x34,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
	0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
	0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
};

bool countfrompass(string pass, unsigned int *st1_4, unsigned int *st5_8, unsigned int *st9_12, unsigned int *st13_14)
{
	// 14���ł��邱��
	if (pass.size() != 14) { return (false); }

	// �L�����������ō\������邱��
	for (auto ch : pass) { if (chrcode[ch] == 0xFF) { return (false); } }

	unsigned int stwk;
	stwk   = 0; stwk += chrcode[pass[ 0]];
	stwk <<= 6; stwk += chrcode[pass[ 1]];
	stwk <<= 6; stwk += chrcode[pass[ 2]];
	stwk <<= 6; stwk += chrcode[pass[ 3]];
	*st1_4 = stwk;
	stwk   = 0; stwk += chrcode[pass[ 4]];
	stwk <<= 6; stwk += chrcode[pass[ 5]];
	stwk <<= 6; stwk += chrcode[pass[ 6]];
	stwk <<= 6; stwk += chrcode[pass[ 7]];
	*st5_8 = stwk;
	stwk   = 0; stwk += chrcode[pass[ 8]];
	stwk <<= 6; stwk += chrcode[pass[ 9]];
	stwk <<= 6; stwk += chrcode[pass[10]];
	stwk <<= 6; stwk += chrcode[pass[11]];
	*st9_12 = stwk;
	stwk   = 0; stwk += chrcode[pass[12]];
	stwk <<= 6; stwk += chrcode[pass[13]];
	*st13_14 = stwk;

	return true;
}

cudaError_t chkthread( forward_list<string>	&, unsigned int, unsigned int, unsigned int, unsigned int);

int main(int argc, char *argv[])
{
	cudaError_t				cudaStatus;
	forward_list<string>	vPasswordList;
	unsigned int			sf1_4(0), sf5_8(0), sf9_12(0), sf13_14(0);
	string					filename, startpass;
	ofstream				writefile;

	switch (argc) {
	case 3: // �T���J�n������w�肠��
		startpass = argv[2];
		{
			bool valid = countfrompass(startpass, &sf1_4, &sf5_8, &sf9_12, &sf13_14);
			if (!valid) {
				fprintf(stderr, "�p�X���[�h�T���J�n�̎w�肪�Ԉ���Ă��܂��B\n");
				return (-1);
			}
		}
		// no break
	case 2:	// �o�̓t�@�C�����w�肠��
		filename = argv[1];
		break;
	case 1:
	default:
		fprintf(stderr, "unpass outfilename.txt [startpass]\n");
		return (-1);
	}

	writefile.open(filename, ios::out);
	if (!writefile.fail()) {
		printf(	"�t�@�C���A�N�Z�X���\�ł��邱�Ƃ��m�F�ł��܂���\n"
				"��U�N���[�Y���܂��B\n"
				"�w�肵���t�@�C���ɂ͐G�炸�ɂ��҂���������\n"
		);
	}
	writefile.close();

	chkthread(vPasswordList, sf1_4, sf5_8, sf9_12, sf13_14);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceReset failed!"); return 1; }

	printf("�\�[�g��...");
	vPasswordList.sort();
	printf("����\n");

	size_t listlen = distance(vPasswordList.begin(), vPasswordList.end());
	if (listlen > 0) {
		cout << listlen << "��������܂����B" << endl;

		printf("�t�@�C���o�͒�...\n");
		writefile.open(filename, ios::out);
		if (!writefile.fail()) {
			printf("�t�@�C�����J���܂����B\n"
				"�w�肵���t�@�C���ɂ͐G�炸�ɂ��҂����������B\n"
			);
			for (auto it : vPasswordList) {
				writefile << it << endl;
			}
		}
		writefile.close();
	}
	else {
		cout << "�w��͈͂ł͌�����܂���ł����B" << endl;
		cout << "�t�@�C���o�͂̓X�L�b�v���܂��B" << endl;

	}
	printf("����\n");

	return 0;
}

// �J�E���^����p�X������ւ̕ϊ�
void count2pass4(unsigned int count, char *pass)
{
	static const char _dict[] = { "AHOV16  BIPW27  CJQX38  DKRY49  ELSZ50  FMT-�!  GNU.Ѻ          " };
	pass[3] = _dict[count & 0x3F];	count >>= 6;
	pass[2] = _dict[count & 0x3F];	count >>= 6;
	pass[1] = _dict[count & 0x3F];	count >>= 6;
	pass[0] = _dict[count & 0x3F];
}
void count2pass2(unsigned int count, char *pass)
{
	static const char _dict[] = { "AHOV16  BIPW27  CJQX38  DKRY49  ELSZ50  FMT-�!  GNU.Ѻ          " };
	pass[1] = _dict[count & 0x3F];	count >>= 6;
	pass[0] = _dict[count & 0x3F];
}


cudaError_t chkthread(
	forward_list<string>	&vPasswordList,
	unsigned int			stidx1,
	unsigned int			stidx2,
	unsigned int			stidx3,
	unsigned int			stidx4
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
	unsigned int  searchidx1;				// �T���J�n�ʒu(�擪4��)
	unsigned int  searchidx5;				// �T���J�n�ʒu(5-8��)
	unsigned int  searchidx9;				// �T���J�n�ʒu(9-12��)
	unsigned int  searchidx13;				// �T���J�n�ʒu(13-14��)

	char	passstr[15] = { "AAAAAAAAAAAAAA" };

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

	printf("\n�S�T���J�n");
	validcnt = 0;
	for (searchidx1 = stidx1; searchidx1 < (64 * 64 * 64 * 64); ++searchidx1) {
		// �����������܂܂�Ă����猟���Ώۏ��O
		if (cpu_unvalid4[searchidx1]) {
			continue;
		}
		count2pass4(searchidx1, &passstr[0]);
		printf("\nTotal %lld items\n", validcnt);
		printf("\n1-4 Loop %5.3f%% %.4s", (float)searchidx1 / (64 * 64 * 64 * 64), &passstr[0]);
		
		// 5-8���̑g�������v�Z����
		calc_4col << < dim3(64 * 64, 64), dim3(64) >> > (dev_res8, dev_unvalid8, dev_res4, dev_chrcode_mask, searchidx1);
		cudaStatus = cudaDeviceSynchronize();		if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); goto Error; }
		cudaStatus = cudaMemcpy(cpu_unvalid8, dev_unvalid8, (64 * 64 * 64 * 64) * sizeof(unsigned char), cudaMemcpyDeviceToHost);

		for (searchidx5 = stidx2; searchidx5 < (64 * 64 * 64 * 64); ++searchidx5) {
			if (cpu_unvalid8[searchidx5]) {
				continue;
			}
			count2pass4(searchidx5, &passstr[4]);
			printf("\nTotal %lld items\n", validcnt);
			printf("\n5-8 Loop %5.3f%% %.4s", (float)searchidx5 / (64 * 64 * 64 * 64), &passstr[4]);
			// 9�`12�����̑g�������v�Z����
			calc_4col << < dim3(64 * 64, 64), 64 >> > (dev_res12, dev_unvalid12, dev_res8, dev_chrcode_mask, searchidx5);
			cudaStatus = cudaDeviceSynchronize();		if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); goto Error; }
			cudaStatus = cudaMemcpy(cpu_unvalid12, dev_unvalid12, (64 * 64 * 64 * 64) * sizeof(unsigned char), cudaMemcpyDeviceToHost);

			for (searchidx9 = stidx3; searchidx9 < (64 * 64 * 64 * 64); ++searchidx9) {
				if (cpu_unvalid12[searchidx9]) {
					continue;
				}
				count2pass4(searchidx9, &passstr[8]);
				// 13,14���ڂ̓��o�ƃ`�F�b�N�f�B�W�b�g���ʂ��Ă��邩�̔���
				calclast_validate << < 64, 64 >> > (dev_result, dev_validpass, dev_res12, searchidx9);
				cudaStatus = cudaDeviceSynchronize();		if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching validation!\n", cudaStatus); goto Error; }

				cudaStatus = cudaMemcpy(cpu_validpass, dev_validpass, 64 * 64 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
				for (searchidx13 = stidx4; searchidx13 < (64 * 64); ++searchidx13) {
					if (!cpu_validpass[searchidx13]) continue;
					cudaStatus = cudaMemcpy(cpu_calcresult, dev_result, (64 * 64 * 8) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
					count2pass2(searchidx13, &passstr[12]);

					vPasswordList.push_front(string(passstr));

					printf("\n%s | ", passstr);
					for (int vidx = 0; vidx < 8; ++vidx) {
						printf("%02X ", cpu_calcresult[searchidx13 * 8 + vidx]);
					}
					validcnt += cpu_validpass[searchidx13];
				}

				// ESC�L�[���̓`�F�b�N
				if (_kbhit() && (_getch() == 27)) {
					printf("\nChecked up to the item \"%s\"."
					"\n�T����ł��؂�܂���\n\n", passstr);
					goto FIN;
				}
			}
		}
	}
	printf("\n�T������\n\n");
FIN:
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
