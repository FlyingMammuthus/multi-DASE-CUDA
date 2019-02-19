// includes, system
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes CUDA
#include <cuda_runtime.h>

// includes, project
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples

//define parameter
#define pi 3.1415
#define MAXCUDADEVICES 1
#define threadNum 246

#define transNum_Host *paraInt_Host
#define inhibitor_Host *(paraInt_Host + 1)
#define lgLength_Host *(paraInt_Host + 2)
#define datalength_Host *(paraInt_Host + 3)
#define pickLength_Host *(paraInt_Host + 4)
#define numZ_Host *(paraInt_Host + 5)
#define numX_Host *(paraInt_Host + 6)

#define delayCoef_Host *paraFloat_Host
#define acoustVel_Host *(paraFloat_Host + 1)
#define pitch_Host *(paraFloat_Host + 2)
#define pSize_Host *(paraFloat_Host + 3)
#define sampFreq_Host *(paraFloat_Host + 4)
#define angleAperture_Host *(paraFloat_Host + 5)

#define transNum_Dev *paraInt_Dev
#define inhibitor_Dev *(paraInt_Dev + 1)
#define lgLength_Dev *(paraInt_Dev + 2)
#define datalength_Dev *(paraInt_Dev + 3)
#define pickLength_Dev *(paraInt_Dev + 4)
#define numZ_Dev *(paraInt_Dev + 5)
#define numX_Dev *(paraInt_Dev + 6)

#define delayCoef_Dev *paraFloat_Dev
#define acoustVel_Dev *(paraFloat_Dev + 1)
#define pitch_Dev *(paraFloat_Dev + 2)
#define pSize_Dev *(paraFloat_Dev + 3)
#define sampFreq_Dev *(paraFloat_Dev + 4)
#define angleAperture_Dev *(paraFloat_Dev + 5)

int *paraInt_Host, *paraInt_Dev, *krev;
float *paraFloat_Host, *paraFloat_Dev, *w_real, *w_imag, *data_Dev, *imgRecons_Dev, *dataPick, *y_real, *y_imag;

// paraInt             : 0-transElement, 1-inhibitor, 2-lgLength, 3-dataLength
// paraInt_Host/_Dev   : 0-transElement, 1-inhibitor, 2-lgLength, 3-dataLength, 4-pickLength, 5-numZ, 6-numX
// paraFloat           : 0-delayCoef, 1-acoustVel, 2-pitch, 3-pixelSize, 4-samFreq
// paraFloat_Host/_Dev : 0-delayCoef, 1-acoustVel, 2-pitch, 3-pixelSize, 4-samFreq, 5-angleAperture
extern "C" void initcudas(int *paraInt, float *paraFloat, float *data, int MAXZ_host, int MAXX_host)
{	
	paraInt_Host = (int *)malloc(7 * sizeof(int));
	memcpy(paraInt_Host, paraInt, 4 * sizeof(int));
	*(paraInt_Host + 4) = 1;
	*(paraInt_Host + 4) <<= *(paraInt + 2);
	*(paraInt_Host + 5) = MAXZ_host;
	*(paraInt_Host + 6) = MAXX_host;

	paraFloat_Host = (float *)malloc(6 * sizeof(float));
	memcpy(paraFloat_Host, paraFloat, 5 * sizeof(float));
	*(paraFloat_Host + 5) = 0.5;

	// imgSize_Dev
	checkCudaErrors(cudaSetDevice(0));

	checkCudaErrors(cudaMalloc((void**)&(paraInt_Dev), 7 * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&(paraFloat_Dev), 6 * sizeof(float)));
	checkCudaErrors(cudaMemcpy(paraInt_Dev, paraInt_Host, 7 * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(paraFloat_Dev, paraFloat_Host, 6 * sizeof(float), cudaMemcpyHostToDevice));

	printf_s("Device parameter setting...\n");
	// float*_Dev malloc
	checkCudaErrors(cudaMalloc((void**)&(data_Dev), transNum_Host*datalength_Host*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&(imgRecons_Dev), numZ_Host*numX_Host*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&(dataPick), pickLength_Host*threadNum*numX_Host*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&(y_real), pickLength_Host*threadNum*numX_Host*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&(y_imag), pickLength_Host*threadNum*numX_Host*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&(w_real), (pickLength_Host - 1)*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&(w_imag), (pickLength_Host - 1)*sizeof(float)));
	// int*_Dev malloc
	checkCudaErrors(cudaMalloc((void**)&(krev), pickLength_Host*sizeof(int)));

	//calculate parameter of fft
	int *krev_Host = (int *)malloc(pickLength_Host*sizeof(int));
	for (int k = 0; k < pickLength_Host; ++k)
	{
		int r = k;
		*(krev_Host + k) = (r & 0x1);
		for (int j = 1; j < lgLength_Host; ++j)
		{
			*(krev_Host + k) = (*(krev_Host + k)) << 1;
			r = r >> 1;
			if (r & 0x1) ++(*(krev_Host + k));
		}
	}
	checkCudaErrors(cudaMemcpy(krev, krev_Host, pickLength_Host*sizeof(int), cudaMemcpyHostToDevice));
	free(krev_Host);

	float *wreal_Host = (float *)malloc((pickLength_Host - 1)*sizeof(float)),
		*wimag_Host = (float *)malloc((pickLength_Host - 1)*sizeof(float));
	int m = 1;
	float wm_real, wm_imag, w_realRec, w_imagRec, *wreal_now = wreal_Host, *wimag_now = wimag_Host;
	for (int s = 1; s <= lgLength_Host; ++s)
	{
		m *= 2;
		wm_real = cos(2 * pi * 1 / m);
		wm_imag = -sin(2 * pi * 1 / m);
		w_realRec = 1;
		w_imagRec = 0;
		for (int j = 0; j < (m / 2); ++j)
		{
			//w = w * wm = t * wm;
			*(wreal_now + j) = w_realRec;
			*(wimag_now + j) = w_imagRec;
			w_realRec = *(wreal_now + j)*wm_real - *(wimag_now + j)*wm_imag;
			w_imagRec = *(wreal_now + j)*wm_imag + *(wimag_now + j)*wm_real;
		}
		wreal_now += m / 2;
		wimag_now += m / 2;
	}
	checkCudaErrors(cudaMemcpy(w_real, wreal_Host, (pickLength_Host - 1)*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(w_imag, wimag_Host, (pickLength_Host - 1)*sizeof(float), cudaMemcpyHostToDevice));
	free(wreal_Host);
	free(wimag_Host);

	// copy host data to device
	checkCudaErrors(cudaMemcpy(data_Dev, data, transNum_Host*datalength_Host*sizeof(float), cudaMemcpyHostToDevice));
}

extern "C" void clearcudas()
{
	checkCudaErrors(cudaFree(paraInt_Dev));
	checkCudaErrors(cudaFree(paraFloat_Dev));
	checkCudaErrors(cudaFree(data_Dev));
	checkCudaErrors(cudaFree(imgRecons_Dev));
	checkCudaErrors(cudaFree(dataPick));
	checkCudaErrors(cudaFree(y_real));
	checkCudaErrors(cudaFree(y_imag));
	checkCudaErrors(cudaFree(w_real));
	checkCudaErrors(cudaFree(w_imag));	
	checkCudaErrors(cudaFree(krev));
	free(paraInt_Host);
	free(paraFloat_Host);
}

__device__ void getEvelope(int *paraInt_Dev, float *paraFloat_Dev, int *krev, float *w_real, float *w_imag, float *x, float *y_real, float *y_imag)
{
	// 2_DFT
	float *px = x;
	for (int k = 0; k < pickLength_Dev; ++k)
	{
		*(y_real + *(krev + k)) = *px;
		*(y_imag + *(krev + k)) = 0;
		++px;
	}
	int m = 1;
	float t_real, t_imag, u_real, u_imag, *wreal_now = w_real, *wimag_now = w_imag;
	for (int s = 1; s <= lgLength_Dev; ++s)
	{
		m *= 2;
		for (int k = 0; k < pickLength_Dev; k += m)
		{
			for (int j = 0; j < (m / 2); ++j)
			{
				//t = w * (*(y+k+j+m/2))
				t_real = *(wreal_now + j)*(*(y_real + k + j + m / 2)) - *(wimag_now + j)*(*(y_imag + k + j + m / 2));
				t_imag = *(wreal_now + j)*(*(y_imag + k + j + m / 2)) + *(wimag_now + j)*(*(y_real + k + j + m / 2));
				u_real = *(y_real + k + j);
				u_imag = *(y_imag + k + j);
				*(y_real + k + j) = u_real + t_real;
				*(y_imag + k + j) = u_imag + t_imag;
				*(y_real + k + j + m / 2) = u_real - t_real;
				*(y_imag + k + j + m / 2) = u_imag - t_imag;
			}
		}
		wreal_now += m / 2;
		wimag_now += m / 2;
	}

	// HilbertTran
	int count = 0;
	for (count = 1; count < pickLength_Dev / 2; ++count) //pickLength must be even
	{
		(*(y_real + count)) *= 2;
		(*(y_imag + count)) *= 2;
	}
	for (count += 1; count < pickLength_Dev; ++count)
	{
		(*(y_real + count)) *= 0;
		(*(y_imag + count)) *= 0;
	}
	for (int k = 0; k < pickLength_Dev; ++k)
	{
		count = *(krev + k);
		if (count == k)
		{
			*(y_imag + k) = -(*(y_imag + k));
		}
		else if (k < count)
		{
			t_real = *(y_real + k);
			t_imag = *(y_imag + k);
			*(y_real + k) = *(y_real + count);
			*(y_imag + k) = -(*(y_imag + count));
			*(y_real + count) = t_real;
			*(y_imag + count) = -t_imag;
		}
	}
	m = 1;
	wreal_now = w_real;
	wimag_now = w_imag;
	for (int s = 1; s <= lgLength_Dev; ++s)
	{
		m *= 2;
		for (int k = 0; k < pickLength_Dev; k += m)
		{
			for (int j = 0; j < (m / 2); ++j)
			{
				//t = w * (*(y+k+j+m/2))
				t_real = *(wreal_now + j)*(*(y_real + k + j + m / 2)) - *(wimag_now + j)*(*(y_imag + k + j + m / 2));
				t_imag = *(wreal_now + j)*(*(y_imag + k + j + m / 2)) + *(wimag_now + j)*(*(y_real + k + j + m / 2));
				u_real = *(y_real + k + j);
				u_imag = *(y_imag + k + j);
				*(y_real + k + j) = u_real + t_real;
				*(y_imag + k + j) = u_imag + t_imag;
				*(y_real + k + j + m / 2) = u_real - t_real;
				*(y_imag + k + j + m / 2) = u_imag - t_imag;
			}
		}
		wreal_now += m / 2;
		wimag_now += m / 2;
	}
	int div_len = pickLength_Dev*pickLength_Dev;
	for (int i = 0; i < pickLength_Dev; ++i)
	{
		*(x + i) = (*(y_real + i))*(*(y_real + i)) + (*(y_imag + i))*(*(y_imag + i));
		*(x + i) /= div_len;
	}
}

__global__ void PArecon(int *paraInt_Dev, float *paraFloat_Dev, float *data_Dev, float *imgRecons_Dev, float *dataPick, int *krev, float *w_real, float *w_imag, float *y_real, float *y_imag, int zdepth, int zstart)
{
	// access thread id
	const unsigned int tidx = threadIdx.x;
	// access block id
	const unsigned int bidx = blockIdx.x;
	if (bidx < zstart)
	{
		return;
	}
	float Distan;
	float Y, Z, y;
	int POINTER, pointer = pickLength_Dev*((bidx % threadNum)*numX_Dev + tidx);
	float *pickBeg = dataPick + pointer;
	int pick_offset = pickLength_Dev / 2;

	Z = bidx * pSize_Dev;
	Y = tidx * pSize_Dev;

	int y_start = (int)((Y - Z*angleAperture_Dev) / pitch_Dev - 0.5);
	if (y_start < 0)
	{
		y_start = 0;
	}
	int y_end = (int)((Y + Z*angleAperture_Dev) / pitch_Dev - 0.5);
	if (y_end > transNum_Dev - 1)
	{
		y_end = transNum_Dev - 1;
	}

	for (int len = 0; len < pickLength_Dev; ++len)
	{
		*(pickBeg + len) = 0;
	}

	int lenMax;
	for (int bidy = y_start; bidy <= y_end; ++bidy)
	{
		y = (bidy + 0.5) * pitch_Dev;
		Distan = sqrt((Y - y)*(Y - y) + Z*Z);
		POINTER = (int)((Distan / acoustVel_Dev - delayCoef_Dev)*sampFreq_Dev + 0.5) - pick_offset;
		lenMax = pickLength_Dev;
		if (POINTER + lenMax >= datalength_Dev){
			lenMax = datalength_Dev - 1 - POINTER;
		}
		if (POINTER >= 0 && POINTER < datalength_Dev)
		{
			POINTER = POINTER + bidy*datalength_Dev;
			for (int len = 0; len < lenMax; ++len)
			{
				*(pickBeg + len) += *(data_Dev + POINTER + len);
			}
		}
	}

	getEvelope(paraInt_Dev, paraFloat_Dev, krev, w_real, w_imag, pickBeg, y_real + pointer, y_imag + pointer);

	lenMax = 0;
	for (int len = 1; len < pickLength_Dev - 1; ++len)
	{
		if (*(pickBeg + len) > *(pickBeg + lenMax))
		{
			lenMax = len;
		}
	}

	if (*(pickBeg + lenMax) > 0)
	{
		*(imgRecons_Dev + tidx*zdepth + bidx) = *(pickBeg + pick_offset);
		for (int i = 1; i < inhibitor_Dev; ++i)
		{
			*(imgRecons_Dev + tidx*zdepth + bidx) *= *(pickBeg + pick_offset);
			*(imgRecons_Dev + tidx*zdepth + bidx) /= *(pickBeg + lenMax);
		}
	}
	__syncthreads();
}

__host__ void parecon(int cudadeviceindex, int zdepth, int zstart, float *imgRecons)
{
	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	checkCudaErrors(cudaSetDevice(cudadeviceindex));

	// setup execution parameters
	dim3 grids(numZ_Host, 1, 1);
	dim3 threads(numX_Host, 1, 1);

	// execute the kernel 
	PArecon << < grids, threads >> >(paraInt_Dev, paraFloat_Dev, data_Dev, imgRecons_Dev, dataPick, krev, w_real, w_imag, y_real, y_imag, zdepth, zstart);

	// check if kernel execution generated and error
	getLastCudaError("Kernel execution failed");

	// copy result from device to host
	checkCudaErrors(cudaMemcpy(imgRecons, imgRecons_Dev, numX_Host*zdepth*sizeof(float), cudaMemcpyDeviceToHost));

	// check if kernel execution generated and error
	getLastCudaError("Kernel execution failed");
}

void MultiDASEnv(int *paraInt, float *paraFloat, float *data, float *imgRecons, int MAXZ_host, int MAXX_host)
{

	int devID = 0;

	printf_s("Initializing...\n");
	initcudas(paraInt, paraFloat, data, MAXZ_host, MAXX_host);

	printf_s("Reconstructing...\n");
	parecon(devID, MAXZ_host, 0, imgRecons);

	printf_s("Clearing...\n");
	clearcudas();
}

int main()
{
	using namespace std;	
	int *paraInt = new int[4];
	paraInt[0] = 128;
	paraInt[1] = 3;
	paraInt[2] = 4;
	paraInt[3] = 1024;
	float *paraFloat = new float[5];
	paraFloat[0] = 0;
	paraFloat[1] = 1.54;
	paraFloat[2] = 0.3;
	paraFloat[3] = 0.1;
	paraFloat[4] = 40;
	int MAXX_host = 384,
		MAXZ_host = (int)(paraInt[3] * paraFloat[1] / paraFloat[4] / paraFloat[3]);
	ifstream fin("C:\\Users\\MX\\Documents\\research\\PA Reconstructions\\GPUProg\\template\\signal\\Rf_032918_113516_OBP_PA_64_15331342.txt");
	float *data = (float *)malloc(paraInt[0] * paraInt[3] * sizeof(float));

	for (int i = 0; i < paraInt[0] * paraInt[3]; ++i)
	{
		fin >> *(data + i);
	}
	fin.close();

	// paraInt   : 0-transElement, 1-inhibitor, 2-lgLength, 3-dataLength
	// paraFloat : 0-delayCoef, 1-acoustVel, 2-pitch, 3-pixelSize, 4-samFreq


	printf_s("PA reconstructing...\n");

	float *imgRecons = (float *)malloc(MAXX_host*MAXZ_host*sizeof(float));

	MultiDASEnv(paraInt, paraFloat, data, imgRecons, MAXZ_host, MAXX_host);

	ofstream fout("C:\\Users\\MX\\Documents\\research\\PA Reconstructions\\GPUProg\\template\\recons\\fig_recons.txt");
	for (int i = 0; i < MAXZ_host*MAXX_host; ++i)
	{
		fout << *(imgRecons + i);
		fout << "  ";
	}
	fout.close();

	free(data);
}

