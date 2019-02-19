// includes, system
#include "stdafx.h"
#include "mex.h"
#include "MultiDASEnv.h"
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

__device__ int  numZ_Dev, numX_Dev, datalength_Dev, transNum_Dev, pickLength_Dev, lgLength_Dev, inhibitor_Dev, windowSize_Dev;
__device__ float pitch_Dev, pSize_Dev, sampFreq_Dev, acoustVel_Dev, angleAperture_Dev, delayCoef_Dev;

int numZ_Host, numX_Host, datalength_Host, transNum_Host, pickLength_Host, lgLength_Host, inhibitor_Host, windowSize_Host;
float pitch_Host, pSize_Host, sampFreq_Host, acoustVel_Host, angleAperture_Host, delayCoef_Host;

float *data_Dev, *imgRecons_Dev, *dataPick, *y_real, *y_imag;
int *krev;
float *w_real, *w_imag;

// paraInt      : 0-transElement, 1-inhibitor, 2-lgLength, 3-dataLength, 4-windowSize
// paraIntDev   : 0-transElement, 1-inhibitor, 2-lgLength, 3-dataLength, 4-windowSize
// paraFloat : 0-delayCoef, 1-acoustVel, 2-pitch, 3-pixelSize, 4-samFreq, 5-angleApertureTan
// paraFloatDev : 0-delayCoef, 1-acoustVel, 2-pitch, 3-pixelSize, 4-samFreq, 5-angleApertureTan
extern "C" void initcudas(int *paraInt, float *paraFloat, float *data, int MAXZ_host, int MAXX_host)
{
	// imgSize_Host
	numZ_Host = MAXZ_host;
	numX_Host = MAXX_host;
	// paraInt_Host
	transNum_Host = *paraInt;
	inhibitor_Host = *(paraInt + 1);
	lgLength_Host = *(paraInt + 2);
	datalength_Host = *(paraInt + 3);
	windowSize_Host = *(paraInt + 4);
	pickLength_Host = 1;
	pickLength_Host <<= lgLength_Host;
	// paraFloat_Host
	delayCoef_Host = *paraFloat;
	acoustVel_Host = *(paraFloat + 1);
	pitch_Host = *(paraFloat + 2);
	pSize_Host = *(paraFloat + 3);
	sampFreq_Host = *(paraFloat + 4);
	angleAperture_Host = *(paraFloat + 5);

	// imgSize_Dev
	checkCudaErrors(cudaSetDevice(0));

	checkCudaErrors(cudaMemcpyToSymbol(numZ_Dev, &numZ_Host, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(numX_Dev, &numX_Host, sizeof(int)));
	// paraInt_Dev
	checkCudaErrors(cudaMemcpyToSymbol(transNum_Dev, &transNum_Host, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(inhibitor_Dev, &inhibitor_Host, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(lgLength_Dev, &lgLength_Host, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(datalength_Dev, &datalength_Host, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(windowSize_Dev, &windowSize_Host, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(pickLength_Dev, &pickLength_Host, sizeof(int)));
	// paraFloat_Dev
	checkCudaErrors(cudaMemcpyToSymbol(delayCoef_Dev, &delayCoef_Host, sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(acoustVel_Dev, &acoustVel_Host, sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(pitch_Dev, &pitch_Host, sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(pSize_Dev, &pSize_Host, sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(sampFreq_Dev, &sampFreq_Host, sizeof(float)));
	checkCudaErrors(cudaMemcpyToSymbol(angleAperture_Dev, &angleAperture_Host, sizeof(float)));

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
	checkCudaErrors(cudaFree(data_Dev));
	checkCudaErrors(cudaFree(imgRecons_Dev));
	checkCudaErrors(cudaFree(dataPick));
	checkCudaErrors(cudaFree(y_real));
	checkCudaErrors(cudaFree(y_imag));
	checkCudaErrors(cudaFree(w_real));
	checkCudaErrors(cudaFree(w_imag));
	checkCudaErrors(cudaFree(krev));
}

__device__ void getEvelope(int *krev, float *w_real, float *w_imag, float *x, float *y_real, float *y_imag)
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

__global__ void PArecon(float *data_Dev, float *imgRecons_Dev, float *dataPick, int *krev, float *w_real, float *w_imag, float *y_real, float *y_imag, int zdepth, int zstart)
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

	getEvelope(krev, w_real, w_imag, pickBeg, y_real + pointer, y_imag + pointer);

	lenMax = pick_offset;
	for (int len = pick_offset - windowSize_Dev + 1; len < pick_offset + windowSize_Dev; ++len)
	{
		if (len >= 0 && len < pickLength_Dev && *(pickBeg + len) > *(pickBeg + lenMax))
		{
			lenMax = len;
		}
	}

	if (*(pickBeg + lenMax) > 0)
	{
		*(imgRecons_Dev + tidx*zdepth + bidx) = *(pickBeg + pick_offset);
		for (int i = 1; i <= inhibitor_Dev; ++i)
		{
			*(imgRecons_Dev + tidx*zdepth + bidx) *= *(pickBeg + pick_offset);
			*(imgRecons_Dev + tidx*zdepth + bidx) /= *(pickBeg + lenMax);
		}
	}

	*(imgRecons_Dev + tidx*zdepth + bidx) = sqrt(*(imgRecons_Dev + tidx*zdepth + bidx));
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
	// calcualte pixels which are posistioned at the same depth at the same time
	// so that the threads may spend similar time completing calculation
	PArecon << < grids, threads >> >(data_Dev, imgRecons_Dev, dataPick, krev, w_real, w_imag, y_real, y_imag, zdepth, zstart);

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

	initcudas(paraInt, paraFloat, data, MAXZ_host, MAXX_host);
	parecon(devID, MAXZ_host, 0, imgRecons);
	clearcudas();
}

