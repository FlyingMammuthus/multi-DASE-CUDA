#include "stdafx.h"
#include "MultiDASEnv.h"
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	mexPrintf("Parameter type checking...\n");
	if ((nrhs != 3) || (nlhs != 1) || !mxIsInt32(prhs[0]) || !mxIsSingle(prhs[1]) || !mxIsSingle(prhs[2]) ||
		(int)mxGetM(prhs[0]) != 5 || (int)mxGetN(prhs[0]) != 1 || (int)mxGetM(prhs[1]) != 6 || (int)mxGetN(prhs[1]) != 1)
	{
		mexErrMsgTxt("Invaid input or output!!!\n\
Proper number of input and output should be 3 and 1.\n\
Input vector data type must be Int32, Single, Single.\n\
The first input vector should be a column vector defined as:\n\
    [transElement, inhibitor, lgLength, dataLength, windowSize].\n\n\
The second input vector should be a column vector defined as:\n\
    [delayCoef, acoustVel, pitch, pixelSize, samFreq, angleApertureTan].\n\n\
The third vector should be a column vector containing signals of all channels.\n\n\
Parameters defined as:\n\
	transElement      : number of transducer, usually 64/128/256.\n\
	inhibitor         : the pow of inhibitor, default set is 3\n\
	lgLength          : log2(pickLength), therefore pickLength is featured as 2 ^ lgLength.\n\
    dataLength        : length of data.\n\
	windowSize        : size of window where the maximum of the picked data is searched([-windowSize, windowSize]).\n\
    delayCoef         : the coefficient of delay, delay = delayCoef * nz.\n\
	acoustVel         : speed of sound, a typical value is 1.54.\n\
	pitch             : space between neighbouring transducer element, a typical value is 0.298 for L7-4.\n\
	pixelSize         : size of pixel(the same in x and z direction), usually set to pitch/3.\n\
	samFreq           : sampling frequency MHz.\n\
    angleApertureTan  : tan value of the aperture angle.\n\
	data              : a 1 - D array with length of dataLength*transElement.\n");
	}
	// paraInt   : 0-transElement, 1-inhibitor, 2-lgLength, 3-dataLength, 4-windowSize
	// paraFloat : 0-delayCoef, 1-acoustVel, 2-pitch, 3-pixelSize, 4-samFreq, 5-angleAperture
	int *paraInt = (int *)mxGetData(prhs[0]);
	float *paraFloat = (float *)mxGetData(prhs[1]);
	float *data = (float *)mxGetData(prhs[2]);
	int MAXZ = (int)(paraInt[3] * paraFloat[1] / paraFloat[4] / paraFloat[3]),
		MAXY = (int)(paraInt[0] * paraFloat[2] / paraFloat[3]);
	//create the output vector
	plhs[0] = mxCreateNumericMatrix(MAXZ, MAXY, mxSINGLE_CLASS, mxREAL);

	float* imgRecons = (float* )mxGetData(plhs[0]);

	MultiDASEnv(paraInt, paraFloat, data, imgRecons, MAXZ, MAXY);
}