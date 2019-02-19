#ifndef __MULTIPLE_DELAY_AND_SUM_ENVELOP__
#define __MULTIPLE_DELAY_AND_SUM_ENVELOP__

/*
transElement : number of transducer, usually 64/128/256
inhibitor    : the pow of inhibitor, default set is 3
lgLength     : log2(pickLength), therefore pickLength is featured as 2^lgLength
dataLength   : length of data
delayCoef    : the coefficient of delay, delay = delayCoef * nz
acoustVel    : speed of sound
pitch        : space between neighbouring transducer element
pixelSize    : size of pixel (the same in x and z direction)
samFreq      : sampling frequency
data         : pointer to a 1-D array with length of dataLength*transElement
*/

extern void MultiDASEnv(int *paraInt, float *paraFloat, float *data, float *imgRecons, int MAXZ_host, int MAXX_host);



#endif // !__MULTIPLE_DELAY_AND_SUM_ENVELOP__
