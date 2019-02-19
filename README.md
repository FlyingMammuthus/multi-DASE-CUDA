# multi-DASE-CUDA
基于DAS（delay and sum的延时求和代码）的改进算法multi-DASE的并行化处理
实现函数
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])

参数详细定义查看MultiDASEnv.cpp

# 使用教程
环境配置：CUDA 8.0及以上
文件配置了matlab接口(MultiDASEnv.cpp)，如无需matlab，需要自定义原始数据格式（txt）作为程序输入。

# 并行化模型
在重建图像时，首先每个像素的计算是独立，是本算法中的第一层并行；其次，在计算每个像素点像素值时，需要按照时间（即不同采样点）求和，每个时间点的计算也是独立的，这是本算法的第二层并行处理；最后，算法中需要设计求取包络，需要希尔伯特变换，在本算法中是用FFT（快速傅里叶变换）实现希尔伯特变换，因而本算法针对信号的特征，重写了相应的FFT并行化算法，这是本算法的第三层并行化。

# 实例展示
该实例采用matlab作为采集数据的平台，数据传输到VS进行重建，随后返回重建的图像，显示到matlab端。
