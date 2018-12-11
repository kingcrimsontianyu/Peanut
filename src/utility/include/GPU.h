#if !defined(_GPU_H_)
#define _GPU_H_

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utility/include/Exception.h"

#define HANDLE_GPU_ERROR(err) \
do \
{ \
    if(err != cudaSuccess) \
    { \
        int currentDevice; \
        cudaGetDevice(&currentDevice); \
        std::cerr << "CUDA device assert: device = " \
                  << currentDevice << ", " << cudaGetErrorString(err) \
                  << " in " << __FILE__ \
                  << " at line " << __LINE__ << std::endl; \
        throw Peanut::PeanutException("A GPU error is detected."); \
    } \
} \
while(0)


#endif
