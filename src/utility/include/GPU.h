#if !defined(_GPU_H_)
#define _GPU_H_

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utility/include/Exception.h"

//------------------------------------------------------------
// Get CUDA API error code. Throw exception on failure.
//------------------------------------------------------------
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

namespace Peanut
{
//------------------------------------------------------------
//------------------------------------------------------------
void QueryGPU();

//------------------------------------------------------------
//------------------------------------------------------------
void SetGPUByName(const std::string& name);

//------------------------------------------------------------
//------------------------------------------------------------
void SetGPUByCudaIndex(const int& cudaIdx);

} // end namespace Peanut

#endif
