#if !defined(_GPU_H_)
#define _GPU_H_

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utility/include/Exception.h"
#include <iostream>

//------------------------------------------------------------
// Get CUDA API error code. Throw exception on failure.
// Use this macro for every CUDA call.
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
// Show information of all available GPUs.
//------------------------------------------------------------
void QueryGPU();

//------------------------------------------------------------
// Call cudaSetDevice() on the selected GPU.
//------------------------------------------------------------
void SetGPUByName(const std::string& name);

//------------------------------------------------------------
// Call cudaSetDevice() on the selected GPU.
//------------------------------------------------------------
void SetGPUByCudaIndex(const int& cudaIdx);

//------------------------------------------------------------
// Simplest implementation of the persistent thread method.
// Given a kernel (__global__ function), the optimal
// numbers of blocks per grid and threads per block are determined
// such that the total number of threads precisely saturate
// the GPU device (i.e. reaching the maximum theoretical occupancy).
//------------------------------------------------------------
template <typename F>
void ImplementPersistentThread(F f,
                               int& blockPerGrid,
                               int& threadPerBlock,
                               size_t dynamicSMemSize = 0,
                               int blockSizeLimit = 64);

} // end namespace Peanut

#include "utility/GPU.tpp"

#endif
