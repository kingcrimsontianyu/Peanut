#include "Test.h"
#include "utility/include/GPU.h"
#include "utility/include/LogBook.h"
#include "random_number/include/Lcg.cuh"

//------------------------------------------------------------
//------------------------------------------------------------
__global__ void BiuGPU_kernel()
{
    Peanut::Lcg_device lcg;
    lcg.Initialize(1000ULL);

    double sum = 0.0;
    for(int i = 0; i < 100; ++i)
    {
        sum += lcg.GenerateRN();
    }

    printf("GPU sum = %f\n", sum);
}

//------------------------------------------------------------
//------------------------------------------------------------
void TestManager::BiuGPU()
{
    HANDLE_GPU_ERROR(cudaSetDevice(0));

    BiuGPU_kernel<<<1, 1>>>();

    HANDLE_GPU_ERROR(cudaDeviceSynchronize());

    HANDLE_GPU_ERROR(cudaDeviceReset());
}