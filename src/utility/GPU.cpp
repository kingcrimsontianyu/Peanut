#include "utility/include/GPU.h"
#include "utility/include/LogBook.h"
#include <iomanip>

namespace Peanut
{
//------------------------------------------------------------
//------------------------------------------------------------
void QueryGPU()
{
    int GPUCount = 0;
    cudaDeviceProp GPUProperty;
    HANDLE_GPU_ERROR(cudaGetDeviceCount(&GPUCount));

    auto& logBook = LogBook::GetLogBook();

    logBook << "--> All available GPUs"<< std::endl;
    logBook << std::setw(9)  << "GPU id" << "|"
            << std::setw(24) << "name" << "|"
            << std::setw(8)  << "com cap" << "|"
            << std::setw(6)  << "ECC on" << "|"
            << std::setw(6)  << "SM #" << "|"
            << std::setw(15) << "freq [MHz]" << "|"
            << std::setw(15) << "gmem [GB]" << "|"
            << std::endl;

    for(int i = 0; i < GPUCount; ++i)
    {
        HANDLE_GPU_ERROR(cudaSetDevice(i));
        HANDLE_GPU_ERROR(cudaGetDeviceProperties(&GPUProperty, i));

        logBook << std::setw(9)  << i << "|"
                << std::setw(24) << GPUProperty.name << "|"
                << std::setw(6)  << GPUProperty.major << "." << GPUProperty.minor << "|"
                << std::setw(6)  << GPUProperty.ECCEnabled << "|"
                << std::setw(6)  << GPUProperty.multiProcessorCount << "|"
                << std::setw(15) << GPUProperty.clockRate / 1000.0 << "|"
                << std::setw(15) << GPUProperty.totalGlobalMem / 1024.0 / 1024.0 / 1024.0 << "|"
                << std::endl;
    }
}

//------------------------------------------------------------
//------------------------------------------------------------
void SetGPUByName(const std::string& name)
{
    int GPUCount = 0;
    cudaDeviceProp GPUProperty;
    HANDLE_GPU_ERROR(cudaGetDeviceCount(&GPUCount));

    auto& logBook = LogBook::GetLogBook();

    for(int i = 0; i < GPUCount; ++i)
    {
        HANDLE_GPU_ERROR(cudaSetDevice(i));
        HANDLE_GPU_ERROR(cudaGetDeviceProperties(&GPUProperty, i));

        std::string GPUName(GPUProperty.name);
        std::size_t found = GPUName.find(name);
        if(found != std::string::npos)
        {
            logBook << "--> GPU " << name << " has been selected."<< std::endl;
            return;
        }
    }

    std::string message = "Named GPU " + name + " is not found.\n";
    throw PeanutException(message);
}

//------------------------------------------------------------
//------------------------------------------------------------
void SetGPUByCudaIndex(const int& cudaIdx)
{
    auto& logBook = LogBook::GetLogBook();

    HANDLE_GPU_ERROR(cudaSetDevice(cudaIdx));
    logBook << "--> GPU " << cudaIdx << " has been selected."<< std::endl;
}

} // end namespace Peanut


