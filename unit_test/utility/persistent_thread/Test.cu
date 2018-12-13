#include "utility/include/LogBook.h"
#include "utility/include/Exception.h"
#include "utility/include/GPU.h"
#include <iostream>
#include <vector>

//------------------------------------------------------------
// This kernel is NOT optimized for memory bandwidth.
// The total computation tasks are distributed among persistent
// threads evenly and statically.
//------------------------------------------------------------
__global__ void Kernel(double* a, double* b, double* c, size_t numElement, int threadPerGrid)
{
    size_t globalThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if(globalThreadIdx < threadPerGrid)
    {
        size_t quotient = numElement / threadPerGrid;
        size_t modulus  = numElement % threadPerGrid;
        size_t numTaskPerThread;
        size_t taskOffset;

        if(globalThreadIdx < modulus)
        {
            numTaskPerThread = quotient + 1;
            taskOffset = (quotient + 1) * globalThreadIdx;
        }
        else
        {
            numTaskPerThread = quotient;
            taskOffset = (quotient + 1) * modulus + quotient * (globalThreadIdx - modulus);
        }

        for(size_t i = 0; i < numTaskPerThread; ++i)
        {
            size_t idx = taskOffset + i;
            c[idx] = a[idx] + b[idx];
        }
    }
}

//------------------------------------------------------------
//------------------------------------------------------------
class TestManager
{
public:
    void Run();

private:
    int blockPerGrid;
    int threadPerBlock;
    int threadPerGrid;

    std::vector<double> a;
    std::vector<double> b;
    std::vector<double> c;

    double* a_d = nullptr;
    double* b_d = nullptr;
    double* c_d = nullptr;
};

//------------------------------------------------------------
//------------------------------------------------------------
void TestManager::Run()
{
    auto& logBook = Peanut::LogBook::GetLogBook();


    size_t numElement = 1024 * 1024;

    a.resize(numElement, 0.0);
    b.resize(numElement, 0.0);
    c.resize(numElement, 0.0);

    for(size_t i = 0; i < numElement; ++i)
    {
        a[i] = 0.25;
        b[i] = 0.75;
    }

    Peanut::SetGPUByCudaIndex(0);

    Peanut::ImplementPersistentThread(Kernel,
                                      blockPerGrid,
                                      threadPerBlock);
    threadPerGrid = blockPerGrid * threadPerBlock;
    logBook << "--> blockPerGrid = " << blockPerGrid << "\n"
            << "    threadPerBlock = " << threadPerBlock << "\n"
            << "    threadPerGrid = " << threadPerGrid << "\n";

    HANDLE_GPU_ERROR(cudaMalloc(&a_d, numElement * sizeof(double)));
    HANDLE_GPU_ERROR(cudaMalloc(&b_d, numElement * sizeof(double)));
    HANDLE_GPU_ERROR(cudaMalloc(&c_d, numElement * sizeof(double)));

    HANDLE_GPU_ERROR(cudaMemcpy(a_d, a.data(), numElement * sizeof(double), cudaMemcpyHostToDevice));
    HANDLE_GPU_ERROR(cudaMemcpy(b_d, b.data(), numElement * sizeof(double), cudaMemcpyHostToDevice));

    Kernel<<<blockPerGrid, threadPerBlock>>>(a_d, b_d, c_d, numElement, threadPerGrid);
    HANDLE_GPU_ERROR(cudaDeviceSynchronize());

    HANDLE_GPU_ERROR(cudaMemcpy(c.data(), c_d, numElement * sizeof(double), cudaMemcpyDeviceToHost));

    HANDLE_GPU_ERROR(cudaFree(a_d));
    HANDLE_GPU_ERROR(cudaFree(b_d));
    HANDLE_GPU_ERROR(cudaFree(c_d));

    double sum = 0.0;
    for(auto&& item : c)
    {
        sum += item;
    }

    logBook << "--> result = " << sum << std::endl;
}

//------------------------------------------------------------
//------------------------------------------------------------
int main()
{
    try
    {
        auto& logBook = Peanut::LogBook::GetLogBook();

        try
        {
            TestManager tm;
            tm.Run();
        }
        catch(Peanut::PeanutException& e)
        {
            logBook << e.what();
        }
        catch(...)
        {
            logBook << "--> Unhandled exception." << std::endl;
        }
    }
    catch(...)
    {
        std::cout << "--> Logbook exception." << std::endl;
    }

    return 0;
}

