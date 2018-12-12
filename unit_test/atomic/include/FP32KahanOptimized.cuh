#ifndef _FP32_KAHAN_OPTIMIZED_CUH_
#define _FP32_KAHAN_OPTIMIZED_CUH_

// This method implements the well-known Kahan compensated summation algorithm
// to solve the roundoff error problem for single-precision atomicAdd().
// The method is based on a combination of CAS and warp-aggregated reduction by key methods.

#include "include/GroundTruth.cuh"
#include "atomic/include/PeanutAtomic.h"
#include <vector>

namespace Peanut
{
//------------------------------------------------------------
//------------------------------------------------------------
__global__ void FP32KahanOptimized_device(Parameter parameter, float2* tallyList_d)
{
    int globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(globalThreadIdx < parameter.numThreadGPU)
    {
        // initialize thread
        Lcg_device rng;

        unsigned long long quotient = parameter.numHistory / parameter.numThreadGPU;
        unsigned long long modulus  = parameter.numHistory % parameter.numThreadGPU;
        unsigned long long numHistoryPerThread;
        unsigned long long historyOffsetThread;

        if(globalThreadIdx < modulus)
        {
            numHistoryPerThread = quotient + 1;
            historyOffsetThread = (quotient + 1) * globalThreadIdx;
        }
        else
        {
            numHistoryPerThread = quotient;
            historyOffsetThread = (quotient + 1) * modulus + quotient * (globalThreadIdx - modulus);
        }

        int tallyIdx = globalThreadIdx % parameter.numTally;

        for(int k = 0; k < numHistoryPerThread; ++k)
        {
            // initialize particle
            unsigned long long historyOffsetParticle = historyOffsetThread + k;
            rng.Initialize(historyOffsetParticle);

            for(int i = 0; i < parameter.numCollision; ++i)
            {
                float dose;
                if(parameter.verify)
                {
                    dose = 1.0f;
                }
                else
                {
                    dose = parameter.doseIncrement * rng.GenerateRN();
                }

                atomicAddKahanOptimized(&tallyList_d[tallyIdx], dose);
            }
        }
    }
}

//------------------------------------------------------------
//------------------------------------------------------------
class FP32KahanOptimized : public Test
{
public:
    FP32KahanOptimized(const Parameter* parameter_ext, GroundTruth* groundTruth_ext);
    void Run() override;
    void ShowResult() override;

private:
    std::vector<float2> tallyList_h;
    float2* tallyList_d;
    size_t numElement;
    GroundTruth* groundTruth;
};

//------------------------------------------------------------
//------------------------------------------------------------
FP32KahanOptimized::FP32KahanOptimized(const Parameter* parameter_ext, GroundTruth* groundTruth_ext) :
Test(parameter_ext),
groundTruth(groundTruth_ext)
{}

//------------------------------------------------------------
//------------------------------------------------------------
void FP32KahanOptimized::Run()
{
    tallyList_h.resize(parameter->numTally);
    HANDLE_GPU_ERROR(cudaMalloc(&tallyList_d, parameter->numTally * sizeof(float2)));

    timer.StartOrResume("FP32 Kahan optimized");
    FP32KahanOptimized_device<<<parameter->blockPerGrid, parameter->threadPerBlock>>>(*parameter, tallyList_d);
    HANDLE_GPU_ERROR(cudaDeviceSynchronize());
    timer.Stop("FP32 Kahan optimized");

    HANDLE_GPU_ERROR(cudaMemcpy(tallyList_h.data(), tallyList_d, parameter->numTally * sizeof(float2), cudaMemcpyDeviceToHost));

    sum = 0.0;
    for(size_t i = 0; i < tallyList_h.size(); ++i)
    {
        sum += static_cast<double>(tallyList_h[i].x) + static_cast<double>(tallyList_h[i].y);
    }

    HANDLE_GPU_ERROR(cudaFree(tallyList_d));
}

//------------------------------------------------------------
//------------------------------------------------------------
void FP32KahanOptimized::ShowResult()
{
    auto& logBook = LogBook::GetLogBook();
    logBook << "--> FP32 Kahan optimized\n"
            << "    Time = "  << std::setprecision(5)  << timer.GetElapsedTimeInSecondByTag("FP32 Kahan optimized") << "\n"
            << "    Value = " << std::setprecision(16) << sum << "\n"
            << "    Diff [%] = "  << std::setprecision(16) << (sum - groundTruth->GetResult()) / groundTruth->GetResult() * 100.0 << std::endl;
}

} // end namespace Peanut

#endif








