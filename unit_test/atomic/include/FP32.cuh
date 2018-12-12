#ifndef _FP32_CUH_
#define _FP32_CUH_

// This method simply uses single-precision atomicAdd().
// Large roundoff errors in tallies may be observed.

#include "include/GroundTruth.cuh"
#include <vector>

namespace Peanut
{
//------------------------------------------------------------
//------------------------------------------------------------
__global__ void FP32_device(Parameter parameter, float* tallyList_d)
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
                atomicAdd(&tallyList_d[tallyIdx], dose);
            }
        }
    }
}

//------------------------------------------------------------
//------------------------------------------------------------
class FP32 : public Test
{
public:
    FP32(const Parameter* parameter_ext, GroundTruth* groundTruth_ext);
    void Run() override;
    void ShowResult() override;

private:
    std::vector<float> tallyList_h;
    float* tallyList_d;
    size_t numElement;
    GroundTruth* groundTruth;
};

//------------------------------------------------------------
//------------------------------------------------------------
FP32::FP32(const Parameter* parameter_ext, GroundTruth* groundTruth_ext) :
Test(parameter_ext),
groundTruth(groundTruth_ext)
{}

//------------------------------------------------------------
//------------------------------------------------------------
void FP32::Run()
{
    tallyList_h.resize(parameter->numTally);
    HANDLE_GPU_ERROR(cudaMalloc(&tallyList_d, parameter->numTally * sizeof(float)));

    timer.StartOrResume("FP32");
    FP32_device<<<parameter->blockPerGrid, parameter->threadPerBlock>>>(*parameter, tallyList_d);
    HANDLE_GPU_ERROR(cudaDeviceSynchronize());
    timer.Stop("FP32");

    HANDLE_GPU_ERROR(cudaMemcpy(tallyList_h.data(), tallyList_d, parameter->numTally * sizeof(float), cudaMemcpyDeviceToHost));

    sum = 0.0;
    for(size_t i = 0; i < tallyList_h.size(); ++i)
    {
        sum += static_cast<double>(tallyList_h[i]);
    }

    HANDLE_GPU_ERROR(cudaFree(tallyList_d));
}

//------------------------------------------------------------
//------------------------------------------------------------
void FP32::ShowResult()
{
    auto& logBook = LogBook::GetLogBook();
    logBook << "--> FP32\n"
            << "    Time = "  << std::setprecision(5)  << timer.GetElapsedTimeInSecondByTag("FP32") << "\n"
            << "    Value = " << std::setprecision(16) << sum << "\n"
            << "    Diff [%] = "  << std::setprecision(16) << (sum - groundTruth->GetResult()) / groundTruth->GetResult() * 100.0 << std::endl;
}

} // end namespace Peanut

#endif




