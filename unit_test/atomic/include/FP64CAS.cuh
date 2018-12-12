#ifndef _FP64CAS_CUH_
#define _FP64CAS_CUH_

// This method uses Nvidia implementation of CAS (compare-and-swap).

#include "include/GroundTruth.cuh"
#include <vector>

namespace Peanut
{
//------------------------------------------------------------
//------------------------------------------------------------
__global__ void FP64CAS_device(Parameter parameter, double* tallyList_d)
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
                double dose;
                if(parameter.verify)
                {
                    dose = 1.0f;
                }
                else
                {
                    dose = parameter.doseIncrement * rng.GenerateRN();
                }

                atomicAddFP64CAS(&tallyList_d[tallyIdx], dose);
            }
        }
    }
}

//------------------------------------------------------------
//------------------------------------------------------------
class FP64CAS : public Test
{
public:
    FP64CAS(const Parameter* parameter_ext, GroundTruth* groundTruth_ext);
    void Run() override;
    void ShowResult() override;

private:
    std::vector<double> tallyList_h;
    double* tallyList_d;
    size_t numElement;
    GroundTruth* groundTruth;
};

//------------------------------------------------------------
//------------------------------------------------------------
FP64CAS::FP64CAS(const Parameter* parameter_ext, GroundTruth* groundTruth_ext) :
Test(parameter_ext),
groundTruth(groundTruth_ext)
{}

//------------------------------------------------------------
//------------------------------------------------------------
void FP64CAS::Run()
{
    tallyList_h.resize(parameter->numTally);
    HANDLE_GPU_ERROR(cudaMalloc(&tallyList_d, parameter->numTally * sizeof(double)));

    timer.StartOrResume("FP64 CAS");
    FP64CAS_device<<<parameter->blockPerGrid, parameter->threadPerBlock>>>(*parameter, tallyList_d);
    HANDLE_GPU_ERROR(cudaDeviceSynchronize());
    timer.Stop("FP64 CAS");

    HANDLE_GPU_ERROR(cudaMemcpy(tallyList_h.data(), tallyList_d, parameter->numTally * sizeof(double), cudaMemcpyDeviceToHost));

    sum = 0.0;
    for(size_t i = 0; i < tallyList_h.size(); ++i)
    {
        sum += static_cast<double>(tallyList_h[i]);
    }

    HANDLE_GPU_ERROR(cudaFree(tallyList_d));
}

//------------------------------------------------------------
//------------------------------------------------------------
void FP64CAS::ShowResult()
{
    auto& logBook = LogBook::GetLogBook();
    logBook << "--> FP64 CAS\n"
            << "    Time = "  << std::setprecision(5)  << timer.GetElapsedTimeInSecondByTag("FP64 CAS") << "\n"
            << "    Value = " << std::setprecision(16) << sum << "\n"
            << "    Diff [%] = "  << std::setprecision(16) << (sum - groundTruth->GetResult()) / groundTruth->GetResult() * 100.0 << std::endl;
}

} // end namespace Peanut

#endif




