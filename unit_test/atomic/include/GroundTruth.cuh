#ifndef _GROUND_TRUTH_CUH_
#define _GROUND_TRUTH_CUH_

#include "include/Test.h"
#include "random_number/include/Lcg.cuh"
#include "utility/include/GPU.h"
#include "utility/include/LogBook.h"
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <iomanip>

namespace Peanut
{
//------------------------------------------------------------
//------------------------------------------------------------
__global__ void GroundTruth_device(Parameter parameter, double* tallyList_d)
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
        int tallyOffset = tallyIdx * parameter.numThreadGPU;

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
                    dose = 1.0;
                }
                else
                {
                    dose = parameter.doseIncrement * rng.GenerateRN();
                }
                tallyList_d[tallyOffset + globalThreadIdx] += dose;
            }
        }
    }
}

//------------------------------------------------------------
//------------------------------------------------------------
class GroundTruth : public Test
{
public:
    GroundTruth(const Parameter* parameter_ext);
    void Run() override;
    void ShowResult() override;
    double GetResult();

private:
    double* tallyList_d;
    size_t numElement;
};

//------------------------------------------------------------
//------------------------------------------------------------
GroundTruth::GroundTruth(const Parameter* parameter_ext) :
Test(parameter_ext)
{}

//------------------------------------------------------------
//------------------------------------------------------------
void GroundTruth::Run()
{
    numElement = parameter->numTally * parameter->numThreadGPU;
    HANDLE_GPU_ERROR(cudaMalloc(&tallyList_d, numElement * sizeof(double)));

    timer.StartOrResume("ground truth");
    GroundTruth_device<<<parameter->blockPerGrid, parameter->threadPerBlock>>>(*parameter, tallyList_d);
    HANDLE_GPU_ERROR(cudaDeviceSynchronize());
    timer.Stop("ground truth");

    thrust::device_ptr<double> dp(tallyList_d);
    sum = thrust::reduce(dp, dp + numElement);

    HANDLE_GPU_ERROR(cudaFree(tallyList_d));
}

//------------------------------------------------------------
//------------------------------------------------------------
void GroundTruth::ShowResult()
{
    auto& logBook = LogBook::GetLogBook();
    logBook << "--> Ground truth\n"
            << "    Time = "  << std::setprecision(3)  << timer.GetElapsedTimeInSecondByTag("ground truth") << "\n"
            << "    Value = " << std::setprecision(16) << sum << std::endl;

}

//------------------------------------------------------------
//------------------------------------------------------------
double GroundTruth::GetResult()
{
    return sum;
}

} // end namespace Peanut

#endif

