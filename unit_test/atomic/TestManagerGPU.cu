#include "include/TestManager.h"
#include "include/GroundTruth.cuh"
#include "include/FP32.cuh"
#include "include/FP32KahanUnoptimized.cuh"
#include "include/FP32KahanOptimized.cuh"
#include "include/FP64CAS.cuh"
#include "include/FP64WAG.cuh"
#include "include/FP64WIB.cuh"
#include "include/FP64.cuh"

namespace Peanut
{
//------------------------------------------------------------
//------------------------------------------------------------
void TestManager::Run()
{
    GroundTruth groundTruth(&parameter);
    FP32 fp32(&parameter, &groundTruth);
    FP32KahanUnoptimized fp32KahanUnoptimized(&parameter, &groundTruth);
    FP32KahanOptimized fp32KahanOptimized(&parameter, &groundTruth);
    FP64 fp64(&parameter, &groundTruth);
    FP64CAS fp64CAS(&parameter, &groundTruth);
    FP64WAG fp64WAG(&parameter, &groundTruth);
    FP64WIB fp64WIB(&parameter, &groundTruth);

    groundTruth.Run();
    groundTruth.ShowResult();

    if(parameter.hbVSwib)
    {
        fp64WIB.Run();
        fp64WIB.ShowResult();

        fp64.Run();
        fp64.ShowResult();
    }
    else
    {
        fp32.Run();
        fp32.ShowResult();

        if(!parameter.skipSlow)
        {
            fp32KahanUnoptimized.Run();
            fp32KahanUnoptimized.ShowResult();
        }

        fp32KahanOptimized.Run();
        fp32KahanOptimized.ShowResult();

        if(!parameter.skipSlow)
        {
            fp64CAS.Run();
            fp64CAS.ShowResult();
        }

        fp64WAG.Run();
        fp64WAG.ShowResult();

        fp64WIB.Run();
        fp64WIB.ShowResult();

        fp64.Run();
        fp64.ShowResult();
    }
}

} // end namespace Peanut
