#if !defined(_LCG_CUH_)
#define _LCG_CUH_

#if defined(__CUDACC__)

#include <cstdint>

namespace Peanut
{
typedef uint64_t lcgState_device;

namespace GPU
{
//------------------------------------------------------------
//------------------------------------------------------------
constexpr uint64_t  lcgMult    =  19073486328125ULL;     // g
constexpr uint64_t  lcgAdd     =  0ULL;                  // c
constexpr int       lcgBits    =  48;                    // number of bits, M
constexpr uint64_t  lcgStride  =  152917ULL;             // number of rns reserved for one history
constexpr uint64_t  lcgSeed0   =  19073486328125ULL;     // initial seed
constexpr uint64_t  lcgMod     =  281474976710656ULL;    // 2^M
constexpr uint64_t  lcgMask    =  281474976710655ULL;    // 2^M - 1
constexpr uint64_t  lcgPeriod  =  70368744177664ULL;     // period
constexpr double    lcgNorm    =  1.0 / 281474976710656.;  // 2^(-M)

//------------------------------------------------------------
//------------------------------------------------------------
__device__ void lcg_skip_ahead(uint64_t* seed, int64_t* numToSkip)
{
    //  skip ahead n rns:   RN_SEED*lcgMult^n mod lcgMod
    uint64_t localSeed  = *seed;
    int64_t  nskip = *numToSkip;

    while(nskip < 0)
    {
        nskip += lcgPeriod;  // add period till >0
    }

    nskip = nskip & lcgMask; // mod lcgMod

    int64_t gen = 1;
    int64_t g = lcgMult;
    int64_t inc = 0;
    int64_t c = lcgAdd;

    // get gen = lcgMult^n,  in log2(n) ops, not n ops !
    for(; nskip; nskip>>=1)
    {
        if( nskip & 1 )
        {
            gen =  gen * g      & lcgMask;
            inc = (inc * g + c) & lcgMask;
        }
        c  = (g * c + c) & lcgMask;
        g  = g * g   & lcgMask;
    }

    *seed = (gen * localSeed + inc) & lcgMask;
}

//------------------------------------------------------------
//------------------------------------------------------------
__device__ double lcg_uniform_double(lcgState_device* state)
{
    *state = (lcgMult * (*state) + lcgAdd) & lcgMask;

    return  (double) (*state * lcgNorm);
}

//------------------------------------------------------------
//------------------------------------------------------------
__device__ void lcg_init(uint64_t particleIdx,
                         uint64_t offset,
                         lcgState_device* state)
{
    *state = lcgSeed0;
    int64_t numToSkip = particleIdx * lcgStride + offset;
    lcg_skip_ahead(state, &numToSkip);
}

} // end namespace GPU






//------------------------------------------------------------
//------------------------------------------------------------
class Lcg_device
{
public:
    __device__ void Initialize(unsigned long long particleIdx);

    __device__ double GenerateRN();

private:
    lcgState_device m_state;
};

//------------------------------------------------------------
//------------------------------------------------------------
__device__ __forceinline__ void Lcg_device::Initialize(unsigned long long particleIdx)
{
    GPU::lcg_init(particleIdx, 0ULL, &m_state);
}

//------------------------------------------------------------
//------------------------------------------------------------
__device__ __forceinline__ double Lcg_device::GenerateRN()
{
    return GPU::lcg_uniform_double(&m_state);
}

} // end namespace Peanut

#endif // __CUDACC__
#endif //_LCG_CUH_

