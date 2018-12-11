// This linear multiplicative congruential generator (LCG) is a slightly modified
// version of the one used in MCNP developed by Brown.
// Brown, F.B. and Nagaya, Y., 2002. The MCNP5 random number generator.
// Tech. Rep. LA-UR-02-3782, Los Alamos National Laboratory.
//
// 1. Each seed Sk (unsigned int64_t) corresponds to a rn (double)
// 2. To clarify, S(k+1) is derived from Sk, but it is not Sk + 1.
// 3. Each particle history has a reserved, fixed number (stride) of rns.
//    So (n+1)th particle starts with a rn that is skipped ahead stride times
//    (i.e. as if lcg_uniform is called stride times) than the rn used
//    at the start of nth particle.
// 4. The 1st particle history can start with a rn that is skipped ahead
//    some distance (as if it is the t-th particle). In this case,
//    for the i-th particle, the starting rn is skipped ahead by
//    (t + n) * stride. Note that the concept of ``stride'' is equivalent to
//    ``sequence'' in curand xorshift.
// 5. Here for lcg we use ``seed'' and ``state'' interchangeably. In curand
//    xorshift state is derived from seed but is a more complex data structure.
// 6. S(k + numToSkip) is the seed (or state) as if lcg_uniform_double() has been called
//    numToSkip times since S(k)

#if !defined(_LCG_H_)
#define _LCG_H_

#include <cstdint>

namespace Peanut
{
typedef uint64_t lcgState;

namespace CPU
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
inline void lcg_skip_ahead(uint64_t* seed, int64_t* numToSkip)
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
inline double lcg_uniform_double(lcgState* state)
{
    *state = (lcgMult * (*state) + lcgAdd) & lcgMask;

    return  (double) (*state * lcgNorm);
}

//------------------------------------------------------------
//------------------------------------------------------------
inline void lcg_init(uint64_t particleIdx,
                     uint64_t offset,
                     lcgState* state)
{
    *state = lcgSeed0;
    int64_t numToSkip = particleIdx * lcgStride + offset;
    lcg_skip_ahead(state, &numToSkip);
}

} // end namespace CPU





//------------------------------------------------------------
//------------------------------------------------------------
class Lcg
{
public:
    void Initialize(unsigned long long particleIdx);

    double GenerateRN();

private:
    lcgState m_state;
};

//------------------------------------------------------------
//------------------------------------------------------------
inline void Lcg::Initialize(unsigned long long particleIdx)
{
    CPU::lcg_init(particleIdx, 0ULL, &m_state);
}

//------------------------------------------------------------
//------------------------------------------------------------
inline double Lcg::GenerateRN()
{
    return CPU::lcg_uniform_double(&m_state);
}

} // end namespace Peanut
#endif //_LCG_H_
