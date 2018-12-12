#if !defined(_CAS_H_)
#define _CAS_H_

namespace Peanut
{
//------------------------------------------------------------
//------------------------------------------------------------
__device__ __forceinline__ double atomicAddFP64CAS(double* address, double val)
{
    // nvidia's code
    // one way of type punning
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do
    {
        assumed = old;

        // atomicCAS(address, compare, val):
        // reads the 32-bit or 64-bit word old located at the address address in global
        // or shared memory, computes (old == compare ? val : old) , and stores the
        // result back to memory at the same address. These three operations are
        // performed in one atomic transaction. The function returns old (Compare And Swap).
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    }
    // keep trying until the address's value happens to be not changed by other threads during
    // the calculation
    // note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    while (assumed != old);
    return __longlong_as_double(old);
}

} // end namespace Peanut

#endif
