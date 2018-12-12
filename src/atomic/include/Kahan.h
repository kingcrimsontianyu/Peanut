#if !defined(_KAHAN_H_)
#define _KAHAN_H_

namespace Peanut
{
//------------------------------------------------------------
//------------------------------------------------------------
__device__ __forceinline__ float2 atomicAddKahanUnoptimized(float2* address, float val)
{
    union HelperUnionFP32
    {
        float2 data;
        unsigned long long int helper;
    };

    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    HelperUnionFP32 old, assumed, temp;

    old.helper = *address_as_ull;

    do
    {
        assumed = old;
        temp = assumed;

        float y = val - temp.data.y;
        float t = temp.data.x + y; // accumulate high-order part
        temp.data.y = (t - temp.data.x) - y; // recover lower-order part
        temp.data.x = t;

        old.helper = atomicCAS(address_as_ull, assumed.helper, temp.helper);

        // increment counter
        // atomicAdd(&loopCount, 1ULL);

    }
    while(assumed.helper != old.helper);

    return old.data;
}

//------------------------------------------------------------
//------------------------------------------------------------
__device__ __forceinline__ float2 atomicAddKahanOptimized(float2* address, float val)
{
    union HelperUnionFP32
    {
        float2* address;
        double helper;
    };

    union HelperUnion3FP32
    {
        float2 data;
        unsigned long long int helper;
    };

    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // step 1: find peers, i.e. threads with the same key
    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    unsigned int lane;
    asm volatile("mov.u32 %0, %laneid;" : "=r"(lane));

    unsigned int peers = 0;
    bool is_peer;

    // in the beginning, all lanes are available
    unsigned unclaimed = __activemask();

    do
    {
        int src = __ffs(unclaimed) - 1;






        // int lo, hi;
        // HelperUnionFP32 addressUnion = {address};
        // asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(addressUnion.helper));

        // lo = __shfl_sync(__activemask(), lo, src);
        // hi = __shfl_sync(__activemask(), hi, src);

        // HelperUnionFP32 tempUnion;
        // asm volatile("mov.b64 %0, {%1,%2};" : "=d"(tempUnion.helper) : "r"(lo), "r"(hi));

        HelperUnionFP32 addressUnion = {address};
        HelperUnionFP32 tempUnion;
        tempUnion.helper = __shfl_sync(__activemask(), addressUnion.helper, src);








        // fetch key of first unclaimed lane and compare with this key
        is_peer = (addressUnion.address == tempUnion.address);

        // determine which lanes have a match
        peers = __ballot_sync(__activemask(), is_peer);

        // remove lanes with matching keys from the pool
        unclaimed ^= peers;
    }
    // quit if we have a match
    while(!is_peer);

    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // step 2: reduce peers. results are stored in the lowest peers
    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // find the peer with lowest lane index
    int first = __ffs(peers)-1;

    // calculate own relative position among peers
    int rel_pos = __popc(peers << (32 - lane));

    // ignore peers with lower (or same) lane index
    peers &= (0xfffffffe << lane);

    while(__any_sync(__activemask(), peers))
    {
        // find next-highest remaining peer
        int next = __ffs(peers);

        // __shfl_sync() only works if both threads participate, so we always do.
        float temp = __shfl_sync(__activemask(), val, next - 1);

        // only add if there was anything to add
        if(next)
        {
            val += temp;
        }

        // all lanes with their least significant index bit set are done
        unsigned int done = rel_pos & 1;

        // remove all peers that are already done
        peers &= ~__ballot_sync(__activemask(), done);

        // abuse relative position as iteration counter
        rel_pos >>= 1;
    }

    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // step 3: Nvidia's compare-and-swap algorithm
    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // only peers with the lowest index perform atomic operation
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    HelperUnion3FP32 old, assumed, temp;
    if(lane == first)
    {
        old.helper = *address_as_ull;

        do
        {
            assumed = old;
            temp = assumed;

            float y = val - temp.data.y;
            float t = temp.data.x + y; // accumulate high-order part
            temp.data.y = (t - temp.data.x) - y; // recover lower-order part
            temp.data.x = t;

            old.helper = atomicCAS(address_as_ull, assumed.helper, temp.helper);

            // increment counter
            // atomicAdd(&loopCount, 1ULL);

        }
        while(assumed.helper != old.helper);
    }

    return old.data;
}

} // end namespace Peanut

#endif

