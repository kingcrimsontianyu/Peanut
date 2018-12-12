#if !defined(_WAG_H_)
#define _WAG_H_

namespace Peanut
{
//------------------------------------------------------------
//------------------------------------------------------------
__device__ __forceinline__ double atomicAddFP64WAG(double* address, double val)
{
    // another way of type punning
    union HelperUnionFP64
    {
        double* address;
        double  helper;
    };

    union HelperUnion2FP64
    {
        double data;
        unsigned long long int helper;
    };

    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // step 1: find peers, i.e. threads that are going to update
    // the same address, which will cause CAS to be excruciatingly slow
    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    unsigned int lane;
     // %laneid is a predefined, read-only special register that returns
     // the thread's lane within the warp. The lane identifier ranges from zero to WARP_SZ-1
    asm volatile("mov.u32 %0, %laneid;" : "=r"(lane));

    unsigned int peers = 0;
    bool is_peer;

    // --> __activemask(): Returns a 32-bit integer mask of all currently active threads in the calling warp.
    //     The Nth bit is set if the Nth lane in the warp is active when __activemask()
    //     is called. Inactive threads are represented by 0 bits in the returned mask.
    //     Threads which have exited the program are always marked as inactive.
    //     Note that threads that are convergent at an __activemask() call are not
    //     guaranteed to be convergent at subsequent instructions unless those
    //     instructions are synchronizing warp-builtin functions.
    // --> __ballot(1) has been replaced by __activemask() since cuda 9.
    unsigned unclaimed = __activemask();

    do
    {
        // __ffs(): find the position of the least significant bit set to 1 in a 32 bit integer
        // return 0 if no bit is set
        // the position is 1-based rather than 0-based
        int src = __ffs(unclaimed) - 1;







        // in the past, cuda does not support 64-bit warp shuffle
        // so we break it into two 32-bit warp shuffles
        // unsigned int lo, hi;
        // HelperUnionFP64 addressUnion = {address};
        // // save the 64-bit pointer to 32-bit lo and hi
        // asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(addressUnion.helper));

        // // According to the programming guide:
        // // The __shfl_sync() intrinsics permit exchanging of a variable between threads
        // // within a warp without use of shared memory. The exchange occurs simultaneously
        // // for all active threads within the warp (and named in mask), moving 4 or 8 (??? o0)
        // // bytes of data per thread depending on the type.
        // // All __shfl_sync() intrinsics return the 4-byte word referenced by var from
        // // the source lane ID as an unsigned integer. If the source lane ID is out of range
        // // or the source thread has exited, the calling thread's own var is returned.
        // // __shfl_sync() returns the value of var held by the thread whose ID is given by srcLane.
        // lo = __shfl_sync(__activemask(), lo, src);
        // hi = __shfl_sync(__activemask(), hi, src);

        // // save the received 32-bit lo and hi to the 64-bit tempUnion
        // HelperUnionFP64 tempUnion;
        // asm volatile("mov.b64 %0, {%1,%2};" : "=d"(tempUnion.helper) : "r"(lo), "r"(hi));

        // since cuda 9, 64-bit warp shuffle is allowed
        HelperUnionFP64 addressUnion = {address};
        HelperUnionFP64 tempUnion;
        tempUnion.helper = __shfl_sync(__activemask(), addressUnion.helper, src);








        // check if the addresses from the source lane is the same with self's
        is_peer = (addressUnion.address == tempUnion.address);

        // determine which lanes have a match with source
        // essentially broadcast the results among active threads
        peers = __ballot_sync(__activemask(), is_peer);

        // remove lanes with the same address with source
        // using xor (bit set if x and y is different)
        unclaimed ^= peers;
    }
    // quit when the source lane has the same address to be updated with self's
    // in the special case where all lanes have their unique address, peer will just be self
    while(!is_peer);

    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // step 2: reduce peers. results are stored in the lowest peers
    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // find the peer with lowest lane index
    int first = __ffs(peers) - 1;

    // calculate relative index among peers
    // __popc(): count the number of bits that are set to 1 in a 32 bit integer
    int rel_pos = __popc(peers << (32 - lane));

    // ignore peers with lower (or same) lane index
    peers &= (0xfffffffe << lane);

    while(__any_sync(__activemask(), peers))
    {
        // find next-highest remaining peer
        int next = __ffs(peers);







        // // Threads may only read data from another thread which is actively participating in the __shfl_sync() command.
        // // If the target thread is inactive, the retrieved value is undefined.
        // // This is why the code below is not placed in the if(next) block
        // unsigned lo, hi;
        // asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(val));

        // lo = __shfl_sync(__activemask(), lo, next - 1);
        // hi = __shfl_sync(__activemask(), hi, next - 1);

        // double temp;
        // asm volatile("mov.b64 %0, {%1,%2};" : "=d"(temp) : "r"(lo), "r"(hi));

        double temp = __shfl_sync(__activemask(), val, next - 1);






        // only add if next-highest remaining peer exists
        if(next)
        {
            val += temp;
        }

        // results in lanes with an odd relative index will be discarded
        // find if self has an odd index
        unsigned int done = rel_pos & 1;

        // remove all peers with an odd relative index
        peers &= ~__ballot_sync(__activemask(), done);

        // use relative index as iteration counter
        rel_pos >>= 1;
    }

    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // step 3: Nvidia's compare-and-swap algorithm
    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // only peers with the lowest index perform atomic operation
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    HelperUnion2FP64 old, assumed, temp;
    if(lane == first)
    {
        // union version
        old.helper = *address_as_ull;
        do
        {
            assumed = old;
            temp = assumed;
            temp.data = val + assumed.data;
            old.helper = atomicCAS(address_as_ull, assumed.helper, temp.helper);
        }
        while (assumed.helper != old.helper);
    }

    return old.data;
}

} // end namespace Peanut

#endif
