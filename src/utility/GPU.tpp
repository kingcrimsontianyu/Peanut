
namespace Peanut
{
//------------------------------------------------------------
//------------------------------------------------------------
template <typename F>
void ImplementPersistentThread(F f,
                               int& blockPerGrid,
                               int& threadPerBlock,
                               size_t dynamicSMemSize,
                               int blockSizeLimit)
{
    HANDLE_GPU_ERROR(cudaOccupancyMaxPotentialBlockSize(&blockPerGrid,
                                                        &threadPerBlock,
                                                        f,
                                                        dynamicSMemSize,
                                                        blockSizeLimit));

}

} // end namespace Peanut

