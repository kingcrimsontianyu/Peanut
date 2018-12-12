#if !defined(_PARAMETER_H_)
#define _PARAMETER_H_

namespace Peanut
{
//------------------------------------------------------------
//------------------------------------------------------------
class Parameter
{
public:
    int numCollision;
    unsigned long long int numHistory;
    int blockPerGrid;
    int threadPerBlock;
    int numThreadGPU;
    int numTally;
    bool skipSlow;
    bool hbVSwib;
    bool verify;
    double doseIncrement;
};

} // end namespace Peanut


#endif //_PARAMETER_H_

