#if !defined(_TEST_H_)
#define _TEST_H_

#include <map>
#include <string>
#include "include/Parameter.h"
#include "utility/include/Timer.h"

namespace Peanut
{
//------------------------------------------------------------
//------------------------------------------------------------
class Test
{
public:
    Test(const Parameter* parameter_ext);
    virtual void Run() = 0;
    virtual void ShowResult() = 0;

protected:
    const Parameter* parameter;
    double doseIncrementDouble;
    unsigned long long int loopCount;
    double sum;
    Timer timer;
};

Test::Test(const Parameter* parameter_ext) :
parameter(parameter_ext)
{}

} // end namespace Peanut


#endif //_TEST_H_

