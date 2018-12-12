#if !defined(_TEST_MANAGER_H_)
#define _TEST_MANAGER_H_

#include <map>
#include <string>
#include "include/Parameter.h"

namespace Peanut
{
//------------------------------------------------------------
//------------------------------------------------------------
class TestManager
{
public:
    TestManager(int argc, char* argv[]);
    void Run();
private:
    void SetDefaultParameter();
    void ParseCMD(int argc, char* argv[]);

    Parameter parameter;
    std::string gpuName;
    std::map<std::string, std::string> parameterMap;
};


} // end namespace Peanut


#endif //_TEST_MANAGER_H_

