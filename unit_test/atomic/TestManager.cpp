#include "include/TestManager.h"
#include "utility/include/LogBook.h"
#include "utility/include/Timer.h"
#include "random_number/include/Lcg.h"
#include "utility/include/Exception.h"
#include "utility/include/GPU.h"
#include <sstream>

namespace Peanut
{
//------------------------------------------------------------
//------------------------------------------------------------
TestManager::TestManager(int argc, char* argv[])
{
    SetDefaultParameter();

    ParseCMD(argc, argv);

    auto& logBook = Peanut::LogBook::GetLogBook();

    logBook << "--> Parameters\n";
    for(auto it = parameterMap.begin(); it != parameterMap.end(); ++it)
    {
        logBook << "    " << it->first << " = "<< it->second << std::endl;
    }

    QueryGPU();

    SetGPUByName(gpuName);
}

//------------------------------------------------------------
//------------------------------------------------------------
void TestManager::ParseCMD(int argc, char* argv[])
{
    std::map<std::string, std::string> result;

    if(argc < 1)
    {
        throw PeanutException("No user-specified parameters found.");
    }

    // skip the first argument since it is the name of the executable itself
    for (int i = 1; i < argc; ++i)
    {
        // remove the leading "--" in --AAA=BBB
        std::string stringToRemove = "--";
        std::string line(argv[i]);
        size_t found = line.find(stringToRemove);

        // command-line parameters must be preceded with "--"
        if (found != 0)
        {
            throw PeanutException("Command-line parameters must be preceded with \"--\"");
        }
        else
        {
            line.erase(found, stringToRemove.length());
        }

        // separate the parameter name and value
        char delimiter = '=';
        found = line.find(delimiter); // line is AAA=BBB
        std::stringstream ss(line);
        std::string parameterKey;
        std::string parameterValue;

        if (found != std::string::npos) // if "=" exists
        {
            std::getline(ss, parameterKey, delimiter);
            std::getline(ss, parameterValue);
        }
        else
        {
            std::getline(ss, parameterKey);
            parameterValue = "true";
        }

        if(parameterMap.find(parameterKey) == parameterMap.end())
        {
            throw PeanutException("Unknown parameter: " + parameterKey +".");
        }
        else
        {
            parameterMap[parameterKey] = parameterValue;
        }
    }

    // set parameter
    parameter.numHistory = static_cast<unsigned long long>(std::stod(parameterMap["num-history"]));
    parameter.numCollision = std::stoi(parameterMap["num-collision"]);
    parameter.numTally = std::stoi(parameterMap["num-tally"]);
    gpuName = parameterMap["gpu-name"];
    parameter.skipSlow = parameterMap["skip-slow"] == "true" ? true : false;
    parameter.hbVSwib = parameterMap["hb-vs-wib"] == "true" ? true : false;
    parameter.verify = parameterMap["verify"] == "true" ? true : false;
    parameter.blockPerGrid = std::stoi(parameterMap["block-per-grid"]);
    parameter.threadPerBlock = std::stoi(parameterMap["thread-per-block"]);
    parameter.numThreadGPU = parameter.blockPerGrid * parameter.threadPerBlock;
    parameter.doseIncrement = std::stod(parameterMap["dose-increment"]);
}

//------------------------------------------------------------
//------------------------------------------------------------
void TestManager::SetDefaultParameter()
{
    parameterMap["num-history"     ]  = "1e3"    ;
    parameterMap["num-collision"   ]  = ""       ;
    parameterMap["num-tally"       ]  = ""       ;
    parameterMap["gpu-name"        ]  = ""       ;
    parameterMap["skip-slow"       ]  = "false"  ;
    parameterMap["hb-vs-wib"       ]  = "false"  ;
    parameterMap["block-per-grid"  ]  = "1024"   ;
    parameterMap["thread-per-block"]  = "64"     ;
    parameterMap["dose-increment"  ]  = "0.1"    ;
    parameterMap["verify"          ]  = "false"  ;
}

} // end namespace Peanut

