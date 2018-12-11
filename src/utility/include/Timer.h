#if !defined(_TIME_H_)
#define _TIME_H_

#include <chrono>
#include <vector>
#include <string>

namespace Peanut
{
class TimeData;

//------------------------------------------------------------
//------------------------------------------------------------
class Timer
{
public:
    void StartOrResume(const std::string& tag);

    void Stop(const std::string& tag);

    double GetElapsedTimeInSecondByTag(const std::string& tag);

    void ShowAll();
private:
    std::vector<TimeData> timeDataList;
};


//------------------------------------------------------------
//------------------------------------------------------------
class TimeData
{
public:
    std::string tag;
    std::chrono::time_point<std::chrono::system_clock> start;
    std::chrono::time_point<std::chrono::system_clock> end;
    std::chrono::duration<double> elapsedTimeInSecond;
};

} // end namespace Peanut

#endif

