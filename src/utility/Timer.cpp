#include "utility/include/Timer.h"
#include "utility/include/LogBook.h"
#include "utility/include/Exception.h"

namespace Peanut
{
//------------------------------------------------------------
//------------------------------------------------------------
void Timer::StartOrResume(const std::string& tag)
{
    // resume
    for(auto&& item : timeDataList)
    {
        if(tag == item.tag)
        {
            item.start = std::chrono::system_clock::now();
            return;
        }
    }

    // start
    TimeData timeData;
    timeData.tag = tag;
    timeData.start = std::chrono::system_clock::now();
    timeData.elapsedTimeInSecond = std::chrono::duration<double>::zero();
    timeDataList.push_back(std::move(timeData));
}

//------------------------------------------------------------
//------------------------------------------------------------
void Timer::Stop(const std::string& tag)
{
    for(auto&& item: timeDataList)
    {
        if(tag == item.tag)
        {
            item.end = std::chrono::system_clock::now();
            item.elapsedTimeInSecond += item.end - item.start;
            return;
        }
    }

    throw PeanutException("--> Time error: tag " + tag +" cannot be found.");
}

//------------------------------------------------------------
//------------------------------------------------------------
double Timer::GetElapsedTimeInSecondByTag(const std::string& tag)
{
    for(auto const& item: timeDataList)
    {
        if(tag == item.tag)
        {
            return item.elapsedTimeInSecond.count();
        }
    }

    throw PeanutException("--> Time error: tag " + tag +" cannot be found.");
}

//------------------------------------------------------------
//------------------------------------------------------------
void Timer::ShowAll()
{
    auto& logBook = LogBook::GetLogBook();
    logBook << "--> Time [sec]" << std::endl;

    for(auto&& item : timeDataList)
    {
        logBook << "    " << item.tag << " = " << item.elapsedTimeInSecond.count() << std::endl;
    }
}

} // end namespace Peanut
