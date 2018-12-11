#include "utility/include/LogBook.h"
#include "utility/include/Timer.h"
#include <thread>


int main()
{
    Peanut::LogBook::GetLogBook();

    Peanut::Timer t;

    t.StartOrResume("total");

    t.StartOrResume("1");
    {
        std::chrono::seconds dura(1);
        std::this_thread::sleep_for(dura);
    }
    t.Stop("1");

    t.StartOrResume("2");
    {
        std::chrono::seconds dura(2);
        std::this_thread::sleep_for(dura);
    }
    t.Stop("2");

    {
        std::chrono::seconds dura(1);
        std::this_thread::sleep_for(dura);
    }

    t.Stop("total");

    t.ShowAll();

    return 0;
}