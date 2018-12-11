#include <thread>

#include "utility/include/LogBook.h"
#include "utility/include/Timer.h"
#include "random_number/include/Lcg.h"
#include "Test.h"
#include "utility/include/Exception.h"

void Test();

//------------------------------------------------------------
//------------------------------------------------------------
int main()
{
    auto& logBook = Peanut::LogBook::GetLogBook();

    try
    {
        Test();
    }
    catch(Peanut::PeanutException& e)
    {
        logBook << e.what();
    }
    catch(...)
    {
        logBook << "--> Unhandled exception." << std::endl;
    }

    return 0;
}

//------------------------------------------------------------
//------------------------------------------------------------
void Test()
{
    // Peanut::Timer t;

    // t.StartOrResume("total");

    // t.StartOrResume("1");
    // {
        // std::chrono::seconds dura(1);
        // std::this_thread::sleep_for(dura);
    // }
    // t.Stop("1");

    // t.StartOrResume("2");
    // {
        // std::chrono::seconds dura(2);
        // std::this_thread::sleep_for(dura);
    // }
    // t.Stop("2");

    // {
        // std::chrono::seconds dura(1);
        // std::this_thread::sleep_for(dura);
    // }

    // t.Stop("total");

    // t.ShowAll();

    TestManager tm;
    tm.BiuCPU();
    tm.BiuGPU();
}

//------------------------------------------------------------
//------------------------------------------------------------
void TestManager::BiuCPU()
{
    Peanut::Lcg lcg;
    lcg.Initialize(1000ULL);

    double sum = 0.0;
    for(int i = 0; i < 100; ++i)
    {
        sum += lcg.GenerateRN();
    }

    auto& logBook = Peanut::LogBook::GetLogBook();
    logBook << "CPU sum = " << sum << std::endl;
}



