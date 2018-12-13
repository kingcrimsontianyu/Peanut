#include "include/TestManager.h"
#include "utility/include/LogBook.h"
#include "utility/include/Exception.h"
#include <iostream>

//------------------------------------------------------------
//------------------------------------------------------------
//------------------------------------------------------------
//------------------------------------------------------------
int main(int argc, char* argv[])
{
    try
    {
        auto& logBook = Peanut::LogBook::GetLogBook();

        try
        {
            Peanut::TestManager tm(argc, argv);
            tm.Run();
        }
        catch(Peanut::PeanutException& e)
        {
            logBook << e.what();
        }
        catch(...)
        {
            logBook << "--> Unhandled exception." << std::endl;
        }
    }
    catch(...)
    {
        std::cout << "--> Logbook exception." << std::endl;
    }

    return 0;
}