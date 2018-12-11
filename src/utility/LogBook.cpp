#include "utility/include/LogBook.h"
#include <cstdio>
#include <iostream>

namespace Peanut
{
//------------------------------------------------------------
//------------------------------------------------------------
LogBook& LogBook::GetLogBook()
{
    static LogBook logBook;
    return logBook;
}

//------------------------------------------------------------
//------------------------------------------------------------
LogBook::LogBook()
{
    ofs.open("logbook.txt");
    std::cout << "--> Logbook opened." << std::endl;
}

//------------------------------------------------------------
//------------------------------------------------------------
LogBook::~LogBook()
{
    std::cout << "--> Logbook closed." << std::endl;
}

//------------------------------------------------------------
//------------------------------------------------------------
LogBook& LogBook::operator << (std::ostream& (*pf)(std::ostream&))
{
    std::cout << pf;

    if(ofs)
    {
        ofs << pf;
    }

    return *this;
}

} // end namespace Peanut