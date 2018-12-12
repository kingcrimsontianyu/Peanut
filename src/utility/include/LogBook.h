#if !defined(_LOG_BOOK_H_)
#define _LOG_BOOK_H_

#include <iostream>
#include <fstream>

namespace Peanut
{
//------------------------------------------------------------
// Singleton
//------------------------------------------------------------
class LogBook
{
public:
    //------------------------------------------------------------
    //------------------------------------------------------------
    static LogBook& GetLogBook();

    //------------------------------------------------------------
    // print value to screen and file
    //------------------------------------------------------------
    template<typename T> LogBook& operator << (const T& value);

    //------------------------------------------------------------
    // special case for std::endl
    //------------------------------------------------------------
    LogBook& operator << (std::ostream& (*pf)(std::ostream&));

private:
    LogBook();
    ~LogBook();
    std::ofstream ofs;
public:
    LogBook(const LogBook& rhs) = delete; // move ctor is not generated
    LogBook& operator = (const LogBook& rhs) = delete; // move assignment op is not generated
};

//------------------------------------------------------------
//------------------------------------------------------------
template<typename T> LogBook& LogBook::operator << (const T& value)
{
    std::cout << value;

    if(ofs)
    {
        ofs << value;
    }

    return *this;
}

} // end namespace Peanut

#endif