#if !defined(_EXCEPTION_H_)
#define _EXCEPTION_H_

#include <stdexcept>

namespace Peanut
{
//------------------------------------------------------------
//------------------------------------------------------------
class PeanutException : public std::runtime_error
{
public:
    explicit PeanutException(const std::string& message);

    explicit PeanutException(const char* message);

    ~PeanutException() noexcept = default;

    virtual const char* what() const noexcept;
};

//------------------------------------------------------------
//------------------------------------------------------------
inline PeanutException::PeanutException(const std::string& message) :
std::runtime_error(message)
{}

//------------------------------------------------------------
//------------------------------------------------------------
inline PeanutException::PeanutException(const char* message) :
std::runtime_error(message)
{}

//------------------------------------------------------------
//------------------------------------------------------------
inline const char* PeanutException::what() const noexcept
{
    return std::runtime_error::what();
}

} // end namespace Peanut

#endif
