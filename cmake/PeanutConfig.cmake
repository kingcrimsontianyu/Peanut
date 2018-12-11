#------------------------------------------------------------
#------------------------------------------------------------
macro(SetGlobalCompileOption)
    # use c++11 standard
    # do not use compile extension
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)

    # note: win32 here includes the case of win64
    if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL MSVC)
        # Od: disables optimization
        # Zi: generate debug info
        set(CMAKE_CXX_FLAGS_DEBUG "-Od -Zi")
        set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -Zi -DNDEBUG")

        # suppress windows crt security related warnings
        add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    elseif(UNIX AND NOT APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL GNU)
        add_compile_options(-fdiagnostics-color=always -m64)
        add_compile_options(-Wuninitialized -Wpedantic -Wextra -Wall -Wshadow)
    else()
        message(FATAL_ERROR "Unsupported OS or compiler.")
    endif()
endmacro()
