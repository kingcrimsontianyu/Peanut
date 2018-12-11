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
        set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
        set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")

        add_compile_options(-fdiagnostics-color=always -m64)
        add_compile_options(-Wuninitialized -Wpedantic -Wextra -Wall -Wshadow)
    else()
        message(FATAL_ERROR "Unsupported OS or compiler.")
    endif()

    SetGPUOption()

endmacro()


#------------------------------------------------------------
#------------------------------------------------------------
macro(SetGPUOption)
    find_package(CUDA REQUIRED)
    if(CUDA_VERSION_STRING VERSION_LESS "9.1")
        string(CONCAT ERROR_MSG "--> ARCHER: Current CUDA version "
                             ${CUDA_VERSION_STRING}
                             " is too old. Must upgrade it to 9.1 or newer.")
        message(FATAL_ERROR ${ERROR_MSG})
    endif()

    set(CUDA_64_BIT_DEVICE_CODE ON)
    set(CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
    set(CUDA_VERBOSE_BUILD ON)
    set(CUDA_NVCC_FLAGS "-Xptxas -dlcm=ca")

    # msvc has no notion of c++ standard o0
    if(NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
        list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
    endif()

    list(APPEND CUDA_NVCC_FLAGS "--generate-code arch=compute_30,code=sm_30 \
--generate-code arch=compute_35,code=sm_35 \
--generate-code arch=compute_52,code=sm_52 \
--generate-code arch=compute_61,code=sm_61 \
--generate-code arch=compute_70,code=sm_70")

    if(NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
        list(APPEND CUDA_NVCC_FLAGS "-ccbin \"${CMAKE_CXX_COMPILER}\"")
    endif()

    set(CUDA_NVCC_FLAGS_RELWITHDEBINFO "-O2 -g -lineinfo")
    set(CUDA_NVCC_FLAGS_RELEASE "-O2")
    set(CUDA_NVCC_FLAGS_DEBUG "-O0 -g -G")
endmacro()




