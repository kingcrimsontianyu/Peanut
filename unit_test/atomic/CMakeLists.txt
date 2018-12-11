if(BUILD_TESTING)
    # visual studio does not need fPIC flag
    if(WIN32 AND ${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
        set(PEANUT_FPIC "")
    else()
        set(PEANUT_FPIC "-fPIC")
    endif()

    cuda_include_directories(${PROJECT_SOURCE_DIR}/src
                             ${PROJECT_SOURCE_DIR}/unit_test/atomic)

    cuda_compile(DEVICE_CODE TestGPU.cu OPTIONS "-Xcompiler ${PEANUT_FPIC}")

    cuda_add_executable(test-atomic ${DEVICE_CODE})

    target_include_directories(test-atomic
                               PRIVATE ${PROJECT_SOURCE_DIR}/src
                                       ${PROJECT_SOURCE_DIR}/unit_test/atomic)
    target_sources(test-atomic
                   PRIVATE Test.cpp
                           ${PROJECT_SOURCE_DIR}/src/utility/LogBook.cpp
                           ${PROJECT_SOURCE_DIR}/src/utility/Timer.cpp)

    add_test(NAME test_atomic
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
             COMMAND test-atomic
             --num-history=1e7
             --num-collision=10
             --num-tally=8
             --gpu-name=K40
             --block-per-grid=1024
             --thread-per-block=64
             --dose-increment=0.1
             )

    add_test(NAME test_cpu_memcheck_atomic
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
             COMMAND valgrind
             --tool=memcheck
             --track-origins=yes
             --leak-check=full
             ${CMAKE_CURRENT_BINARY_DIR}/test-atomic
             --num-history=1e7
             --num-collision=10
             --num-tally=8
             --gpu-name=K40
             --block-per-grid=1024
             --thread-per-block=64
             --dose-increment=0.1
             )

    add_test(NAME test_gpu_memcheck_atomic
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
             COMMAND cuda-memcheck --tool memcheck
             ${CMAKE_CURRENT_BINARY_DIR}/test-atomic
             --num-history=1e7
             --num-collision=10
             --num-tally=8
             --gpu-name=K40
             --block-per-grid=1024
             --thread-per-block=64
             --dose-increment=0.1
             )
endif()