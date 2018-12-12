if(BUILD_TESTING)
    # visual studio does not need fPIC flag
    if(WIN32 AND ${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
        set(PEANUT_FPIC "")
    else()
        set(PEANUT_FPIC "-fPIC")
    endif()

    cuda_include_directories(${PROJECT_SOURCE_DIR}/src
                             ${PROJECT_SOURCE_DIR}/unit_test/atomic)

    cuda_compile(DEVICE_CODE TestManagerGPU.cu OPTIONS "-Xcompiler ${PEANUT_FPIC}")

    cuda_add_executable(test-atomic ${DEVICE_CODE})

    target_include_directories(test-atomic
                               PRIVATE ${PROJECT_SOURCE_DIR}/src
                                       ${PROJECT_SOURCE_DIR}/unit_test/atomic)
    target_sources(test-atomic
                   PRIVATE Main.cpp
                           TestManager.cpp
                           ${PROJECT_SOURCE_DIR}/src/utility/LogBook.cpp
                           ${PROJECT_SOURCE_DIR}/src/utility/Timer.cpp
                           ${PROJECT_SOURCE_DIR}/src/utility/GPU.cpp)

    # add_test(NAME test_cpu_memcheck_atomic
             # WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
             # COMMAND valgrind
             # --tool=memcheck
             # --track-origins=yes
             # --leak-check=full
             # ${CMAKE_CURRENT_BINARY_DIR}/test-atomic
             # --num-history=1e7
             # --num-collision=10
             # --num-tally=8
             # --gpu-name=K40
             # --block-per-grid=1024
             # --thread-per-block=64
             # --dose-increment=0.1
             # )

    # add_test(NAME test_gpu_memcheck_atomic
             # WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
             # COMMAND cuda-memcheck --tool memcheck
             # ${CMAKE_CURRENT_BINARY_DIR}/test-atomic
             # --num-history=1e7
             # --num-collision=10
             # --num-tally=8
             # --gpu-name=K40
             # --block-per-grid=1024
             # --thread-per-block=64
             # --dose-increment=0.1
             # )

    add_test(NAME test_gpu_atomic_K40
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
             COMMAND
             test-atomic
             --num-history=1e7
             --num-collision=10
             --num-tally=8
             --gpu-name=K40
             --block-per-grid=1024
             --thread-per-block=64
             --dose-increment=0.1
             )

    add_test(NAME test_gpu_atomic_skip_slow_K40
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
             COMMAND
             test-atomic
             --num-history=1e7
             --num-collision=10
             --num-tally=8
             --gpu-name=K40
             --block-per-grid=1024
             --thread-per-block=64
             --dose-increment=0.1
             --skip-slow
             )

    add_test(NAME test_gpu_atomic_M6000
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
             COMMAND
             test-atomic
             --num-history=1e7
             --num-collision=10
             --num-tally=8
             --gpu-name=M6000
             --block-per-grid=1024
             --thread-per-block=64
             --dose-increment=0.1
             --skip-slow
             )

    add_test(NAME test_gpu_atomic_TitanXPascal
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
             COMMAND
             test-atomic
             --num-history=1e7
             --num-collision=10
             --num-tally=8
             --gpu-name=TITAN
             --block-per-grid=1024
             --thread-per-block=64
             --dose-increment=0.1
             --skip-slow
             )

    add_test(NAME test_gpu_atomic_V100
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
             COMMAND
             test-atomic
             --num-history=1e7
             --num-collision=10
             --num-tally=8
             --gpu-name=V100
             --block-per-grid=1024
             --thread-per-block=64
             --dose-increment=0.1
             --skip-slow
             )

    add_test(NAME test_gpu_atomic_hb_wib_V100
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
             COMMAND
             test-atomic
             --num-history=3e9
             --num-collision=10
             --num-tally=8
             --gpu-name=V100
             --block-per-grid=1024
             --thread-per-block=64
             --dose-increment=0.1
             --hb-vs-wib
             )

    add_test(NAME test_skip_slow_gpu_atomic
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
             COMMAND
             test-atomic
             --num-history=1e6
             --num-collision=10
             --num-tally=8
             --gpu-name=K40
             --skip-slow
             --block-per-grid=1024
             --thread-per-block=64
             --dose-increment=0.1
             --skip-slow
             )

    add_test(NAME test_verify_gpu_atomic
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
             COMMAND
             test-atomic
             --num-history=1e2
             --num-collision=10
             --num-tally=8
             --gpu-name=K40
             --skip-slow
             --block-per-grid=1024
             --thread-per-block=64
             --verify
             )
endif()