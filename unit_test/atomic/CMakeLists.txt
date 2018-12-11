if(BUILD_TESTING)
    add_executable(test-atomic Test.cpp
                               ${PROJECT_SOURCE_DIR}/src/utility/LogBook.cpp
                               ${PROJECT_SOURCE_DIR}/src/utility/Timer.cpp)
    target_include_directories(test-atomic PRIVATE ${PROJECT_SOURCE_DIR}/src)

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