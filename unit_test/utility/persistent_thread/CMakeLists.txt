if(BUILD_TESTING)
    cuda_include_directories(${PROJECT_SOURCE_DIR}/src)
    cuda_add_executable(test-persistent-thread Test.cu
                                               ${PROJECT_SOURCE_DIR}/src/utility/LogBook.cpp
                                               ${PROJECT_SOURCE_DIR}/src/utility/Timer.cpp
                                               ${PROJECT_SOURCE_DIR}/src/utility/GPU.cpp)
    target_include_directories(test-persistent-thread PRIVATE ${PROJECT_SOURCE_DIR}/src)

    add_test(NAME test_persistent_thread
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
             COMMAND test-persistent-thread
             )
endif()

