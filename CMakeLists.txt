cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(Peanut)

# create boolean BUILD_TESTING
include(CTest)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(PeanutConfig)
SetGlobalCompileOption()

add_subdirectory(unit_test)


