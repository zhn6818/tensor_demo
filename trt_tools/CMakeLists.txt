cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 14)
add_definitions(-O0 -g)


add_executable(testinterface test.cpp trtInterface.cpp trtInterface.h)
