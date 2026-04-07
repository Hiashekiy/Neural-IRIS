cmake_minimum_required(VERSION 3.10)
project(decomputil_cli LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(decomp_cli decomp_cli.cpp)

target_include_directories(decomp_cli PRIVATE
    ${CMAKE_SOURCE_DIR}/../../../third_party/DecompUtil/include
    /usr/include/eigen3
)
