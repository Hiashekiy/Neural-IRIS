cmake_minimum_required(VERSION 3.10)
project(firi_cli LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(firi_cli firi_cli.cpp)

target_include_directories(firi_cli PRIVATE
    ${CMAKE_SOURCE_DIR}/../../../third_party/GCOPTER/gcopter/include
    /usr/include/eigen3
)
