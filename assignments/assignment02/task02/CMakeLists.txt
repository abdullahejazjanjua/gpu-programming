cmake_minimum_required(VERSION 3.10)
project(matmul-cpu LANGUAGES CXX)

add_library(utils STATIC
    ../utils/utils.cpp
    ../utils/utils.h
)

target_include_directories(utils PUBLIC ../utils)

target_compile_options(utils PRIVATE -O3 -Wall)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE utils)