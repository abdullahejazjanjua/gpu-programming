cmake_minimum_required(VERSION 3.10)
project(matmul-gpu LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 86)


add_library(utils STATIC
    ../utils/utils.cpp
    ../utils/utils.h
    matmul.cu
)

target_include_directories(utils PUBLIC ../utils)

target_compile_options(utils PRIVATE -O3 -Wall)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE utils)