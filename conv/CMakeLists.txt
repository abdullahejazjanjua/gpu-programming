cmake_minimum_required(VERSION 3.10)
project(conv LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 75)

add_executable(${PROJECT_NAME} main.cpp)

add_library(utils STATIC
    utils.cpp utils.h common.h conv-kernels.cu
)

target_link_libraries(
    ${PROJECT_NAME} utils
)

target_compile_options(utils PUBLIC -O3 -Wall)