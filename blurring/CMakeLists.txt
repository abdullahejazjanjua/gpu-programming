cmake_minimum_required(VERSION 3.10)
project(blur LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} main.cpp)

add_library(utils STATIC
    utils.cpp utils.h blur.cu
)

target_link_libraries(
    ${PROJECT_NAME} utils ${OpenCV_LIBS}
)

target_compile_options(utils PUBLIC -O3 -Wall)