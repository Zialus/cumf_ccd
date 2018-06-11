cmake_minimum_required(VERSION 3.0)
project(cumf_ccd)

set(CMAKE_CXX_STANDARD 11)

#set(CMAKE_VERBOSE_MAKEFILE ON)

#set(CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/exec/")

set(EXTRA_NVCC_FLAGS -w -arch=sm_35 -Xptxas -dlcm=ca --std=c++11)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${EXTRA_NVCC_FLAGS})

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

file(GLOB src "${PROJECT_SOURCE_DIR}/src/*.cpp" "${PROJECT_SOURCE_DIR}/src/*.c" )
file(GLOB hdr "${PROJECT_SOURCE_DIR}/src/*.hpp" "${PROJECT_SOURCE_DIR}/src/*.h" )
file(GLOB cuda_cu "${PROJECT_SOURCE_DIR}/src/*.cu" )

CUDA_ADD_EXECUTABLE(ccdp_gpu ${src} ${hdr} ${cuda_cu})
target_link_libraries(ccdp_gpu OpenMP::OpenMP_CXX)
