cmake_minimum_required(VERSION 3.9)
project(cumf_ccd)

set(CMAKE_CXX_STANDARD 11)

#set(CMAKE_VERBOSE_MAKEFILE ON)

#set(CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)


set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -rdynamic")

# There seems to be a bug with CMake, which makes it not add OpenMP flags to the nvcc Xcompiler option
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Add debugging to CUDA NVCC flags for NVidia's NSight tools.
set(CUDA_NVCC_FLAGS_DEBUG "${CUDA_NVCC_FLAGS_DEBUG} -G")

set(EXTRA_NVCC_FLAGS -arch=sm_35 -Xptxas -dlcm=ca --std=c++11)
#set(EXTRA_NVCC_FLAGS -arch=sm_35 -Xptxas -dlcm=ca --std=c++11 -lineinfo --ptxas-options=-v --maxrregcount 255)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${EXTRA_NVCC_FLAGS})

# Compiler flags to enable extra warning flags during all compilations
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
endif ()

# Compiler flags to enable all warnings during all compilations
#if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
#elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weverything")
#elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Wall")
#endif ()

# Compiler flags to silence/add some warnings during all compilations
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44365")
endif ()

# Compiler flags to silence/or add some warnings only during nvcc compilation
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
#    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Wno-deprecated-declarations")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
#    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Wno-deprecated-declarations")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
#    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Wno-deprecated-declarations")
endif ()


# Silence warning C4996 on Microsoft Compilers
if (MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif ()


file(GLOB src "${PROJECT_SOURCE_DIR}/src/*.cpp" "${PROJECT_SOURCE_DIR}/src/*.c")
file(GLOB hdr "${PROJECT_SOURCE_DIR}/src/*.hpp" "${PROJECT_SOURCE_DIR}/src/*.h")
file(GLOB cuda "${PROJECT_SOURCE_DIR}/src/*.cu")

set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/exec/")
CUDA_ADD_EXECUTABLE(ccdp_gpu ${src} ${hdr} ${cuda})
target_link_libraries(ccdp_gpu OpenMP::OpenMP_CXX)
