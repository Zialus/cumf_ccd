#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

const int maxThreadsPerBlock = 1024;
const int BLOCKSIZE = 128;
cudaStream_t stream[10 + 1]; //hard coded
dim3 block(BLOCKSIZE, 1, 1), grid(1, 1, 1);


#define FULL_MASK 0xffffffff

template<typename T>
inline __device__ T shfl_down(const T val, unsigned int delta) {
#if CUDA_VERSION >= 9000
    return __shfl_down_sync(FULL_MASK , val, delta);
#else
    return __shfl_down(val, delta);
#endif
}

#endif // CUDA_COMMON_H
