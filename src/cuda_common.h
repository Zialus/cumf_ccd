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

struct GpuTimer {
    cudaEvent_t start{};
    cudaEvent_t stop{};

    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start() {
        cudaEventRecord(start, nullptr);
    }

    void Stop() {
        cudaEventRecord(stop, nullptr);
    }


    float Elapsed() {
        float miliseconds;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&miliseconds, start, stop);
        return miliseconds / 1000;
    }

};

#endif // CUDA_COMMON_H
