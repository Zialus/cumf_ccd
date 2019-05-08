#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cassert>
#include <cstdio>

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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s - %s %d\n", cudaGetErrorString(code), file, line);
        assert(code == cudaSuccess);
    }
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

void cuda_timerStart(cudaEvent_t start, cudaStream_t streamT) {
    cudaEventRecord(start, streamT);
}

float cuda_timerEnd(cudaEvent_t start, cudaEvent_t stop, cudaStream_t streamT) {
    float mili = 0;
    cudaDeviceSynchronize();
    cudaEventRecord(stop, streamT);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&mili, start, stop);
    return mili;
}


#endif // CUDA_COMMON_H
