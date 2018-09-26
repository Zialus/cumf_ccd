#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include <cuda_runtime.h>

const int maxThreadsPerBlock = 1024;
const int BLOCKSIZE = 128;
cudaStream_t stream[10 + 1]; //hard coded
dim3 block(BLOCKSIZE, 1, 1), grid(1, 1, 1);

#endif // CUDA_COMMON_H