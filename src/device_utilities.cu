#include "device_utilities.h"

__global__ void weighted_H_all(int const* __restrict__ R_colPtr, DTYPE* __restrict__ H, DTYPE* __restrict__ temp_H, int m, int k) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < m) {
        int nnz = R_colPtr[c + 1] - R_colPtr[c];
        if (nnz != 0) {
            for (int t = 0; t < k; ++t) {
                H[c * k + t] = temp_H[c * k + t] / nnz;
            }
        }
    }
}

__global__ void weighted_H(int const* __restrict__ R_colPtr, int const* __restrict__ R_rowLim, DTYPE* __restrict__ H,
                           DTYPE* __restrict__ temp_H, int m, int k) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < m) {
        int nnz = R_rowLim[c] - R_colPtr[c];    ////////////-R_colPtr[c];
        if (nnz != 0) {
            for (int t = 0; t < k; ++t) {
                H[c * k + t] = temp_H[c * k + t] / nnz;
            }
        }
    }
}

__global__ void assignment(int const* __restrict__ R_colPtr, DTYPE* __restrict__ v, DTYPE* __restrict__ g, DTYPE* __restrict__ h,
                           DTYPE lambda, int m) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < m) {
        DTYPE gc = g[c], hc = h[c];
        if (hc == 0) {
            v[c] = 0; //
        } else {
            v[c] = gc / hc;
        }
    }
}

__global__ void GPU_rmse(int const* __restrict__ test_row, int const* __restrict__ test_col, DTYPE const* __restrict__ test_val,
                         DTYPE* __restrict__ pred_v, DTYPE* __restrict__ rmse, DTYPE const* __restrict__ W,
                         DTYPE const* __restrict__ H, int m, int k, int rows, int cols) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < m) {
        for (int t = 0; t < k; t++) {
            int i = test_row[c];
            int j = test_col[c];
            pred_v[c] += W[t * rows + (i - 1)] * H[t * cols + (j - 1)]; //W[i-1][t] * H[j-1][t];
        }
        rmse[c] = (pred_v[c] - test_val[c]) * (pred_v[c] - test_val[c]);
    }
}