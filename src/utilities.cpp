#include "utilities.h"

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

void copy_R(SparseMatrix& R, DTYPE* copy_R) {
    auto val_ptr = R.get_csr_val();
#pragma omp parallel for
    for (int c = 0; c < R.cols_; ++c) {
        for (int idx = R.get_csc_col_ptr()[c]; idx < R.get_csc_col_ptr()[c + 1]; ++idx) {
            copy_R[idx] = val_ptr[idx];
        }
    }
}

void copy_R1(DTYPE* copy_R, SparseMatrix& R) {
    auto val_ptr = R.get_csr_val();
#pragma omp parallel for
    for (int c = 0; c < R.cols_; ++c) {
        for (int idx = R.get_csc_col_ptr()[c]; idx < R.get_csc_col_ptr()[c + 1]; ++idx) {
            val_ptr[idx] = copy_R[idx];
        }
    }
}

void make_tile(SparseMatrix& R, MatInt& tiled_bin, const int TS) {
#pragma omp parallel for
    for (int c = 0; c < R.cols_; ++c) {
        long idx = R.get_csc_col_ptr()[c];
        tiled_bin[0][c] = idx;
        for (int tile = TS; tile < (R.rows_ + TS - 1); tile += TS) {
            int tile_no = tile / TS; // - 1;
            while (R.get_csc_row_indx()[idx] < tile && idx < R.get_csc_col_ptr()[c + 1]) {
                idx++;
            }
            tiled_bin[tile_no][c] = idx;
        }
    }
}

void make_tile_odd(SparseMatrix& R, MatInt& tiled_bin, const int TS) {
#pragma omp parallel for
    for (int c = 0; c < R.cols_; ++c) {
        long idx = R.get_csc_col_ptr()[c];
        tiled_bin[0][c] = idx;
        for (int tile = TS + (TS / 2); tile < (R.rows_ + (TS + (TS / 2)) - 1); tile += TS) {
            int tile_no = tile / TS; // - 1;
            while (R.get_csc_row_indx()[idx] < tile && idx < R.get_csc_col_ptr()[c + 1]) {
                idx++;
            }
            tiled_bin[tile_no][c] = idx;
        }
    }
}

void tiled_binning(SparseMatrix& R, int* host_rowGroupPtr, int* LB, int* UB, int* count, MatInt& tiled_bin, const int tile_no) {
    for (int i = 0; i < NUM_THRDS; i++) {
        count[i] = 0;
        UB[i] = (1 << i) * THREADLOAD;
        LB[i] = UB[i] >> 1;
    }
    LB[0] = 0;
    UB[NUM_THRDS - 1] = R.max_col_nnz_ + 1;
    //***********binned
    // omp_set_num_threads(NUM_THRDS);  // create as many CPU threads as there are # of bins
    // #pragma omp parallel
    // {
    // unsigned int cpu_thread_id = omp_get_thread_num();
    // int i = cpu_thread_id; count[i] = 0;
    // for (int col = 0; col < R.cols; col++){
    // //for (int col = tile_no_c*5*tileSize_H; col < ((tile_no_c+1)*5*tileSize_H) && col < R.cols ; col++){
    //         int NNZ = tiled_bin[tile_no+1][col] -  tiled_bin[tile_no][col]; // R.col_ptr[col + 1] - R.col_ptr[col];
    //         if (NNZ >= LB[i] && NNZ < UB[i]){
    //             host_rowGroupPtr[R.cols * i + count[i]++] = col;
    //          }
    //     }
    // }

    //*********non-binned
    int i = 6;
    count[i] = 0;
    for (int col = 0; col < R.cols_; col++) {
        host_rowGroupPtr[R.cols_ * i + count[i]++] = col;
    }

    //*********non-binned

    // int i = 6;
    // count[i] = 0;
    // for (int col = 0; col < R.cols; col++){
    //     int NNZ = R.col_ptr[col+1] -  R.col_ptr[col];
    //     host_rowGroupPtr[R.cols * i + count[i]++] = col;
    //     printf("%d %d\n",col, NNZ );
    //  }
    //  printf("done for R\n");
}

void binning(SparseMatrix& R, int* host_rowGroupPtr, int* LB, int* UB, int* count) {
    for (int i = 0; i < NUM_THRDS; i++) {
        count[i] = 0;
        UB[i] = (1 << i) * THREADLOAD + 1;
        LB[i] = UB[i] >> 1;

    }
    LB[0] = 0;
    UB[NUM_THRDS - 1] = R.max_col_nnz_ + 1;

    omp_set_num_threads(NUM_THRDS); // create as many CPU threads as there are # of bins
#pragma omp parallel
    {
        unsigned int cpu_thread_id = omp_get_thread_num();
        int i = cpu_thread_id;
        for (int col = 0; col < R.cols_; col++) {
            int NNZ = R.get_csc_col_ptr()[col + 1] - R.get_csc_col_ptr()[col];
            if (NNZ > LB[i] && NNZ < UB[i]) {
                host_rowGroupPtr[R.cols_ * i + count[i]++] = col;    ////changed
            }
        }
    }
}
