/**
 *
 * OHIO STATE UNIVERSITY SOFTWARE DISTRIBUTION LICENSE
 *
 * Parallel CCD++ on GPU (the “Software”) Copyright (c) 2017, The Ohio State
 * University. All rights reserved.
 *
 * The Software is available for download and use subject to the terms and
 * conditions of this License. Access or use of the Software constitutes acceptance
 * and agreement to the terms and conditions of this License. Redistribution and
 * use of the Software in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the capitalized paragraph below.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the capitalized paragraph below in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. The names of Ohio State University, or its faculty, staff or students may not
 * be used to endorse or promote products derived from the Software without
 * specific prior written permission.
 *
 * This software was produced with support from the National Science Foundation
 * (NSF) through Award 1629548. Nothing in this work should be construed as
 * reflecting the official policy or position of the Defense Department, the United
 * States government, Ohio State University.
 *
 * THIS SOFTWARE HAS BEEN APPROVED FOR PUBLIC RELEASE, UNLIMITED DISTRIBUTION. THE
 * SOFTWARE IS PROVIDED “AS IS” AND WITHOUT ANY EXPRESS, IMPLIED OR STATUTORY
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, WARRANTIES OF ACCURACY, COMPLETENESS,
 * NONINFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED.  ACCESS OR USE OF THE SOFTWARE IS ENTIRELY AT THE USER’S RISK.  IN
 * NO EVENT SHALL OHIO STATE UNIVERSITY OR ITS FACULTY, STAFF OR STUDENTS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
 * TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  THE SOFTWARE
 * USER SHALL INDEMNIFY, DEFEND AND HOLD HARMLESS OHIO STATE UNIVERSITY AND ITS
 * FACULTY, STAFF AND STUDENTS FROM ANY AND ALL CLAIMS, ACTIONS, DAMAGES, LOSSES,
 * LIABILITIES, COSTS AND EXPENSES, INCLUDING ATTORNEYS’ FEES AND COURT COSTS,
 * DIRECTLY OR INDIRECTLY ARISING OUT OF OR IN CONNECTION WITH ACCESS OR USE OF THE
 * SOFTWARE.
 *
 */

/**
 *
 * Author:
 * 			Israt (nisa.1@osu.edu)
 *
 * Contacts:
 * 			Israt (nisa.1@osu.edu)
 * 			Aravind Sukumaran-Rajam (sukumaranrajam.1@osu.edu)
 * 			P. (Saday) Sadayappan (sadayappan.1@osu.edu)
 *
 */

#include <cuda.h>

#include "utilities.h"
#include "device_utilities.h"
#include "helper_fusedR.h"
#include "helper_updateH.h"

// Cyclic Coordinate Descent for Matrix Factorization
void ccdr1(SparseMatrix& R, MatData& W, MatData& H, TestData& T, Options& param) {

    int k = param.k;
    int maxiter = param.maxiter;
    int tileSize_H = param.tileSizeH;
    int tileSize_W = param.tileSizeW;
    DTYPE lambda = param.lambda;

    DTYPE* d_R_val;
    DTYPE* d_R_val_t;
    DTYPE* d_gArrU;
    DTYPE* d_hArrU;
    DTYPE* d_gArrV;
    DTYPE* d_hArrV;
    DTYPE* d_u;
    DTYPE* d_v;

    int LB[NUM_THRDS];
    int UB[NUM_THRDS];
    int LB_Rt[NUM_THRDS];
    int UB_Rt[NUM_THRDS];
    int* d_R_colPtr;
    int* d_R_rowPtr;
    int* d_row_lim_R;
    int* d_row_lim_Rt;
    int* d_test_row;
    int* d_test_col;
    int sum = 0;

    DTYPE* d_loss;
    DTYPE* d_v_new;
    DTYPE* d_Wt;
    DTYPE* d_Ht;
    DTYPE* d_W;
    DTYPE* d_H;
    DTYPE* d_test_val;
    DTYPE* d_pred_v;
    DTYPE* d_rmse;

    unsigned* d_R_rowIdx;
    unsigned* d_R_colIdx;

    DTYPE* rmse = (DTYPE*) malloc((T.nnz_ + 1) * sizeof(DTYPE));

    //omp_set_num_threads(param.threads);

    // Create transpose view of R
    SparseMatrix Rt;
    Rt = R.get_shallow_transpose();
    // initial value of the regularization term

    // H is a zero matrix now.
    for (int t = 0; t < k; ++t) {
        for (long c = 0; c < R.cols_; ++c) {
            H[t][c] = 0;
        }
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float mili = 0;

    //**************************CUDA COPY************************

    gpuErrchk(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
    size_t RCols_memsize = (R.cols_) * sizeof(DTYPE);
    size_t RRows_memsize = (R.rows_) * sizeof(DTYPE);

    size_t R_colPtr_memsize = (R.cols_ + 1) * sizeof(int);
    size_t R_rowPtr_memsize = (R.rows_ + 1) * sizeof(int);
    size_t R_rowIdx_memsize = R.nnz_ * sizeof(unsigned);
    size_t R_val_memsize = R.nnz_ * sizeof(DTYPE);

    gpuErrchk(cudaMalloc((void**) &d_W, k * R.rows_ * sizeof(DTYPE)));
    gpuErrchk(cudaMalloc((void**) &d_H, k * R.cols_ * sizeof(DTYPE)));
    gpuErrchk(cudaMalloc((void**) &d_Wt, R.rows_ * sizeof(DTYPE)));
    gpuErrchk(cudaMalloc((void**) &d_Ht, R.cols_ * sizeof(DTYPE)));
    gpuErrchk(cudaMalloc((void**) &d_u, RRows_memsize));
    gpuErrchk(cudaMalloc((void**) &d_v, RCols_memsize));
    gpuErrchk(cudaMalloc((void**) &d_v_new, RCols_memsize));
    gpuErrchk(cudaMalloc((void**) &d_gArrU, RRows_memsize));
    gpuErrchk(cudaMalloc((void**) &d_hArrU, RRows_memsize));
    gpuErrchk(cudaMalloc((void**) &d_gArrV, RCols_memsize));
    gpuErrchk(cudaMalloc((void**) &d_hArrV, RCols_memsize));
    gpuErrchk(cudaMalloc((void**) &d_R_colPtr, R_colPtr_memsize));
    gpuErrchk(cudaMalloc((void**) &d_R_rowPtr, R_rowPtr_memsize));
    gpuErrchk(cudaMalloc((void**) &d_R_rowIdx, R_rowIdx_memsize));
    gpuErrchk(cudaMalloc((void**) &d_R_colIdx, R_rowIdx_memsize));
    gpuErrchk(cudaMalloc((void**) &d_R_val, R_val_memsize));
    gpuErrchk(cudaMalloc((void**) &d_R_val_t, R_val_memsize));
    gpuErrchk(cudaMalloc((void**) &d_loss, 1 * sizeof(DTYPE)));
    gpuErrchk(cudaMalloc((void**) &d_test_row, (T.nnz_ + 1) * sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_test_col, (T.nnz_ + 1) * sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_test_val, (T.nnz_ + 1) * sizeof(DTYPE)));
    gpuErrchk(cudaMalloc((void**) &d_pred_v, (T.nnz_ + 1) * sizeof(DTYPE)));
    gpuErrchk(cudaMalloc((void**) &d_rmse, (T.nnz_ + 1) * sizeof(DTYPE)));

    gpuErrchk(cudaEventRecord(start));
    gpuErrchk(cudaMemcpy(d_R_colPtr, R.get_csc_col_ptr(), R_colPtr_memsize, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_R_rowPtr, R.get_csr_row_ptr(), R_rowPtr_memsize, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_R_rowIdx, R.get_csc_row_indx(), R_rowIdx_memsize, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_R_colIdx, R.get_csr_col_indx(), R_rowIdx_memsize, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_R_val, R.get_csc_val(), R_val_memsize, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_R_val_t, R.get_csr_val(), R_val_memsize, cudaMemcpyHostToDevice));

    for (int t = 0; t < k; ++t) {
        gpuErrchk(cudaMemcpy(d_W + t * R.rows_, &(W[t][0]), R.rows_ * sizeof(DTYPE), cudaMemcpyHostToDevice));
    }
    gpuErrchk(cudaMemset(d_H, 0, k * R.cols_ * sizeof(DTYPE)));


    //copying test
    gpuErrchk(cudaMemcpy(d_test_row, T.getTestRow(), (T.nnz_ + 1) * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_test_col, T.getTestCol(), (T.nnz_ + 1) * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_test_val, T.getTestVal(), (T.nnz_ + 1) * sizeof(DTYPE), cudaMemcpyHostToDevice));

    gpuErrchk(cudaEventRecord(stop));
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaEventElapsedTime(&mili, start, stop));

    float ACSRTime = 0;

    cudaStream_t streamT;
    gpuErrchk(cudaStreamCreate(&streamT));

    //****************** Preprocessing TILING*************

    long total_tileInRows = (R.rows_ + tileSize_H - 1) / tileSize_H;
    long total_tileInCols = (R.cols_ + tileSize_W - 1) / tileSize_W;

    MatInt row_lim_R = MatInt(total_tileInRows + 1, VecInt(R.cols_ + 1));
    MatInt row_lim_Rt = MatInt(total_tileInCols + 1, VecInt(R.rows_ + 1));
    MatInt row_lim_R_odd = MatInt(total_tileInRows + 1, VecInt(R.cols_ + 1));
    MatInt row_lim_Rt_odd = MatInt(total_tileInCols + 1, VecInt(R.rows_ + 1));

    make_tile(R, row_lim_R, tileSize_H);
    make_tile(Rt, row_lim_Rt, tileSize_W);

    //copying tiles limit rowPointers

    gpuErrchk(cudaEventRecord(start));

    gpuErrchk(cudaMalloc((void**) &d_row_lim_R, (total_tileInRows + 1) * (R.cols_ + 1) * sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &d_row_lim_Rt, (total_tileInCols + 1) * (R.rows_ + 1) * sizeof(int)));

    gpuErrchk(cudaMemcpy(d_row_lim_R, R.get_csc_col_ptr(), (R.cols_ + 1) * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_row_lim_Rt, R.get_csr_row_ptr(), (R.rows_ + 1) * sizeof(int), cudaMemcpyHostToDevice));

    for (int tile = tileSize_H; tile < (R.rows_ + tileSize_H - 1); tile += tileSize_H) {
        int tile_no = tile / tileSize_H;    // - 1;
        gpuErrchk(cudaMemcpy(d_row_lim_R + tile_no * (R.cols_ + 1), &(row_lim_R[tile_no][0]),
                             (R.cols_ + 1) * sizeof(int), cudaMemcpyHostToDevice));
    }
    for (int tile = tileSize_W; tile < (R.cols_ + tileSize_W - 1); tile += tileSize_W) {
        int tile_no = tile / tileSize_W;    // - 1;
        gpuErrchk(cudaMemcpy(d_row_lim_Rt + (tile_no * R.rows_) + tile_no, &(row_lim_Rt[tile_no][0]),
                             (R.rows_ + 1) * sizeof(int), cudaMemcpyHostToDevice));
    }

    mili = cuda_timerEnd(start, stop, streamT);

    //******************PreProcess for TILED binning*******************************
    gpuErrchk(cudaEventRecord(start));
    int** tiled_count = new int* [total_tileInRows];
    int** tiled_count_Rt = new int* [total_tileInCols];
    for (int i = 0; i < total_tileInRows; ++i) {
        tiled_count[i] = new int[NUM_THRDS];
    }
    for (int i = 0; i < total_tileInCols; ++i) {
        tiled_count_Rt[i] = new int[NUM_THRDS];
    }

    int* tiled_rowGroupPtr;
    int* tiled_rowGroupPtr_Rt;

    // Extract CSR group info on CPU
    int** tiled_host_rowGroupPtr = new int* [total_tileInRows];
    int** tiled_host_rowGroupPtr_Rt = new int* [total_tileInCols];

    for (int i = 0; i < total_tileInRows; ++i) {
        tiled_host_rowGroupPtr[i] = new int[NUM_THRDS * R.cols_];
    }

    for (int i = 0; i < total_tileInCols; ++i) {
        tiled_host_rowGroupPtr_Rt[i] = new int[NUM_THRDS * R.rows_];
    }

    for (int tile = tileSize_H; tile < (R.rows_ + tileSize_H - 1); tile += tileSize_H) {
        int tile_no = tile / tileSize_H - 1;
        tiled_binning(R, (tiled_host_rowGroupPtr[tile_no]), LB, UB, tiled_count[tile_no], row_lim_R, tile_no);
    }
    for (int tile = tileSize_W; tile < (R.cols_ + tileSize_W - 1); tile += tileSize_W) {
        int tile_no = tile / tileSize_W - 1;
        tiled_binning(Rt, (tiled_host_rowGroupPtr_Rt[tile_no]), LB_Rt, UB_Rt, tiled_count_Rt[tile_no], row_lim_Rt,
                      tile_no);
    }
    gpuErrchk(cudaMalloc((void**) &tiled_rowGroupPtr, total_tileInRows * R.cols_ * sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &tiled_rowGroupPtr_Rt, total_tileInCols * R.rows_ * sizeof(int)));

    for (int tile = tileSize_H; tile < (R.rows_ + tileSize_H - 1); tile += tileSize_H) {
        int tile_no = tile / tileSize_H - 1;
        sum = 0;
        for (int i = 0; i < NUM_THRDS; i++) {
            if (tiled_count[tile_no][i] > 0) {
                gpuErrchk(cudaMemcpy(tiled_rowGroupPtr + (tile_no * R.cols_) + sum, &(tiled_host_rowGroupPtr[tile_no][i * R.cols_]),
                                     tiled_count[tile_no][i] * sizeof(int), cudaMemcpyHostToDevice));
                sum += tiled_count[tile_no][i];
            }
        }
    }

    for (int tile = tileSize_W; tile < (Rt.rows_ + tileSize_W - 1); tile += tileSize_W) {
        int tile_no = tile / tileSize_W - 1;
        sum = 0;
        for (int i = 0; i < NUM_THRDS; i++) {
            if (tiled_count_Rt[tile_no][i] > 0) {
                gpuErrchk(cudaMemcpy(tiled_rowGroupPtr_Rt + (tile_no * Rt.cols_) + sum, &(tiled_host_rowGroupPtr_Rt[tile_no][i * R.rows_]),
                                     tiled_count_Rt[tile_no][i] * sizeof(int), cudaMemcpyHostToDevice));
                sum += tiled_count_Rt[tile_no][i];
            }
        }
    }
    mili = cuda_timerEnd(start, stop, streamT);

    //********************STARTING CCD++ ALGORTIHM************************
    printf("tileSize_H,W: %d, %d k: %d lambda:  %f\n", tileSize_H, tileSize_W, k, lambda);

    float mergeR = 0;
    float mergeRT = 0;
    float updateR = 0;
    float updateRT = 0;

    float update_time_acc = 0;
    float rank_time_acc = 0;

    for (int oiter = 1; oiter <= maxiter; ++oiter) {

        float update_time = 0;
        float rank_time = 0;
        GpuTimer update_timer;
        GpuTimer rmse_timer;
        GpuTimer rank_timer;

        int kk = 0;

        for (int tt = 0; tt < k; ++tt) {
            int t = tt;
            VecData& Wt = W[t], & Ht = H[t];
            cudaMemset(d_hArrU, 0, RRows_memsize);
            cudaMemset(d_gArrV, 0, RCols_memsize);
            cudaMemset(d_hArrV, 0, RCols_memsize);
            cudaMemset(d_gArrU, 0, RRows_memsize);

            //if (oiter > 1)
            {
                update_timer.Start();

                /**************************Updating R with add true**********************************/
                mergeR = 0;
                for (int tile = tileSize_H; tile < (R.rows_ + tileSize_H - 1); tile += tileSize_H) {
                    int tile_no = tile / tileSize_H;
//                    printf("tile from R %d\n", tile_no);
                    cuda_timerStart(start, streamT);
                    if (t == 0) {
                        kk = t;
                    } else {
                        kk = t - 1;
                    }
                    helper_UpdateR(
                            d_row_lim_R + ((tile_no - 1) * (R.cols_ + 1)),
                            d_row_lim_R + (tile_no * R.cols_) + tile_no,
                            d_R_rowIdx, d_R_val, d_W + t * R.rows_,
                            d_H + t * R.cols_, R.rows_, R.cols_, true,
                            tiled_rowGroupPtr + ((tile_no - 1) * R.cols_),
                            &(tiled_count[tile_no - 1][0]), lambda, d_gArrV,
                            d_hArrV, d_W + t * R.rows_, d_W + kk * R.rows_,
                            d_H + kk * R.cols_, t);

                    mili = cuda_timerEnd(start, stop, streamT);
                    ACSRTime += mili;
                    mergeR += mili;
                }

                cuda_timerStart(start, streamT);
                assignment <<<(R.cols_ + 1023) / 1024, 1024>>> (d_R_colPtr, d_v_new, d_gArrV, d_hArrV, lambda, R.cols_);
                mili = cuda_timerEnd(start, stop, streamT);
                ACSRTime += mili;
                mergeR += mili;
                if (oiter == 1 && t == 1) {
                    printf("time to merge R %f\n", mergeR);
                }

                /**************************Updating RTranspose with add true**********************************/
                mergeRT = 0;
                for (int tile = tileSize_W; tile < (R.cols_ + tileSize_W - 1); tile += tileSize_W) {
                    int tile_no = tile / tileSize_W;
//                    printf("tile_no from RT %d\n", tile_no);
                    cuda_timerStart(start, streamT);
                    if (t == 0) {
                        kk = t;
                    } else {
                        kk = t - 1;
                    }
                    helper_UpdateR(
                            d_row_lim_Rt + ((tile_no - 1) * (R.rows_ + 1)),
                            d_row_lim_Rt + (tile_no * R.rows_) + (tile_no),
                            d_R_colIdx, d_R_val_t, d_H + t * R.cols_,
                            d_W + t * R.rows_, R.cols_, R.rows_, true,
                            tiled_rowGroupPtr_Rt + ((tile_no - 1) * Rt.cols_),
                            &(tiled_count_Rt[tile_no - 1][0]), lambda, d_gArrU,
                            d_hArrU, d_v_new, d_H + kk * R.cols_,
                            d_W + kk * R.rows_, t);
                    mili = cuda_timerEnd(start, stop, streamT);
                    ACSRTime += mili;
                    mergeRT += mili;
//                    printf("update R in GPU takes %f \n", mili);
                }
                cuda_timerStart(start, streamT);
                assignment <<<(R.cols_ + 1023) / 1024, 1024>>>(d_R_colPtr, d_H + t * R.cols_, d_gArrV, d_hArrV, lambda, R.cols_);
                assignment <<<(R.rows_ + 1023) / 1024, 1024>>>(d_R_rowPtr, d_W + t * R.rows_, d_gArrU, d_hArrU, lambda, R.rows_);
                mili = cuda_timerEnd(start, stop, streamT);
                ACSRTime += mili;
                mergeRT += mili;
                if (oiter == 1 && t == 1) {
                    printf("time to merge Rt %f\n", mergeRT);
                }

                update_timer.Stop();
                update_time += update_timer.Elapsed();

            }

            /*************************inner iter*****************/

            rank_timer.Start();

            int maxit = param.maxinneriter;
            for (int iter = 1; iter < maxit; ++iter) {

                /*************************Update Ht***************/
                float updateHT = 0;
                for (int tile = tileSize_H; tile < (R.rows_ + tileSize_H - 1); tile += tileSize_H) {
                    int tile_no = tile / tileSize_H;
//                    printf("*****tile no HT%d\n", tile_no);
                    cuda_timerStart(start, streamT);
                    helper_rankOneUpdate_v(
                            d_row_lim_R + ((tile_no - 1) * R.cols_) + (tile_no - 1),
                            d_row_lim_R + (tile_no * R.cols_) + tile_no,
                            d_R_rowIdx, d_R_val, d_W + t * R.rows_,
                            d_H + t * R.cols_, R.rows_, R.cols_, true,
                            tiled_rowGroupPtr + ((tile_no - 1) * R.cols_),
                            &(tiled_count[tile_no - 1][0]), lambda, d_gArrV,
                            d_hArrV, d_W + t * R.rows_);
                    mili = cuda_timerEnd(start, stop, streamT);
                    ACSRTime += mili;
                    updateHT += mili;
                }
                cuda_timerStart(start, streamT);
                assignment <<<(R.cols_ + 1023) / 1024, 1024>>>(d_R_colPtr, d_H + t * R.cols_, d_gArrV, d_hArrV, lambda, R.cols_);
                mili = cuda_timerEnd(start, stop, streamT);
                ACSRTime += mili;
                updateHT += mili;
                if (oiter == 1 && t == 0 && iter == maxit - 1) {
                    printf("time to update Ht %f\n", updateHT);
                }

                /*************************Update Wt***************/
                float updateWT = 0;
                for (int tile = tileSize_W; tile < (R.cols_ + tileSize_W - 1); tile += tileSize_W) {
                    int tile_no = tile / tileSize_W;
//                    printf("*****tile no WT%d\n", tile_no);
                    cuda_timerStart(start, streamT);
                    helper_rankOneUpdate_v(
                            d_row_lim_Rt + ((tile_no - 1) * R.rows_) + (tile_no - 1),
                            d_row_lim_Rt + (tile_no * R.rows_) + (tile_no),
                            d_R_colIdx, d_R_val_t, d_H + t * R.cols_,
                            d_W + t * R.rows_, R.cols_, R.rows_, true,
                            tiled_rowGroupPtr_Rt + ((tile_no - 1) * Rt.cols_),
                            &(tiled_count_Rt[tile_no - 1][0]), lambda, d_gArrU,
                            d_hArrU, d_v_new);
                    mili = cuda_timerEnd(start, stop, streamT);
                    ACSRTime += mili;
                    updateWT += mili;
                }
                cuda_timerStart(start, streamT);
                assignment <<<(R.rows_ + 1023) / 1024, 1024>>>(d_R_rowPtr, d_W + t * R.rows_, d_gArrU, d_hArrU, lambda, R.rows_);
                mili = cuda_timerEnd(start, stop, streamT);
                ACSRTime += mili;
                updateWT += mili;
                if (oiter == 1 && t == 0 && iter == maxit - 1) {
                    printf("time to update Wt %f\n", updateWT);
                }
            }

            rank_timer.Stop();
            rank_time += rank_timer.Elapsed();


            update_timer.Start();
            /**************************Updating R = R - Wt * Ht  *****************************/
            updateR = 0;
            if (t == k - 1) {
                for (int tile = tileSize_H; tile < (R.rows_ + tileSize_H - 1); tile += tileSize_H) {
                    int tile_no = tile / tileSize_H;
//                    printf("tile no %d\n", tile_no);
                    cuda_timerStart(start, streamT);
                    helper_UpdateR(
                            d_row_lim_R + ((tile_no - 1) * (R.cols_ + 1)),
                            d_row_lim_R + (tile_no * R.cols_) + tile_no,
                            d_R_rowIdx, d_R_val, d_W + t * R.rows_,
                            d_H + t * R.cols_, R.rows_, R.cols_, false,
                            tiled_rowGroupPtr + ((tile_no - 1) * R.cols_),
                            &(tiled_count[tile_no - 1][0]), lambda, d_gArrU,
                            d_hArrU, d_W + t * R.rows_, d_W + t * R.rows_,
                            d_H + t * R.cols_, t);
                    mili = cuda_timerEnd(start, stop, streamT);
                    ACSRTime += mili;
                    updateR += mili;
                }
            }
            if (oiter == 1 && t == k - 1) {
                printf("time to update R %f ms\n", updateR);
            }

            /**************************Updating RT = RT - Wt * Ht  *****************************/
            updateRT = 0;

            if (t == k - 1) {
                for (int tile = tileSize_W; tile < (R.cols_ + tileSize_W - 1); tile += tileSize_W) {
                    int tile_no = tile / tileSize_W;
//                    printf("tile no %d\n", tile_no);
                    cuda_timerStart(start, streamT);
                    helper_UpdateR(
                            d_row_lim_Rt + ((tile_no - 1) * (R.rows_ + 1)),
                            d_row_lim_Rt + (tile_no * R.rows_) + (tile_no),
                            d_R_colIdx, d_R_val_t, d_H + t * R.cols_,
                            d_W + t * R.rows_, R.cols_, R.rows_, false,
                            tiled_rowGroupPtr_Rt + ((tile_no - 1) * Rt.cols_),
                            &(tiled_count_Rt[tile_no - 1][0]), lambda, d_gArrU,
                            d_hArrU, d_H + t * R.cols_, d_H + t * R.cols_,
                            d_W + t * R.rows_, t);
                    mili = cuda_timerEnd(start, stop, streamT);
                    ACSRTime += mili;
                    updateRT += mili;
                }
            }
            if (oiter == 1 && t == k - 1) {
                printf("time to update Rt %f ms\n", updateRT);
            }


            if (oiter == 1 && t == 1) {
                printf("iter %d time for 1 feature: %f ms\n", oiter, ACSRTime);
            }

            update_timer.Stop();
            update_time += update_timer.Elapsed();

        }


        update_time_acc += update_time;
        rank_time_acc += rank_time;
        /****************Check RMSE********************/
        rmse_timer.Start();

        cudaMemset(d_rmse, 0, (T.nnz_ + 1) * sizeof(DTYPE));
        cudaMemset(d_pred_v, 0, (T.nnz_ + 1) * sizeof(DTYPE));

        GPU_rmse <<<(T.nnz_ + 1023) / 1024, 1024>>>(d_test_row, d_test_col,
                d_test_val, d_pred_v, d_rmse, d_W, d_H, T.nnz_, k, R.rows_, R.cols_);

        double tot_rmse = 0;
        double f_rmse = 0;

        gpuErrchk(cudaMemcpy(&(rmse[0]), d_rmse, (T.nnz_ + 1) * sizeof(DTYPE), cudaMemcpyDeviceToHost));

#pragma omp parallel for reduction(+:tot_rmse)
        for (int i = 0; i < T.nnz_; ++i) {
            tot_rmse += rmse[i];
        }
        f_rmse = sqrt(tot_rmse / T.nnz_);
        rmse_timer.Stop();

        float rmse_time = rmse_timer.Elapsed();
        printf("[-INFO-] iteration num %d \trank_time %.4lf|%.4lf s \tupdate_time %.4lf|%.4lfs \tRMSE=%lf time:%fs\n",
               oiter, rank_time, rank_time_acc, update_time, update_time_acc, f_rmse, rmse_time);

    }

    //Copy Matrices W and H back to host
    for (int t = 0; t < k; ++t) {
        gpuErrchk(cudaMemcpy(&(W[t][0]), d_W + t * R.rows_, R.rows_ * sizeof(DTYPE), cudaMemcpyDeviceToHost));
    }
    for (int t = 0; t < k; ++t) {
        gpuErrchk(cudaMemcpy(&(H[t][0]), d_H + t * R.cols_, R.cols_ * sizeof(DTYPE), cudaMemcpyDeviceToHost));
    }

    gpuErrchk(cudaFree(d_u));
    gpuErrchk(cudaFree(d_v));
    gpuErrchk(cudaFree(d_W));
    gpuErrchk(cudaFree(d_H));
    gpuErrchk(cudaFree(d_R_rowIdx));
    gpuErrchk(cudaFree(d_R_colPtr));
    gpuErrchk(cudaFree(d_R_val));
    gpuErrchk(cudaFree(d_R_colIdx));
    gpuErrchk(cudaFree(d_R_rowPtr));
    gpuErrchk(cudaFree(d_R_val_t));
    gpuErrchk(cudaFree(d_gArrU));
    gpuErrchk(cudaFree(d_gArrV));
    gpuErrchk(cudaFree(d_hArrU));
    gpuErrchk(cudaFree(d_hArrV));

}
