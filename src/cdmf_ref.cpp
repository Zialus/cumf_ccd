#include <omp.h>

#include "util.h"
#include "utilities.h"

#define kind dynamic,500

inline DTYPE RankOneUpdate(const SparseMatrix& R, const long j, const VecData& u, const DTYPE lambda) {
    DTYPE g = 0, h = lambda;
    if (R.get_csc_col_ptr()[j + 1] == R.get_csc_col_ptr()[j]) { return 0; }
    for (unsigned idx = R.get_csc_col_ptr()[j]; idx < R.get_csc_col_ptr()[j + 1]; ++idx) {
        unsigned i = R.get_csc_row_indx()[idx];
        g += u[i] * R.get_csc_val()[idx];
        h += u[i] * u[i];
    }
    DTYPE newvj = g / h;
    return newvj;
}

inline DTYPE UpdateRating(SparseMatrix& R, const VecData& Wt, const VecData& Ht, const VecData& oldWt, const VecData& oldHt) {
    DTYPE loss = 0;
#pragma omp parallel for  schedule(kind) reduction(+:loss)
    for (long c = 0; c < R.cols_; ++c) {
        DTYPE Htc = Ht[c], oldHtc = oldHt[c], loss_inner = 0;
        for (unsigned idx = R.get_csc_col_ptr()[c]; idx < R.get_csc_col_ptr()[c + 1]; ++idx) {
            R.get_csc_val()[idx] -= Wt[R.get_csc_row_indx()[idx]] * Htc - oldWt[R.get_csc_row_indx()[idx]] * oldHtc;
            loss_inner += R.get_csc_val()[idx] * R.get_csc_val()[idx];
        }
        loss += loss_inner;
    }
    return loss;
}

inline DTYPE UpdateRating(SparseMatrix& R, const VecData& Wt2, const VecData& Ht2) {
    DTYPE loss = 0;
#pragma omp parallel for schedule(kind) reduction(+:loss)
    for (long c = 0; c < R.cols_; ++c) {
        DTYPE Htc = Ht2[2 * c], oldHtc = Ht2[2 * c + 1], loss_inner = 0;
        for (unsigned idx = R.get_csc_col_ptr()[c]; idx < R.get_csc_col_ptr()[c + 1]; ++idx) {
            R.get_csc_val()[idx] -= Wt2[2 * R.get_csc_row_indx()[idx]] * Htc - Wt2[2 * R.get_csc_row_indx()[idx] + 1] * oldHtc;
            loss_inner += R.get_csc_val()[idx] * R.get_csc_val()[idx];
        }
        loss += loss_inner;
    }
    return loss;
}

inline DTYPE UpdateRating(SparseMatrix& R, const VecData& Wt, const VecData& Ht, bool add) {
    DTYPE loss = 0;
    if (add) {
#pragma omp parallel for schedule(kind) reduction(+:loss)
        for (long c = 0; c < R.cols_; ++c) {
            DTYPE Htc = Ht[c], loss_inner = 0;
            for (unsigned idx = R.get_csc_col_ptr()[c]; idx < R.get_csc_col_ptr()[c + 1]; ++idx) {
                R.get_csc_val()[idx] += Wt[R.get_csc_row_indx()[idx]] * Htc;
                loss_inner += R.get_csc_val()[idx] * R.get_csc_val()[idx];
            }
            loss += loss_inner;
        }
        return loss;
    } else {
#pragma omp parallel for schedule(kind) reduction(+:loss)
        for (long c = 0; c < R.cols_; ++c) {
            DTYPE Htc = Ht[c], loss_inner = 0;
            for (unsigned idx = R.get_csc_col_ptr()[c]; idx < R.get_csc_col_ptr()[c + 1]; ++idx) {
                R.get_csc_val()[idx] -= Wt[R.get_csc_row_indx()[idx]] * Htc;
                loss_inner += R.get_csc_val()[idx] * R.get_csc_val()[idx];
            }
            loss += loss_inner;
        }
        return loss;
    }
}

// Matrix Factorization based on Coordinate Descent
void cdmf_ref(SparseMatrix& R, MatData& W, MatData& H, TestData& T, Options& param) {
    DTYPE lambda = param.lambda;

    int num_threads_old = omp_get_num_threads();
    omp_set_num_threads(param.threads);

    // Create transpose view of R
    SparseMatrix Rt;
    Rt = R.get_shallow_transpose();
    // H is a zero matrix now.
    for (unsigned t = 0; t < param.k; ++t) {
        for (unsigned c = 0; c < R.cols_; ++c) {
            H[t][c] = 0;
        }
    }

    VecData u(R.rows_);
    VecData v(R.cols_);

    double t_update_ratings_acc = 0;
    double t_rank_one_update_acc = 0;

    for (int oiter = 1; oiter <= param.maxiter; ++oiter) {

        double t_update_ratings = 0;
        double t_rank_one_update = 0;

        for (unsigned t = 0; t < param.k; ++t) {
            double start = omp_get_wtime();

            VecData& Wt = W[t];
            VecData& Ht = H[t];

#pragma omp parallel for
            for (long i = 0; i < R.rows_; ++i) { u[i] = Wt[i]; }
#pragma omp parallel for
            for (long i = 0; i < R.cols_; ++i) { v[i] = Ht[i]; }

            // Create Rhat = R - Wt Ht^T
            if (oiter > 1) {
                UpdateRating(R, Wt, Ht, true);
                UpdateRating(Rt, Ht, Wt, true);
            }

            t_update_ratings += omp_get_wtime() - start;

            for (int iter = 1; iter <= param.maxinneriter; ++iter) {
                // Update H[t]
                start = omp_get_wtime();
#pragma omp parallel for schedule(kind) shared(u, v)
                for (long c = 0; c < R.cols_; ++c) {
                    v[c] = RankOneUpdate(R, c, u, lambda * (R.get_csc_col_ptr()[c + 1] - R.get_csc_col_ptr()[c]));
                }
                t_rank_one_update += omp_get_wtime() - start;

                // Update W[t]
                start = omp_get_wtime();
#pragma omp parallel for schedule(kind) shared(u, v)
                for (long c = 0; c < Rt.cols_; ++c) {
                    u[c] = RankOneUpdate(Rt, c, v, lambda * (Rt.get_csc_col_ptr()[c + 1] - Rt.get_csc_col_ptr()[c]));
                }
                t_rank_one_update += omp_get_wtime() - start;

            }

            // Update R and Rt
            start = omp_get_wtime();

#pragma omp parallel for
            for (long i = 0; i < R.rows_; ++i) { Wt[i] = u[i]; }
#pragma omp parallel for
            for (long i = 0; i < R.cols_; ++i) { Ht[i] = v[i]; }

            UpdateRating(R, u, v, false);
            UpdateRating(Rt, v, u, false);

            t_update_ratings += omp_get_wtime() - start;

        }

        t_rank_one_update_acc += t_rank_one_update;
        t_update_ratings_acc += t_update_ratings;

        double start = omp_get_wtime();
        double rmse = calculate_rmse_directly(W, H, T, param.k, false);

        double end = omp_get_wtime();
        double rmse_timer = end - start;

        printf("[-INFO-] iteration num %d \trank_time %.4lf|%.4lf s \tupdate_time %.4lf|%.4lfs \tRMSE=%lf time:%fs\n",
               oiter, t_rank_one_update, t_rank_one_update_acc, t_update_ratings, t_update_ratings_acc, rmse, rmse_timer);

    }
    omp_set_num_threads(num_threads_old);
}
