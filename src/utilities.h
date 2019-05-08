#include "pmf_util.h"
#include "common.h"

void copy_R(SparseMatrix& R, DTYPE* copy_R);

void copy_R1(DTYPE* copy_R, SparseMatrix& R);

void make_tile(SparseMatrix& R, MatInt& tiled_bin, int TS);

void make_tile_odd(SparseMatrix& R, MatInt& tiled_bin, int TS);

void tiled_binning(SparseMatrix& R, int* host_rowGroupPtr, int* LB, int* UB, int* count, MatInt& tiled_bin, int tile_no);

void binning(SparseMatrix& R, int* host_rowGroupPtr, int* LB, int* UB, int* count);

void golden_compare(MatData W, MatData W_ref, unsigned k, unsigned m);

double calculate_rmse_directly(MatData& W, MatData& H, TestData& T, int rank, bool ifALS);

double dot(const MatData& W, long i, const MatData& H, long j, bool ifALS);

float dot(const VecData& a, const VecData& b);

double calrmse(TestData& T, const MatData& W, const MatData& H, bool ifALS, bool iscol = false);

void print_matrix(MatData M, unsigned k, unsigned n);
