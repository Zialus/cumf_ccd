#ifndef PMF_UTIL_H
#define PMF_UTIL_H

#include "util.h"
#include <memory>

class TestData;
class SparseMatrix;
class Options;

using VecData = std::vector<DTYPE>;
using MatData = std::vector<VecData>;
using VecInt = std::vector<int>;
using MatInt = std::vector<VecInt>;

void cdmf_ref(SparseMatrix& R, MatData& W, MatData& H, TestData& T, Options& param);
void ccdr1(SparseMatrix& R, MatData& W, MatData& H, TestData& T, Options& param);
void load_from_binary(const char* srcdir, SparseMatrix& R, TestData& T);
void init_random(MatData& X, long k, long n);

class SparseMatrix {
public:
    long rows_, cols_, nnz_, max_row_nnz_, max_col_nnz_;

    void read_binary_file(const std::string& fname_csr_row_ptr, const std::string& fname_csr_col_indx,
                          const std::string& fname_csr_val,
                          const std::string& fname_csc_col_ptr, const std::string& fname_csc_row_indx,
                          const std::string& fname_csc_val) {
        /// read csr
        this->read_compressed(fname_csr_row_ptr, fname_csr_col_indx, fname_csr_val,
                              this->csr_row_ptr_, this->csr_col_indx_, this->csr_val_, this->rows_ + 1,
                              this->max_row_nnz_);

        /// read csc
        this->read_compressed(fname_csc_col_ptr, fname_csc_row_indx, fname_csc_val,
                              this->csc_col_ptr_, this->csc_row_indx_, this->csc_val_, this->cols_ + 1,
                              this->max_col_nnz_);
    }

    void initialize_matrix(long rows, long cols, long nnz) {
        this->rows_ = rows;
        this->cols_ = cols;
        this->nnz_ = nnz;

        /// alloc csr
        this->alloc_space(this->csr_row_ptr_, this->csr_col_indx_, this->csr_val_, rows + 1);

        /// alloc csc
        this->alloc_space(this->csc_col_ptr_, this->csc_row_indx_, this->csc_val_, cols + 1);
    }


    SparseMatrix get_shallow_transpose() {
        SparseMatrix shallow_transpose;
        shallow_transpose.cols_ = rows_;
        shallow_transpose.rows_ = cols_;
        shallow_transpose.nnz_ = nnz_;
        shallow_transpose.csc_val_ = csr_val_;
        shallow_transpose.csr_val_ = csc_val_;
        shallow_transpose.csc_col_ptr_ = csr_row_ptr_;
        shallow_transpose.csr_row_ptr_ = csc_col_ptr_;
        shallow_transpose.csr_col_indx_ = csc_row_indx_;
        shallow_transpose.csc_row_indx_ = csr_col_indx_;
        shallow_transpose.max_col_nnz_ = max_row_nnz_;
        shallow_transpose.max_row_nnz_ = max_col_nnz_;

        return shallow_transpose;
    }

    unsigned* get_csc_col_ptr() const {
        return csc_col_ptr_.get();
    }

    unsigned* get_csc_row_indx() const {
        return csc_row_indx_.get();
    }

    DTYPE* get_csc_val() const {
        return csc_val_.get();
    }

    unsigned* get_csr_col_indx() const {
        return csr_col_indx_.get();
    }

    unsigned* get_csr_row_ptr() const {
        return csr_row_ptr_.get();
    }

    DTYPE* get_csr_val() const {
        return csr_val_.get();
    }

private:
    void read_compressed(const std::string& fname_cs_ptr, const std::string& fname_cs_indx, const std::string& fname_cs_val,
                         std::shared_ptr<unsigned>& cs_ptr, std::shared_ptr<unsigned>& cs_indx, std::shared_ptr<DTYPE>& cs_val,
                         long num_elems_in_cs_ptr, long& max_nnz_in_one_dim) {

        FILE* f_indx = fopen(fname_cs_indx.c_str(), "rb");
        FILE* f_val = fopen(fname_cs_val.c_str(), "rb");

        CHECK_FREAD(fread(&cs_indx.get()[0], sizeof(unsigned) * this->nnz_, 1, f_indx), 1);
        CHECK_FREAD(fread(&cs_val.get()[0], sizeof(float) * this->nnz_, 1, f_val), 1);


        std::ifstream f_ptr(fname_cs_ptr, std::ios::binary);
        max_nnz_in_one_dim = std::numeric_limits<long>::min();

        int cur = 0;
        for (long i = 0; i < num_elems_in_cs_ptr; i++) {
            int prev = cur;
            f_ptr.read((char*) &cur, sizeof(int));
            cs_ptr.get()[i] = cur;

            if (i > 0) { max_nnz_in_one_dim = std::max<long>(max_nnz_in_one_dim, cur - prev); }
        }


//    fread(&cs_ptr.get()[0], sizeof(unsigned) * num_elems_in_cs_ptr, 1, f_val);

        fclose(f_indx);
        fclose(f_val);
    }

    void alloc_space(std::shared_ptr<unsigned>& cs_ptr, std::shared_ptr<unsigned>& cs_indx,
                     std::shared_ptr<DTYPE>& cs_val, long num_elems_in_cs_ptr) {

        cs_ptr = std::shared_ptr<unsigned>(new unsigned[num_elems_in_cs_ptr], std::default_delete<unsigned[]>());
        cs_indx = std::shared_ptr<unsigned>(new unsigned[this->nnz_], std::default_delete<unsigned[]>());
        cs_val = std::shared_ptr<DTYPE>(new DTYPE[this->nnz_], std::default_delete<DTYPE[]>());
    }

    std::shared_ptr<unsigned> csc_col_ptr_, csr_row_ptr_, col_nnz_, row_nnz_;
    std::shared_ptr<DTYPE> csr_val_, csc_val_;
    std::shared_ptr<unsigned> csc_row_indx_, csr_col_indx_;
};

class TestData {
public:
    long rows_, cols_, nnz_;

    void read(long rows, long cols, long nnz, const std::string& filename) {
        this->rows_ = rows;
        this->cols_ = cols;
        this->nnz_ = nnz;

        test_row = std::unique_ptr<unsigned[]>(new unsigned[nnz]);
        test_col = std::unique_ptr<unsigned[]>(new unsigned[nnz]);
        test_val = std::unique_ptr<DTYPE[]>(new DTYPE[nnz]);

        std::ifstream fp(filename);
        for (long idx = 0; idx < nnz; ++idx) {
            fp >> test_row[idx] >> test_col[idx] >> test_val[idx];
        }
    }


    void read_binary_file(long rows, long cols, long nnz,
                          const std::string& fname_data,
                          const std::string& fname_row,
                          const std::string& fname_col) {
        this->rows_ = rows;
        this->cols_ = cols;
        this->nnz_ = nnz;

        test_row = std::unique_ptr<unsigned[]>(new unsigned[nnz]);
        test_col = std::unique_ptr<unsigned[]>(new unsigned[nnz]);
        test_val = std::unique_ptr<DTYPE[]>(new DTYPE[nnz]);

        FILE* f_val = fopen(fname_data.c_str(), "rb");
        FILE* f_row = fopen(fname_row.c_str(), "rb");
        FILE* f_col = fopen(fname_col.c_str(), "rb");

        CHECK_FREAD(fread(&test_val.get()[0], sizeof(DTYPE) * this->nnz_, 1, f_val), 1);
        CHECK_FREAD(fread(&test_row.get()[0], sizeof(unsigned) * this->nnz_, 1, f_row), 1);
        CHECK_FREAD(fread(&test_col.get()[0], sizeof(unsigned) * this->nnz_, 1, f_col), 1);

        fclose(f_val);
        fclose(f_row);
        fclose(f_col);
    }

    unsigned* getTestCol() const {
        return test_col.get();
    }

    unsigned* getTestRow() const {
        return test_row.get();
    }

    DTYPE* getTestVal() const {
        return test_val.get();
    }

private:
    std::unique_ptr<unsigned[]> test_row, test_col;
    std::unique_ptr<DTYPE[]> test_val;
};

#endif //PMF_UTIL_H
