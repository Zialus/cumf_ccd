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

#include "util.h"

void load_from_binary(const char* srcdir, SparseMatrix& R, TestData& data) {
    char filename[1024];
    snprintf(filename, sizeof(filename), "%s/meta_modified_all", srcdir);
    FILE* fp = fopen(filename, "r");

    if (fp == nullptr) {
        printf("Can't open meta input file.\n");
        exit(EXIT_FAILURE);
    }

    long m;
    long n;
    long nnz;
    CHECK_FSCAN(fscanf(fp, "%ld %ld %ld", &m, &n, &nnz), 3);

    char buf[1024];
    char binary_filename_val[1024];
    char binary_filename_row[1024];
    char binary_filename_col[1024];
    char binary_filename_rowptr[1024];
    char binary_filename_colidx[1024];
    char binary_filename_csrval[1024];
    char binary_filename_colptr[1024];
    char binary_filename_rowidx[1024];
    char binary_filename_cscval[1024];

    CHECK_FSCAN(fscanf(fp, "%1023s", buf), 1);
    snprintf(binary_filename_val, sizeof(binary_filename_val), "%s/%s", srcdir, buf);
    CHECK_FSCAN(fscanf(fp, "%1023s", buf), 1);
    snprintf(binary_filename_row, sizeof(binary_filename_row), "%s/%s", srcdir, buf);
    CHECK_FSCAN(fscanf(fp, "%1023s", buf), 1);
    snprintf(binary_filename_col, sizeof(binary_filename_col), "%s/%s", srcdir, buf);
    CHECK_FSCAN(fscanf(fp, "%1023s", buf), 1);
    snprintf(binary_filename_rowptr, sizeof(binary_filename_rowptr), "%s/%s", srcdir, buf);
    CHECK_FSCAN(fscanf(fp, "%1023s", buf), 1);
    snprintf(binary_filename_colidx, sizeof(binary_filename_colidx), "%s/%s", srcdir, buf);
    CHECK_FSCAN(fscanf(fp, "%1023s", buf), 1);
    snprintf(binary_filename_csrval, sizeof(binary_filename_csrval), "%s/%s", srcdir, buf);
    CHECK_FSCAN(fscanf(fp, "%1023s", buf), 1);
    snprintf(binary_filename_colptr, sizeof(binary_filename_colptr), "%s/%s", srcdir, buf);
    CHECK_FSCAN(fscanf(fp, "%1023s", buf), 1);
    snprintf(binary_filename_rowidx, sizeof(binary_filename_rowidx), "%s/%s", srcdir, buf);
    CHECK_FSCAN(fscanf(fp, "%1023s", buf), 1);
    snprintf(binary_filename_cscval, sizeof(binary_filename_cscval), "%s/%s", srcdir, buf);


    auto t0 = std::chrono::high_resolution_clock::now();
    R.initialize_matrix(m, n, nnz);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> deltaT = t1 - t0;
    std::cout << "[info] LOL TIMER: " << deltaT.count() << "s.\n";


    auto t2 = std::chrono::high_resolution_clock::now();

    R.read_binary_file(m, n, nnz,
//                       binary_filename_val, binary_filename_row, binary_filename_col,
                       binary_filename_rowptr, binary_filename_colidx, binary_filename_csrval,
                       binary_filename_colptr, binary_filename_rowidx, binary_filename_cscval);
    auto t3 = std::chrono::high_resolution_clock::now();
    deltaT = t3 - t2;
    std::cout << "[info] LOL TIMER: " << deltaT.count() << "s.\n";

    auto t4 = std::chrono::high_resolution_clock::now();

    if (fscanf(fp, "%ld %1023s", &nnz, buf) != EOF) {
        snprintf(filename, sizeof(filename), "%s/%s", srcdir, buf);
        data.read(m, n, nnz, filename);
    }

    auto t5 = std::chrono::high_resolution_clock::now();
    deltaT = t5 - t4;
    std::cout << "[info] LOL TIMER: " << deltaT.count() << "s.\n";


    fclose(fp);
}

void init_random(MatData& X, long k, long n) {
    X = MatData(k, VecData(n));
    srand(0L);
    for (long i = 0; i < n; ++i) {
        for (long j = 0; j < k; ++j) {
            X[j][i] = (DTYPE) 0.1 * ((DTYPE) rand() / RAND_MAX) + (DTYPE) 0.001;
        }
    }
}


void SparseMatrix::initialize_matrix(long rows, long cols, long nnz) {
    this->rows_ = rows;
    this->cols_ = cols;
    this->nnz_ = nnz;

    /// alloc csr
    this->alloc_space(this->csr_row_ptr_, this->csr_col_indx_, this->csr_val_, rows + 1);

    /// alloc csc
    this->alloc_space(this->csc_col_ptr_, this->csc_row_indx_, this->csc_val_, cols + 1);
}

void SparseMatrix::alloc_space(std::shared_ptr<unsigned>& cs_ptr, std::shared_ptr<unsigned>& cs_indx,
                               std::shared_ptr<DTYPE>& cs_val, long num_elems_in_cs_ptr) {
    cs_ptr = std::shared_ptr<unsigned>(new unsigned[num_elems_in_cs_ptr], std::default_delete<unsigned[]>());
    cs_indx = std::shared_ptr<unsigned>(new unsigned[this->nnz_], std::default_delete<unsigned[]>());
    cs_val = std::shared_ptr<DTYPE>(new DTYPE[this->nnz_], std::default_delete<DTYPE[]>());
}

void SparseMatrix::read_binary_file(long rows, long cols, long nnz,
//                                    std::string fname_data, std::string fname_row, std::string fname_col,
                                    const std::string& fname_csr_row_ptr, const std::string& fname_csr_col_indx,
                                    const std::string& fname_csr_val,
                                    const std::string& fname_csc_col_ptr, const std::string& fname_csc_row_indx,
                                    const std::string& fname_csc_val) {
    /// read csr
    this->read_compressed(fname_csr_row_ptr, fname_csr_col_indx, fname_csr_val,
                          this->csr_row_ptr_, this->csr_col_indx_, this->csr_val_, rows + 1,
                          this->max_row_nnz_);

    /// read csc
    this->read_compressed(fname_csc_col_ptr, fname_csc_row_indx, fname_csc_val,
                          this->csc_col_ptr_, this->csc_row_indx_, this->csc_val_, cols + 1,
                          this->max_col_nnz_);
}

void SparseMatrix::read_compressed(const std::string& fname_cs_ptr, const std::string& fname_cs_indx, const std::string& fname_cs_val,
                                   std::shared_ptr<unsigned>& cs_ptr, std::shared_ptr<unsigned>& cs_indx, std::shared_ptr<DTYPE>& cs_val,
                                   long num_elems_in_cs_ptr, long& max_nnz_in_one_dim) {
    std::ifstream f_indx(fname_cs_indx, std::ios::binary);
    std::ifstream f_val(fname_cs_val, std::ios::binary);

    for (long i = 0; i < this->nnz_; i++) {
        f_indx.read((char*) &cs_indx.get()[i], sizeof(unsigned));
        f_val.read((char*) &cs_val.get()[i], sizeof(float));
    }

    std::ifstream f_ptr(fname_cs_ptr, std::ios::binary);
    max_nnz_in_one_dim = std::numeric_limits<long>::min();

    int cur = 0;
    for (long i = 0; i < num_elems_in_cs_ptr; i++) {
        int prev = cur;
        f_ptr.read((char*) &cur, sizeof(int));
        cs_ptr.get()[i] = cur;

        if (i > 0) { max_nnz_in_one_dim = std::max<long>(max_nnz_in_one_dim, cur - prev); }
    }
}

SparseMatrix SparseMatrix::get_shallow_transpose() {
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
