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
#include <cstdio>

void load_from_binary(const char* srcdir, SparseMatrix& R, TestData& T) {
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

    char binary_filename_val_test[1024];
    char binary_filename_row_test[1024];
    char binary_filename_col_test[1024];

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

//    CHECK_FSCAN(fscanf(fp, "%1023s", buf), 1);
    snprintf(binary_filename_val_test, sizeof(binary_filename_val_test), "%s/R_test_coo.data.bin", srcdir);
//    CHECK_FSCAN(fscanf(fp, "%1023s", buf), 1);
    snprintf(binary_filename_row_test, sizeof(binary_filename_row_test), "%s/R_test_coo.row.bin", srcdir);
//    CHECK_FSCAN(fscanf(fp, "%1023s", buf), 1);
    snprintf(binary_filename_col_test, sizeof(binary_filename_col_test), "%s/R_test_coo.col.bin", srcdir);


    auto t0 = std::chrono::high_resolution_clock::now();
    R.initialize_matrix(m, n, nnz);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> deltaT = t1 - t0;
    std::cout << "[info] Alloc TIMER: " << deltaT.count() << "s.\n";


    auto t2 = std::chrono::high_resolution_clock::now();

    R.read_binary_file(binary_filename_rowptr, binary_filename_colidx, binary_filename_csrval,
                       binary_filename_colptr, binary_filename_rowidx, binary_filename_cscval);
    auto t3 = std::chrono::high_resolution_clock::now();
    deltaT = t3 - t2;
    std::cout << "[info] Train TIMER: " << deltaT.count() << "s.\n";

    auto t4 = std::chrono::high_resolution_clock::now();

    if (fscanf(fp, "%ld %1023s", &nnz, buf) != EOF) {
        snprintf(filename, sizeof(filename), "%s/%s", srcdir, buf);
        T.read_binary_file(m, n, nnz, binary_filename_val_test, binary_filename_row_test, binary_filename_col_test);
    }

    auto t5 = std::chrono::high_resolution_clock::now();
    deltaT = t5 - t4;
    std::cout << "[info] Tests TIMER: " << deltaT.count() << "s.\n";


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
