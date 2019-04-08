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
#include "utilities.h"

void print_help_and_exit() {
    printf(
            "options:\n"
            "    -k rank/feature : set the rank (default 10)\n"
            "    -l lambda : set the regularization parameter lambda (default 0.05)\n"
            "    -a tile size: set tile size for input matrix R (default 499999999)\n"
            "    -b tile size: set tile size for input matrix R Transpose (default 499999999)\n"
            "    -t max_iter: number of iterations (default 5)\n"
            "    -T max_iter: number of inner iterations (default 1)\n"
    );
    exit(EXIT_FAILURE);
}

Options parse_cmd_options(int argc, char** argv) {
    Options param;

    int i;
    for (i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            break;
        }
        if (++i >= argc) {
            print_help_and_exit();
        }
        switch (argv[i - 1][1]) {
            case 'k':
                param.k = atoi(argv[i]);
                break;
            case 'l':
                param.lambda = (DTYPE) atof(argv[i]);
                break;
            case 't':
                param.maxiter = atoi(argv[i]);
                break;
            case 'T':
                param.maxinneriter = atoi(argv[i]);
                break;
            case 'a':
                param.tileSizeH = atoi(argv[i]);
                break;
            case 'b':
                param.tileSizeW = atoi(argv[i]);
                break;
            default:
                fprintf(stderr, "unknown option: -%c\n", argv[i - 1][1]);
                print_help_and_exit();
                break;
        }
    }

    if (i >= argc) {
        print_help_and_exit();
    }

    snprintf(param.data_directory, 1024, "%s", argv[i]);
    return param;
}

void run_ccdr1(Options& param, const char* data_directory) {
    SparseMatrix R;
    TestData T;

    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "[info] Loading R matrix..." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    load_from_binary(data_directory, R, T);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> deltaT = t1 - t0;
    std::cout << "[info] Loading rating data time: " << deltaT.count() << "s.\n";
    std::cout << "------------------------------------------------------" << std::endl;

    MatData W;
    MatData H;
    init_random(W, param.k, R.rows_);
    init_random(H, param.k, R.cols_);

    MatData W_ref;
    MatData H_ref;
    init_random(W_ref, param.k, R.rows_);
    init_random(H_ref, param.k, R.cols_);

    printf("Computing with CCD!!\n");
    auto t2 = std::chrono::high_resolution_clock::now();
    ccdr1(R, W, H, T, param);
    auto t3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> deltaT23 = t3 - t2;
    std::cout << "[info] CUDA Predict Time: " << deltaT23.count() << " s.\n";

    std::cout << "------------------------------------------------------" << std::endl;
    auto t13 = std::chrono::high_resolution_clock::now();
    cdmf_ref(R, W_ref, H_ref, T, param);
    auto t14 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> deltaT13_14 = t14 - t13;
    std::cout << "[info] OMP Predict Time: " << deltaT13_14.count() << " s.\n";


    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "[info] validate the results." << std::endl;
    auto t11 = std::chrono::high_resolution_clock::now();
    golden_compare(W, W_ref, param.k, R.rows_);
    golden_compare(H, H_ref, param.k, R.cols_);
    auto t12 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> deltaT11_12 = t12 - t11;
    std::cout << "[info] Validate Time: " << deltaT11_12.count() << " s.\n";

//    print_matrix(W, param.k, R.rows_);
//    printf("\n");
//    print_matrix(H, param.k, R.cols_);
//    printf("\n");
//    print_matrix(W_ref, param.k, R.rows_);
//    printf("\n");
//    print_matrix(H_ref, param.k, R.cols_);

}

int main(int argc, char* argv[]) {
    Options options = parse_cmd_options(argc, argv);
    options.print();

    run_ccdr1(options, options.data_directory);
    return EXIT_SUCCESS;
}
