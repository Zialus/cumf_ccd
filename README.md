## How to build

    mkdir build && cd build
    cmake .. && make

## How to run

    $ ./exec/ccdp_gpu data/toy_example

(more examples below)

## Input format

The input should be in binary format (See toy_example directory for details).

"meta_modified_all" has the name of the input files.

Details:
- line 1 has #rows #cols
- line 2 has nnz in training dataset
- line 3 to 11 has file names
- line 4 has nnz in test dataset and test filename


## Arguments

    $ ./ccdp_gpu [options] [input file directoty containing meta_modified_all]

    options:
        -k rank/feature : set the rank (default 10)
        -l lambda : set the regularization parameter lambda (default 0.05)
        -a tile size: set tile size for input matrix R (default 499999999)
        -b tile size: set tile size for input matrix R Transpose (default 499999999)
        -t max_iter: number of iterations (default 5)
        -T max_iter: number of inner iterations (default 1)


## Examples:

To run Netflix:

    $ ./ccdp_gpu -T 1 -a 100000 -b 100000 -l .058 -k 40 -t 10 ../../../DATASETS/netflix

To run Yahoo Music:

    $ ./ccdp_gpu -T 1 -a 100000 -b 100000 -l 1.2 -k 40 -t 10 ../../../DATASETS/yahooc15/

To run Movielens:

    $ ./ccdp_gpu -T 1 -a 100000 -b 100000 -l .05 -k 40 -t 15 ../../../DATASETS/ml20M/
