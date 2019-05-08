#include "extras.h"

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