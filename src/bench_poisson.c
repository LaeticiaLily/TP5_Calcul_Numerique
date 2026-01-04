
#include "lib_poisson1D.h"
#include <lapacke.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void die_alloc(void *p, const char *name) {
    if (!p) { fprintf(stderr, "Allocation failed: %s\n", name); exit(1); }
}

static double bench_lapacke_dgbtrf_dgbtrs(int la, int nrepeats, int nwarmup) {
    // Band parameters for 1D Poisson tridiagonal matrix
    int kv = 1;
    int ku = 1;
    int kl = 1;
    int lab = kv + kl + ku + 1; // leading dimension in band storage

    // Allocate base operator + base RHS
    double *AB0  = (double*)malloc(sizeof(double) * (size_t)lab * (size_t)la);
    double *RHS0 = (double*)malloc(sizeof(double) * (size_t)la);
    die_alloc(AB0, "AB0");
    die_alloc(RHS0, "RHS0");

    set_GB_operator_colMajor_poisson1D(AB0, &lab, &la, &kv);

    double BC0 = 0.0, BC1 = 0.0;
    set_dense_RHS_DBC_1D(RHS0, &la, &BC0, &BC1);

    // Work buffers (copied each repeat)
    double *AB  = (double*)malloc(sizeof(double) * (size_t)lab * (size_t)la);
    double *RHS = (double*)malloc(sizeof(double) * (size_t)la);
    int *ipiv   = (int*)malloc(sizeof(int) * (size_t)la);
    die_alloc(AB, "AB");
    die_alloc(RHS, "RHS");
    die_alloc(ipiv, "ipiv");

    // Warm-up (not timed)
    for (int w = 0; w < nwarmup; ++w) {
        memcpy(AB,  AB0,  sizeof(double) * (size_t)lab * (size_t)la);
        memcpy(RHS, RHS0, sizeof(double) * (size_t)la);

        int info = LAPACKE_dgbtrf(LAPACK_COL_MAJOR, la, la, kl, ku, AB, lab, ipiv);
        if (info == 0) {
            info = LAPACKE_dgbtrs(LAPACK_COL_MAJOR, 'N', la, kl, ku, 1, AB, lab, ipiv, RHS, la);
        }
        (void)info;
    }

    // Timed repeats
    double total = 0.0;
    for (int r = 0; r < nrepeats; ++r) {
        memcpy(AB,  AB0,  sizeof(double) * (size_t)lab * (size_t)la);
        memcpy(RHS, RHS0, sizeof(double) * (size_t)la);

        double t0 = now_s();
        int info = LAPACKE_dgbtrf(LAPACK_COL_MAJOR, la, la, kl, ku, AB, lab, ipiv);
        if (info == 0) {
            info = LAPACKE_dgbtrs(LAPACK_COL_MAJOR, 'N', la, kl, ku, 1, AB, lab, ipiv, RHS, la);
        }
        double t1 = now_s();
        total += (t1 - t0);
    }

    free(AB0); free(RHS0);
    free(AB); free(RHS); free(ipiv);
    return total / (double)nrepeats;
}

static double bench_tridiag_lu_factor(int la, int nrepeats, int nwarmup) {
    int kv = 1;
    int ku = 1;
    int kl = 1;
    int lab = kv + kl + ku + 1;

    double *AB0  = (double*)malloc(sizeof(double) * (size_t)lab * (size_t)la);
    double *RHS0 = (double*)malloc(sizeof(double) * (size_t)la);
    die_alloc(AB0, "AB0");
    die_alloc(RHS0, "RHS0");

    set_GB_operator_colMajor_poisson1D(AB0, &lab, &la, &kv);

    double BC0 = 0.0, BC1 = 0.0;
    set_dense_RHS_DBC_1D(RHS0, &la, &BC0, &BC1);

    double *AB  = (double*)malloc(sizeof(double) * (size_t)lab * (size_t)la);
    double *RHS = (double*)malloc(sizeof(double) * (size_t)la);
    int *ipiv   = (int*)malloc(sizeof(int) * (size_t)la);
    die_alloc(AB, "AB");
    die_alloc(RHS, "RHS");
    die_alloc(ipiv, "ipiv");

    // Warm-up
    for (int w = 0; w < nwarmup; ++w) {
        memcpy(AB,  AB0,  sizeof(double) * (size_t)lab * (size_t)la);
        memcpy(RHS, RHS0, sizeof(double) * (size_t)la);

        int info = 0;
        dgbtrftridiag(&la, &la, &kl, &ku, AB, &lab, ipiv, &info);
        if (info == 0) {
            // backsolve using LAPACK (AB contains LU factors)
            LAPACKE_dgbtrs(LAPACK_COL_MAJOR, 'N', la, kl, ku, 1, AB, lab, ipiv, RHS, la);
        }
    }

    double total = 0.0;
    for (int r = 0; r < nrepeats; ++r) {
        memcpy(AB,  AB0,  sizeof(double) * (size_t)lab * (size_t)la);
        memcpy(RHS, RHS0, sizeof(double) * (size_t)la);

        double t0 = now_s();
        int info = 0;
        dgbtrftridiag(&la, &la, &kl, &ku, AB, &lab, ipiv, &info);
        if (info == 0) {
            LAPACKE_dgbtrs(LAPACK_COL_MAJOR, 'N', la, kl, ku, 1, AB, lab, ipiv, RHS, la);
        }
        double t1 = now_s();
        total += (t1 - t0);
    }

    free(AB0); free(RHS0);
    free(AB); free(RHS); free(ipiv);
    return total / (double)nrepeats;
}

int main(void) {
    // Tailles à tester
    int sizes[] = {100, 150, 200, 300, 500, 800, 1200, 2000, 3500, 5000, 7500, 10000};
    int nsizes = (int)(sizeof(sizes)/sizeof(sizes[0]));

    // Paramètres de bench  
    const int nwarmup  = 5;
    const int nrepeats = 50;

    FILE *f = fopen("timings.csv", "w");
    if (!f) { perror("timings.csv"); return 1; }
    fprintf(f, "n,method,time_s\n");

    for (int i = 0; i < nsizes; ++i) {
        int la = sizes[i];

        double t_lap = bench_lapacke_dgbtrf_dgbtrs(la, nrepeats, nwarmup);
        fprintf(f, "%d,LAPACKE_DGBTRF_DGBTRS,%.9f\n", la, t_lap);

        double t_tri = bench_tridiag_lu_factor(la, nrepeats, nwarmup);
        fprintf(f, "%d,TRIDIAG_LU_FACTOR,%.9f\n", la, t_tri);

        fflush(f);
    }

    fclose(f);
    return 0;
}
