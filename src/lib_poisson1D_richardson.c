/**********************************************/
/* lib_poisson1D_richardson.c                           */
/* Numerical library developed to solve 1D    */ 
/* Poisson problem (Heat equation)            */
/**********************************************/
#include "lib_poisson1D.h"
#include <math.h>
#include <stdlib.h>



/* ------------------------------------------------------------ */
/* BLAS / LAPACK Fortran interfaces (suffix _)                   */
/* ------------------------------------------------------------ */
extern void dgbmv_(char *TRANS, int *M, int *N, int *KL, int *KU,
                   double *ALPHA, double *AB, int *LDAB,
                   double *X, int *INCX, double *BETA, double *Y, int *INCY);

extern double dnrm2_(int *N, double *X, int *INCX);
extern void dcopy_(int *N, double *X, int *INCX, double *Y, int *INCY);
extern void daxpy_(int *N, double *ALPHA, double *X, int *INCX, double *Y, int *INCY);




#define M_PI 3.14159265358979323846

void eig_poisson1D(double* eigval, int *la)
{
  int n = *la;
  for (int k = 1; k <= n; k++) {
    double theta = (k * M_PI) / (n + 1.0);
    eigval[k - 1] = 2.0 - 2.0 * cos(theta);
  }
}

double eigmax_poisson1D(int *la)
{
  int n = *la;
  /* max = lambda_n */
  double theta = (n * M_PI) / (n + 1.0);
  return 2.0 - 2.0 * cos(theta);
}

double eigmin_poisson1D(int *la) 
{
  int n = *la;
  /* min = lambda_1 */
  double theta = (1.0 * M_PI) / (n + 1.0);
  return 2.0 - 2.0 * cos(theta);
}

double richardson_alpha_opt(int *la)
{
  double lmin = eigmin_poisson1D(la);
  double lmax = eigmax_poisson1D(la);
  return 2.0 / (lmin + lmax);
}



/**
 * Solve linear system Ax=b using Richardson iteration with fixed relaxation parameter alpha.
 * The iteration is: x^(k+1) = x^(k) + alpha*(b - A*x^(k))
 * Stops when ||b - A*x^(k)||_2  / ||b||_2 < tol or when reaching maxit iterations.
 */
void richardson_alpha(double *AB, double *RHS, double *X, double *alpha_rich, int *lab, int *la,int *ku, int*kl, double *tol, int *maxit, double *resvec, int *nbite){
  // TODO: Implement Richardson iteration
  // 1. Compute residual r = b - A*x (use dgbmv for matrix-vector product)
  // 2. Update x = x + alpha*r (use daxpy)
  // 3. Check convergence: ||r||_2 < tol (use dnrm2)
  // 4. Store residual norm in resvec and repeat

  int n   = *la;
  int ld  = *lab;
  int inc = 1;

  double *Ax = (double*)malloc((size_t)n * sizeof(double));
  double *r  = (double*)malloc((size_t)n * sizeof(double));
  if (!Ax || !r) {
    free(Ax); free(r);
    *nbite = 0;
    return;
  }

  double bnorm = dnrm2_(&n, RHS, &inc);
  if (bnorm == 0.0) bnorm = 1.0;

  char trans = 'N';
  double one = 1.0, zero = 0.0, minus_one = -1.0;
  double alpha = *alpha_rich;

  int k;
  for (k = 0; k < *maxit; ++k) {

    /* Ax = A * x (GB + BLAS) */
    dgbmv_(&trans, &n, &n, kl, ku, &one, AB, &ld, X, &inc, &zero, Ax, &inc);

    /* r = b - Ax */
    dcopy_(&n, RHS, &inc, r, &inc);
    daxpy_(&n, &minus_one, Ax, &inc, r, &inc);

    /* res = ||r|| / ||b|| */
    double rnorm = dnrm2_(&n, r, &inc);
    resvec[k] = rnorm / bnorm;

    if (resvec[k] < *tol) { k++; break; }

    /* x = x + alpha * r */
    daxpy_(&n, &alpha, r, &inc, X, &inc);
  }

  *nbite = k;

  free(Ax);
  free(r);
}




/**
 * Extract MB for Jacobi method from tridiagonal matrix.
 * Such as the Jacobi iterative process is: x^(k+1) = x^(k) + D^(-1)*(b - A*x^(k))
 */
void extract_MB_jacobi_tridiag(double *AB, double *MB, int *lab, int *la,int *ku, int*kl, int *kv){
  // TODO: Extract diagonal elements from AB and store in MB
  // MB should contain only the diagonal of A
  (void)AB; (void)ku; (void)kl;

  int n  = *la;
  int ld = *lab;

  /* zero MB */
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < ld; ++i)
      MB[j*ld + i] = 0.0;

  /* keep only diagonal (Poisson 1D: 2) */
  int id_diag = (*kv) + 1;
  for (int j = 0; j < n; ++j)
    MB[j*ld + id_diag] = 2.0;
}

/**
 * Extract MB for Gauss-Seidel method from tridiagonal matrix.
 * Such as the Gauss-Seidel iterative process is: x^(k+1) = x^(k) + (D-E)^(-1)*(b - A*x^(k))
 */
void extract_MB_gauss_seidel_tridiag(double *AB, double *MB, int *lab, int *la,int *ku, int*kl, int *kv){
  // TODO: Extract diagonal and lower diagonal from AB
  // MB should contain the lower triangular part (including diagonal) of A
  (void)AB; (void)ku; (void)kl;

  int n  = *la;
  int ld = *lab;

  /* zero MB */
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < ld; ++i)
      MB[j*ld + i] = 0.0;

  int id_sub  = (*kv);      /* sub-diagonal row in GB */
  int id_diag = (*kv) + 1;  /* diagonal row in GB */

  for (int j = 0; j < n; ++j) {
    MB[j*ld + id_diag] = 2.0;
    MB[j*ld + id_sub]  = -1.0;
  }
  /* first column has no sub-diagonal */
  MB[0*ld + id_sub] = 0.0;
}

/**
 * Solve linear system Ax=b using preconditioned Richardson iteration.
 * The iteration is: x^(k+1) = x^(k) + M^(-1)*(b - A*x^(k))
 * where M is either D for Jacobi or (D-E) for Gauss-Seidel.
 * Stops when ||b - A*x^(k)||_2  / ||b||_2 < tol or when reaching maxit iterations.
 */
void richardson_MB(double *AB, double *RHS, double *X, double *MB, int *lab, int *la,int *ku, int*kl, double *tol, int *maxit, double *resvec, int *nbite){
  {
  int n   = *la;
  int ld  = *lab;
  int inc = 1;

  /* norme de b */
  double bnorm = dnrm2_(&n, RHS, &inc);
  if (bnorm == 0.0) bnorm = 1.0;

  double *Ax = (double*)malloc((size_t)n * sizeof(double));
  double *r  = (double*)malloc((size_t)n * sizeof(double));
  double *z  = (double*)malloc((size_t)n * sizeof(double));
  if (!Ax || !r || !z) {
    free(Ax); free(r); free(z);
    *nbite = 0;
    return;
  }

  /* Pour la matrice tridiagonale en format GB :
     si kv = 1 alors:
       - sous-diagonale = ligne kv (=1)
       - diagonale      = ligne kv+1 (=2)
  */
  int kv = 1;                 /* dans ce TP, kv vaut 1 */
  int id_sub  = kv;
  int id_diag = kv + 1;

  char trans = 'N';
  double one = 1.0, zero = 0.0, minus_one = -1.0;

  int k;
  for (k = 0; k < *maxit; ++k) {

    /* Ax = A*X */
    dgbmv_(&trans, &n, &n, kl, ku, &one, AB, &ld, X, &inc, &zero, Ax, &inc);

    /* r = RHS - Ax */
    dcopy_(&n, RHS, &inc, r, &inc);
    daxpy_(&n, &minus_one, Ax, &inc, r, &inc);

    /* res = ||r||/||b|| */
    double rnorm = dnrm2_(&n, r, &inc);
    resvec[k] = rnorm / bnorm;
    if (resvec[k] < *tol) { k++; break; }

    /* z = M^{-1} r
       - Jacobi: MB sous-diag = 0 -> z[i] = r[i]/diag
       - Gauss-Seidel: MB sous-diag = -1 -> substitution avant
    */
    z[0] = r[0] / MB[0*ld + id_diag];
    for (int i = 1; i < n; ++i) {
      double li = MB[i*ld + id_sub];     /* 0 (Jacobi) ou -1 (GS) */
      double di = MB[i*ld + id_diag];    /* 2 */
      z[i] = (r[i] - li * z[i-1]) / di;
    }

    /* X = X + z */
    daxpy_(&n, &one, z, &inc, X, &inc);
  }

  *nbite = k;

  free(Ax);
  free(r);
  free(z);
}
}