/******************************************/
/* tp_poisson1D_iter.c                    */
/* This file contains the main function   */
/* to solve the Poisson 1D problem using  */
/* iterative methods (Richardson)         */
/******************************************/
#include "lib_poisson1D.h"

#define ALPHA 0  /* Richardson iteration with optimal alpha */
#define JAC 1    /* Richardson with Jacobi preconditioning */
#define GS 2     /* Richardson with Gauss-Seidel preconditioning */

/**
 * Main function to solve the 1D Poisson equation using iterative methods.
 * 
 * @param argc: Number of command-line arguments
 * @param argv: Array of argument strings
 *              argv[1] (optional): Method selection (0=ALPHA, 1=JAC, 2=GS)
 * @return 0 on success
 */
int main(int argc,char *argv[])
/* ** argc: Number of arguments */
/* ** argv: Values of arguments */
{
  int ierr;                           /* Error code */
  int jj;                             /* Loop counter */
  int nbpoints, la;                   /* nbpoints: total points, la: interior points */
  int ku, kl, lab, kv;               /* Band matrix parameters */
  int *ipiv;                          /* Pivot indices (unused in iterative methods) */
  int info;                           /* Info parameter */
  int NRHS;                           /* Number of right-hand sides */
  int IMPLEM = 0;                     /* Implementation method (ALPHA, JAC, or GS) */
  double T0, T1;                      /* Boundary conditions */
  double *RHS, *SOL, *EX_SOL, *X;     /* RHS, solution, exact solution, grid points */
  double *AB;                         /* Coefficient matrix */
  double *MB;                         /* Preconditioner matrix */
  
  double temp, relres;                /* Temporary variable and relative residual */

  double opt_alpha;                   /* Optimal relaxation parameter */

  if (argc == 2) {
    IMPLEM = atoi(argv[1]);
  } else if (argc > 2) {
    perror("Application takes at most one argument");
    exit(1);
  }

  /* Problem size setup */
  NRHS=1;           /* Single right-hand side */
  nbpoints=12;      /* Total discretization points */
  la=nbpoints-2;    /* Interior points only */

  /* Dirichlet Boundary conditions */
  T0=5.0;           /* Left boundary value */
  T1=20.0;          /* Right boundary value */

  printf("--------- Poisson 1D ---------\n\n");
  /* Allocate memory for vectors */
  RHS=(double *) malloc(sizeof(double)*la);       /* Right-hand side */
  SOL=(double *) calloc(la, sizeof(double));      /* Solution (initialized to 0) */
  EX_SOL=(double *) malloc(sizeof(double)*la);    /* Exact solution */
  X=(double *) malloc(sizeof(double)*la);         /* Grid points */

  /* Setup the Poisson 1D problem */
  /* General Band Storage */
  set_grid_points_1D(X, &la);                              /* Generate uniform grid */
  set_dense_RHS_DBC_1D(RHS,&la,&T0,&T1);                  /* Set RHS with BC */
  set_analytical_solution_DBC_1D(EX_SOL, X, &la, &T0, &T1); /* Compute exact solution */
  
  /* Write initial data to files */
  write_vec(RHS, &la, "RHS.dat");
  write_vec(EX_SOL, &la, "EX_SOL.dat");
  write_vec(X, &la, "X_grid.dat");

  /* Set up band storage parameters */
  kv=0;             /* No extra space needed for problem construction */
  ku=1;             /* One superdiagonal */
  kl=1;             /* One subdiagonal */
  lab=kv+kl+ku+1;   /* Leading dimension of band storage */
  
  /* Allocate and initialize coefficient matrix */
  AB = (double *) malloc(sizeof(double)*lab*la);
  set_GB_operator_colMajor_poisson1D(AB, &lab, &la, &kv);
  
  /* uncomment the following to check matrix A */
  write_GB_operator_colMajor_poisson1D(AB, &lab, &la, "AB.dat");
  
  /********************************************/
  /* Solution (Richardson with optimal alpha) */

  /* Computation of optimum alpha */
  opt_alpha = richardson_alpha_opt(&la);
  printf("Optimal alpha for simple Richardson iteration is : %lf",opt_alpha); 

  /* Solve with Richardson iteration parameters */
  double tol=1e-3;      /* Convergence tolerance for residual norm */
  int maxit=1000;       /* Maximum number of iterations */
  double *resvec;       /* Array to store residual history */
  int nbite=0;          /* Number of iterations performed */

  resvec=(double *) calloc(maxit, sizeof(double));

  /* Solve with Richardson alpha (simple Richardson with optimal alpha) */
  if (IMPLEM == ALPHA) {
    richardson_alpha(AB, RHS, SOL, &opt_alpha, &lab, &la, &ku, &kl, &tol, &maxit, resvec, &nbite);
  }

  /* Richardson General Tridiag (Preconditioned methods) */

  /* get MB (:=M, D for Jacobi, (D-E) for Gauss-seidel) */
  kv = 1;               /* Need extra space for factorization */
  ku = 1;
  kl = 1;
  MB = (double *) malloc(sizeof(double)*(lab)*la);
  
  /* Extract preconditioner matrix based on method */
  if (IMPLEM == JAC) {
    /* Jacobi: MB = D (diagonal of A) */
    extract_MB_jacobi_tridiag(AB, MB, &lab, &la, &ku, &kl, &kv);
  } else if (IMPLEM == GS) {
    /* Gauss-Seidel: MB = D - E (lower triangular + diagonal) */
    extract_MB_gauss_seidel_tridiag(AB, MB, &lab, &la, &ku, &kl, &kv);
  }

  /* Solve with General Richardson (preconditioned) */
  if (IMPLEM == JAC || IMPLEM == GS) {
    write_GB_operator_colMajor_poisson1D(MB, &lab, &la, "MB.dat");
    richardson_MB(AB, RHS, SOL, MB, &lab, &la, &ku, &kl, &tol, &maxit, resvec, &nbite);
  }

  /* Write solution and convergence history to files */
  write_vec(SOL, &la, "SOL.dat");              /* Final solution */

  /* Write convergence history */
  write_vec(resvec, &nbite, "RESVEC.dat");     /* Residual norm at each iteration */

  /* Free allocated memory */
  free(resvec);
  free(RHS);
  free(SOL);
  free(EX_SOL);
  free(X);
  free(AB);
  free(MB);
  printf("\n\n--------- End -----------\n");
}
