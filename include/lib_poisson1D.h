/**********************************************/
/* lib_poisson1D.h                            */
/* Header for Numerical library developed to  */ 
/* solve 1D Poisson problem (Heat equation)   */
/**********************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include "atlas_headers.h"

/**
 * Set up the Poisson 1D operator in General Band (GB) storage format with column-major ordering
 * @param AB: Output matrix in GB format (allocated with size lab*la)
 * @param lab: Leading dimension of AB (number of rows in the band storage)
 * @param la: Number of columns in the matrix (problem size)
 * @param kv: Number of superdiagonals in the band storage
 */
void set_GB_operator_colMajor_poisson1D(double* AB, int* lab, int *la, int *kv);

/**
 * Set up the identity operator in General Band (GB) storage format with column-major ordering
 * @param AB: Output matrix in GB format (allocated with size lab*la)
 * @param lab: Leading dimension of AB (number of rows in the band storage)
 * @param la: Number of columns in the matrix (problem size)
 * @param kv: Number of superdiagonals in the band storage
 */
void set_GB_operator_colMajor_poisson1D_Id(double* AB, int* lab, int *la, int *kv);

/**
 * Set up the right-hand side (RHS) vector for 1D Poisson problem with Dirichlet boundary conditions
 * @param RHS: Output right-hand side vector (allocated with size la)
 * @param la: Problem size (number of interior grid points)
 * @param BC0: Boundary condition at x=0 (left boundary)
 * @param BC1: Boundary condition at x=1 (right boundary)
 */
void set_dense_RHS_DBC_1D(double* RHS, int* la, double* BC0, double* BC1);

/**
 * Compute the analytical solution for 1D Poisson problem with Dirichlet boundary conditions
 * @param EX_SOL: Output analytical solution vector (allocated with size la)
 * @param X: Grid points where solution is computed (size la)
 * @param la: Problem size (number of interior grid points)
 * @param BC0: Boundary condition at x=0 (left boundary)
 * @param BC1: Boundary condition at x=1 (right boundary)
 */
void set_analytical_solution_DBC_1D(double* EX_SOL, double* X, int* la, double* BC0, double* BC1);

/**
 * Generate uniformly spaced grid points in [0,1]
 * @param x: Output array of grid points (allocated with size la)
 * @param la: Number of interior grid points
 */
void set_grid_points_1D(double* x, int* la);

/**
 * Compute the relative forward error between two vectors
 * @param x: Computed solution vector (size la)
 * @param y: Reference/exact solution vector (size la)
 * @param la: Vector size
 * @return Relative forward error ||x-y||/||x||
 */
double relative_forward_error(double* x, double* y, int* la);

/**
 * Write the GB operator matrix in AIJ (triplet) format to a file
 * @param AB: Matrix in GB storage format
 * @param la: Problem size
 * @param filename: Output filename
 */
void write_GB2AIJ_operator_poisson1D(double* AB, int *la, char* filename);

/**
 * Write the GB operator matrix in row-major format to a file
 * @param AB: Matrix in GB storage format
 * @param lab: Leading dimension of AB
 * @param la: Problem size
 * @param filename: Output filename
 */
void write_GB_operator_rowMajor_poisson1D(double* AB, int* lab, int *la, char* filename);

/**
 * Write the GB operator matrix in column-major format to a file
 * @param AB: Matrix in GB storage format
 * @param lab: Leading dimension of AB
 * @param la: Problem size
 * @param filename: Output filename
 */
void write_GB_operator_colMajor_poisson1D(double* AB, int* lab, int* la, char* filename);

/**
 * Write a vector to a file (one value per line)
 * @param vec: Vector to write (size la)
 * @param la: Vector size
 * @param filename: Output filename
 */
void write_vec(double* vec, int* la, char* filename);

/**
 * Write two vectors as x-y pairs to a file (tab-separated)
 * @param vec: Y-values vector (size la)
 * @param x: X-values vector (size la)
 * @param la: Vector size
 * @param filename: Output filename
 */
void write_xy(double* vec, double* x, int* la, char* filename);

/**
 * Compute all eigenvalues of the 1D Poisson operator
 * @param eigval: Output array of eigenvalues (allocated with size la)
 * @param la: Problem size
 */
void eig_poisson1D(double* eigval, int *la);

/**
 * Compute the maximum eigenvalue of the 1D Poisson operator
 * @param la: Problem size
 * @return Maximum eigenvalue
 */
double eigmax_poisson1D(int *la);

/**
 * Compute the minimum eigenvalue of the 1D Poisson operator
 * @param la: Problem size
 * @return Minimum eigenvalue
 */
double eigmin_poisson1D(int *la);

/**
 * Compute the optimal relaxation parameter alpha for Richardson iteration
 * @param la: Problem size
 * @return Optimal alpha value
 */
double richardson_alpha_opt(int *la);

/**
 * Solve linear system using Richardson iteration with fixed relaxation parameter alpha
 * @param AB: Coefficient matrix in GB storage format
 * @param RHS: Right-hand side vector (size la)
 * @param X: Solution vector (size la, input: initial guess, output: solution)
 * @param alpha_rich: Relaxation parameter alpha
 * @param lab: Leading dimension of AB
 * @param la: Problem size
 * @param ku: Number of superdiagonals
 * @param kl: Number of subdiagonals
 * @param tol: Convergence tolerance for residual norm
 * @param maxit: Maximum number of iterations
 * @param resvec: Output residual history (allocated with size maxit)
 * @param nbite: Output number of iterations performed
 */
void richardson_alpha(double *AB, double *RHS, double *X, double *alpha_rich, int *lab, int *la,int *ku, int*kl, double *tol, int *maxit, double *resvec, int *nbite);

/**
 * Extract the preconditioner matrix for Jacobi method from tridiagonal matrix
 * @param AB: Input matrix in GB storage format
 * @param MB: Output preconditioner matrix (diagonal of AB) in GB format
 * @param lab: Leading dimension of AB
 * @param la: Problem size
 * @param ku: Number of superdiagonals
 * @param kl: Number of subdiagonals
 * @param kv: Number of superdiagonals in output MB
 */
void extract_MB_jacobi_tridiag(double *AB, double *MB, int *lab, int *la,int *ku, int*kl, int *kv);

/**
 * Extract the preconditioner matrix for Gauss-Seidel method from tridiagonal matrix
 * @param AB: Input matrix in GB storage format
 * @param MB: Output preconditioner matrix (lower triangular + diagonal) in GB format
 * @param lab: Leading dimension of AB
 * @param la: Problem size
 * @param ku: Number of superdiagonals
 * @param kl: Number of subdiagonals
 * @param kv: Number of superdiagonals in output MB
 */
void extract_MB_gauss_seidel_tridiag(double *AB, double *MB, int *lab, int *la,int *ku, int*kl, int *kv);

/**
 * Solve linear system using preconditioned Richardson iteration
 * @param AB: Coefficient matrix in GB storage format
 * @param RHS: Right-hand side vector (size la)
 * @param X: Solution vector (size la, input: initial guess, output: solution)
 * @param MB: Preconditioner matrix in GB format
 * @param lab: Leading dimension of AB
 * @param la: Problem size
 * @param ku: Number of superdiagonals
 * @param kl: Number of subdiagonals
 * @param tol: Convergence tolerance for residual norm
 * @param maxit: Maximum number of iterations
 * @param resvec: Output residual history (allocated with size maxit)
 * @param nbite: Output number of iterations performed
 */
void richardson_MB(double *AB, double *RHS, double *X, double *MB, int *lab, int *la,int *ku, int*kl, double *tol, int *maxit, double *resvec, int *nbite);

/**
 * Compute the index in the band storage for element (i,j) in column-major format
 * @param i: Row index (0-based)
 * @param j: Column index (0-based)
 * @param lab: Leading dimension of the band storage
 * @return Linear index in the band storage array
 */
int indexABCol(int i, int j, int *lab);

/**
 * LU factorization for tridiagonal matrices (optimized version of dgbtrf)
 * @param la: Leading dimension of the matrix
 * @param n: Order of the matrix
 * @param kl: Number of subdiagonals (should be 1 for tridiagonal)
 * @param ku: Number of superdiagonals (should be 1 for tridiagonal)
 * @param AB: Matrix in GB format (input: matrix, output: LU factors)
 * @param lab: Leading dimension of AB
 * @param ipiv: Pivot indices array (size min(la,n))
 * @param info: Output info (0: success, >0: singular matrix)
 * @return info value
 */
int dgbtrftridiag(int *la, int *n, int *kl, int *ku, double *AB, int *lab, int *ipiv, int *info);
