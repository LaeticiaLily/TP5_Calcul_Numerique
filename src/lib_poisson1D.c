/**********************************************/
/* lib_poisson1D.c                            */
/* Numerical library developed to solve 1D    */ 
/* Poisson problem (Heat equation)            */
/**********************************************/
#include "lib_poisson1D.h"


void set_GB_operator_colMajor_poisson1D(double* AB, int *lab, int *la, int *kv){
  int n    = *la;
  int lab_ = *lab;

  for (int j = 0; j < n; j++) {
    AB[indexABCol(0, j, lab)] = 0.0;   //ligne 0 : toujours 0 (workspace) //

    /* ligne 1 : sur-diagonale (A(j-1,j)) */
    if (j >0)
      AB[indexABCol(1, j, lab)] = -1.0;
    else
      AB[indexABCol(1, j, lab)] = 0.0;

    
    AB[indexABCol(2,j, lab)] = 2.0;   // ligne 2 : diagonale principale //

    /* ligne 3 : sous-diagonale (A(j+1,j)) */
    if (j < n - 1)
      AB[indexABCol(3, j, lab)] = -1.0;
    else
      AB[indexABCol(3, j, lab)] = 0.0;
  }
}


void set_GB_operator_colMajor_poisson1D_Id(double* AB, int *lab, int *la, int *kv){
  int n = *la;
  int lab_ = *lab;

  /* Tout à 0 */
  for (int j = 0; j< n; j++) {
    for (int i = 0; i < lab_; i++) {AB[indexABCol(i, j, lab)] = 0.0;}
  }
  /* Diagonale = 1 */
  for (int j = 0; j < n; j++) {
    AB[indexABCol(2, j, lab)] = 1.0;  // ligne 2 = diag
  }
}


void set_dense_RHS_DBC_1D(double* RHS, int* la, double* BC0, double* BC1){
  int n = *la;

  for (int i = 0; i < n; i++) { RHS[i] = 0.0;}
  if (n > 0) {
    RHS[0] += *BC0;
    RHS[n - 1] += *BC1;
  }
}




void set_analytical_solution_DBC_1D(double* EX_SOL, double* X, int* la, double* BC0, double* BC1){
  int n = *la;
  double T0 = *BC0;
  double T1 = *BC1;
  double dT = T1 - T0 ;

  for (int i = 0; i < n; i++) {
    double x = X[i];
    EX_SOL[i] = T0 + x * dT;
  }
}



  void set_grid_points_1D(double* x, int* la){
  int n = *la;
  double h = 1.0 /(n + 1);

  for (int i = 0; i < n; i++) { x[i] = (i + 1) * h;}
}



double relative_forward_error(double* x, double* y, int* la){
  // TODO: Compute the relative error using BLAS functions (dnrm2, daxpy or manual loop)

  int n = *la;
  double num2 =0.0; // ||x - y||^2
  double den2 = 0.0; // ||x||^2

  for (int i = 0; i < n; i++) {
    double diff = x[i] - y[i];
    num2 += diff * diff;
    den2 += x[i] * x[i];
  }

  if (den2== 0.0)
    return 0.0 ;

  return sqrt(num2)/ sqrt(den2);


  return 0.0;
}



int indexABCol(int i, int j,int *lab){
    return j*(*lab)+ i;
}





int dgbtrftridiag(int *la, int*n, int *kl, int *ku, double *AB, int *lab, int *ipiv, int *info){
  // TODO: Implement specialized LU factorization for tridiagonal matrices
  int N = *la;      // taille
  int lab_ = *lab;

  (void)n;   // on ne les utilise pas, mais on les garde dans la signature
  (void)kl;
  (void)ku;
  (void)lab_ ;

  *info = 0 ;

  //ipiv(i) = i+1 (pas de pivot) 
  for (int i= 0; i< N; i++) {
    ipiv[i] = i + 1;
  }

   //indices de lignes dans AB : sur=1, diag=2, sous=3 
  int super = 1;
  int diag= 2 ;
  int sub = 3;

  //LU tridiagonale 
  for (int j = 0; j< N - 1; j++){

    int idx_diag_j = indexABCol(diag, j,   lab); // a_{j,j}
    int idx_sub_j = indexABCol(sub,  j,  lab );       // a_{j+1,j}
    int idx_diag_jp1 = indexABCol(diag, j+1, lab);  // a_{j+1,j+1}
    int idx_super_jp1 = indexABCol(super, j+1, lab );   // a_{j,j+1}

    double ajj =AB[idx_diag_j];

    if (ajj== 0.0) {
      *info = j + 1; // pivot nul, convention LAPACK (1-based)
      return *info;
    }

    //L_{j+1,j} 
    double lij = AB[idx_sub_j]/ ajj;
    AB[idx_sub_j] = lij;

    
    AB[idx_diag_jp1] -= lij * AB[idx_super_jp1]; // mise à jour de U_{j+1,j+1} 
  }

  return *info;
}
