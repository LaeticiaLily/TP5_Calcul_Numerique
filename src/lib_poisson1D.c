/**********************************************/
/* lib_poisson1D.c                            */
/* Numerical library developed to solve 1D    */ 
/* Poisson problem (Heat equation)            */
/**********************************************/
#include "lib_poisson1D.h"

void set_GB_operator_colMajor_poisson1D(double* AB, int *lab, int *la, int *kv){
  int ii, jj, kk;
  for (jj=0;jj<(*la);jj++){
    kk = jj*(*lab);
    if (*kv>=0){
      for (ii=0;ii< *kv;ii++){
	      AB[kk+ii]=0.0;
      }
    }
    AB[kk+ *kv]=-1.0;
    AB[kk+ *kv+1]=2.0;
    AB[kk+ *kv+2]=-1.0;
  }
  AB[0]=0.0;
  if (*kv == 1) {AB[1]=0;}
  
  AB[(*lab)*(*la)-1]=0.0;
}

void set_GB_operator_colMajor_poisson1D_Id(double* AB, int *lab, int *la, int *kv){
  int ii, jj, kk;
  for (jj=0;jj<(*la);jj++){
    kk = jj*(*lab);
    if (*kv>=0){
      for (ii=0;ii< *kv;ii++){
	      AB[kk+ii]=0.0;
      }
    }
    AB[kk+ *kv]=0.0;
    AB[kk+ *kv+1]=1.0;
    AB[kk+ *kv+2]=0.0;
  }
  AB[1]=0.0;
  AB[(*lab)*(*la)-1]=0.0;
}

void set_dense_RHS_DBC_1D(double* RHS, int* la, double* BC0, double* BC1){
  int jj;
  RHS[0]= *BC0;
  RHS[(*la)-1]= *BC1;
  for (jj=1;jj<(*la)-1;jj++){
    RHS[jj]=0.0;
  }
}  

void set_analytical_solution_DBC_1D(double* EX_SOL, double* X, int* la, double* BC0, double* BC1){
  int jj;
  double h, DELTA_T;
  DELTA_T=(*BC1)-(*BC0);
  for (jj=0;jj<(*la);jj++){
    EX_SOL[jj] = (*BC0) + X[jj]*DELTA_T;
  }
}  

void set_grid_points_1D(double* x, int* la){
  int jj;
  double h;
  h=1.0/(1.0*((*la)+1));
  for (jj=0;jj<(*la);jj++){
    x[jj]=(jj+1)*h;
  }
}

double relative_forward_error(double* x, double* y, int* la){
  double temp, relres;
  temp = cblas_ddot(*la, x, 1, x,1);
  temp = sqrt(temp);
  cblas_daxpy(*la, -1.0, x, 1, y, 1);
  relres = cblas_ddot(*la, y, 1, y,1);
  relres = sqrt(relres);
  relres = relres / temp;
  return relres;
}

int indexABCol(int i, int j, int *lab){
  return j*(*lab)+i;
}

/* LU tridiagonale (pas de pivot) :
   - L stocké dans la sous-diagonale (ligne 3)
   - U diag stockée en ligne 2
   - U super stockée en ligne 1 (dans colonne suivante)
*/
int dgbtrftridiag(int *la, int *n, int *kl, int *ku,
                  double *AB, int *lab, int *ipiv, int *info)
{
  int N = *la, LD = *lab;
  (void)n; (void)kl; (void)ku;
  *info = 0;

  for (int i = 0; i < N; i++) ipiv[i] = i + 1;

  for (int j = 0; j < N - 1; j++) {
    int kk  = j * LD;
    int kk1 = (j + 1) * LD;

    double piv = AB[kk + 2];
    if (piv == 0.0) { *info = j + 1; return *info; }

    AB[kk + 3] /= piv;                         /* L(j+1,j) */
    AB[kk1 + 2] -= AB[kk + 3] * AB[kk1 + 1];   /* U(j+1,j+1) */
  }
  return *info;
}
