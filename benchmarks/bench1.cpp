#include <iostream>
//#define EAD_DEBUG       // to debug
#include "Ead/ead.hpp"
#include <cstdlib>
#include <ctime>
#include <iomanip>      // std::setprecision

using namespace std;
using namespace ead;

typedef ead::DFad<double, 30> adouble;

const double pi = 3.14159265;

int const dim  = 3;
int const Nqp  = 5;
int const Npts = 15;

// fictitious finite element code
double Phi[Npts][Nqp];
double dLPhi[3][Npts][Nqp];
double weight[Nqp];

template<class TensorType, class Double>
void invert_a(TensorType & a, Double & det);

double  X[Npts];

void print_jac(double J[])
{
  for (int i = 0; i < Npts; ++i)
  {
    for (int j = 0; j < Npts; ++j)
      cout << J[i*Npts + j] << " ";
    cout << endl;
  }
  cout << "\n\n";
}

void element_residue_ad(double u_[], double R_[], double J_[])
{
  adouble u[Npts],  R[Npts]; // ead
  
  for (int i = 0; i < Npts; ++i)
  {
    u[i].val() = (i+1.)/2.; // random values
    u[i].setDiff(i, Npts);
    R[i].setNumVars(Npts);
  }
  
  for (int q = 0; q < Nqp; ++q)
  {
    adouble uqp(u[0]*Phi[0][q]);
    
    for (int j = 1; j < Npts; ++j)
      uqp += u[j]*Phi[j][q];
    
    adouble sqrt_u (sqrt(uqp));

    for (int i = 0; i < Npts; ++i)
    {
      R[i] += (uqp + pow(sqrt_u,3) + sqrt_u) * Phi[i][q] * weight[q];
    }
  }
  
  for (int i = 0; i < Npts; ++i)
  {
    R_[i] = R[i].val();
    for (int j = 0; j < Npts; ++j)
      J_[i*Npts + j] = R[i].dx(j);
  }
  
}



inline void element_residue_exact(double u_[], double R_[], double J_[])
{
  double u[Npts], *R = R_, *J = J_;
  
  for (int i = 0; i < Npts; ++i)
  {
    u[i] = u_[i];
    R[i] = 0;
    for (int j = 0; j < Npts; ++j)
      J[i*Npts + j] = 0;
  }

  // residue
  for (int q = 0; q < Nqp; ++q)
  {
    double uqp(u[0]*Phi[0][q]);
    
    for (int j = 1; j < Npts; ++j)
      uqp += u[j]*Phi[j][q];

    double sqrt_u = sqrt(uqp);

    for (int i = 0; i < Npts; ++i)
    {
      
      R[i] += (uqp + pow(sqrt_u,3) + sqrt_u) * Phi[i][q] * weight[q];
      
      for (int j = 0; j < Npts; ++j)
      {
        J[i*Npts + j] += (1. + 1.5*sqrt_u + .5/sqrt_u ) * Phi[j][q] * Phi[i][q] * weight[q];
      }
    }
  }
  
  //volatile double lixo(RR[I%Npts]); // volatile to avoid optimization
}


int main(int argc, char *argv[])
{
  int N_iterations = int(1e+3);
  
  double u[Npts], R[Npts], J[Npts*Npts];
  
  if (argc > 1)
  {
    N_iterations = atoi(argv[1]);
  }
  
  srand(1234);
  
  // set variables with random numbers
  for (int q = 0; q < Nqp; ++q)
  {
    for (int i = 0; i < Npts; ++i)
    {
      Phi[i][q] = double(rand())/RAND_MAX;
      dLPhi[0][i][q] = double(rand())/RAND_MAX;
      dLPhi[1][i][q] = double(rand())/RAND_MAX;
      dLPhi[2][i][q] = double(rand())/RAND_MAX;
    }
    weight[q] = double(rand())/RAND_MAX;
  }

  for (int i = 0; i < Npts; ++i)
  {
    u[i] = (i+1.)/2.; // random values
    X[i] = double((i+1)*(i+2))/3.; // random values
  }

  // ============================================================
  
  clock_t begin = clock();
  
  for (int i = 0; i < N_iterations; ++i)
    element_residue_ad(u, R, J);

  clock_t end = clock();
  double elapsed_ad = double(end - begin) / CLOCKS_PER_SEC;
  
  cout << "EAD:\n";
  print_jac(J);
  
  // ============================================================

  begin = clock();
  
  for (int i = 0; i < N_iterations; ++i)
    element_residue_exact(u, R, J);

  end = clock();
  double elapsed_ex = double(end - begin) / CLOCKS_PER_SEC;
  
  cout << "EXACT:\n";
  print_jac(J);
  
  
  // ====================== TIMES ============================

  cout << "Time elapsed:\n";
  cout << "EAD: "; cout << std::setprecision(9) << elapsed_ad;
  cout << endl << endl;
    
  cout << "Time elapsed:\n";
  cout << "EXACT: "; cout << std::setprecision(9) << elapsed_ex;
  cout << endl << endl;
  

  cout << " ratio: " << elapsed_ad/elapsed_ex << endl << endl;

  // ============================================================

  //for (int i = 0; i < Npts; ++i)
  //{
  //  for (int j = 0; j < Npts; ++j)
  //  {
  //    cout << J[i][j] << " ";
  //  }
  //  cout << endl;
  //}
  //cout << endl << endl;
  //
  //for (int i = 0; i < Npts; ++i)
  //{
  //  for (int j = 0; j < Npts; ++j)
  //  {
  //    cout << R[i].dx(j) << " ";
  //  }
  //  cout << endl;
  //}
  //cout << endl << endl;
  
}


template<class TensorType, class Double>
void invert_a(TensorType & a, Double & det)
{
  det = a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])+a[0][1]*(a[1][2]*a[2][0]-a[1][0]*a[2][2])+a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]);

  Double const inv00 = ( a[1][1]*a[2][2]-a[1][2]*a[2][1] )/det;
  Double const inv01 = ( a[0][2]*a[2][1]-a[0][1]*a[2][2] )/det;
  Double const inv02 = ( a[0][1]*a[1][2]-a[0][2]*a[1][1] )/det;
  Double const inv10 = ( a[1][2]*a[2][0]-a[1][0]*a[2][2] )/det;
  Double const inv11 = ( a[0][0]*a[2][2]-a[0][2]*a[2][0] )/det;
  Double const inv12 = ( a[0][2]*a[1][0]-a[0][0]*a[1][2] )/det;
  Double const inv20 = ( a[1][0]*a[2][1]-a[1][1]*a[2][0] )/det;
  Double const inv21 = ( a[0][1]*a[2][0]-a[0][0]*a[2][1] )/det;
  Double const inv22 = ( a[0][0]*a[1][1]-a[0][1]*a[1][0] )/det;
  
  a[0][0] = inv00;
  a[0][1] = inv01;
  a[0][2] = inv02;
  a[1][0] = inv10;
  a[1][1] = inv11;
  a[1][2] = inv12;
  a[2][0] = inv20;
  a[2][1] = inv21;
  a[2][2] = inv22;
}

