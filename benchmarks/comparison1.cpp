#include <iostream>
//#define EAD_DEBUG       // to debug
#include "Ead/ead.hpp"
#include <cstdlib>
#include <ctime>
#include <iomanip>      // std::setprecision

#ifdef HAS_ADEPT
#include "include/adept.h"
#endif

#ifdef HAS_FADBAD
#include "fadiff.h"
#endif

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
    u[i].val() = u_[i]; // random values
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


void element_residue_exact(double u_[], double R_[], double J_[])
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
  
}

#ifdef HAS_ADEPT
void element_residue_adept(double u_[], double R_[], double J_[])
{
  using adept::adouble;
  adept::Stack stack;

  adept::adouble u[Npts]; // adept
  
  adept::set_values(&u[0], Npts, u_);
  stack.new_recording();

  adouble R[Npts];

  // residue
  for (int q = 0; q < Nqp; ++q)
  {
    adouble uqp(u[0]*Phi[0][q]);
    
    for (int j = 1; j < Npts; ++j)
      uqp += u[j]*Phi[j][q];

    adouble sqrt_u = sqrt(uqp);

    for (int i = 0; i < Npts; ++i)
      R[i] += (uqp + pow(sqrt_u,3) + sqrt_u) * Phi[i][q] * weight[q];
  }
  
  stack.independent(&u[0], Npts);
  stack.dependent(&R[0], Npts);
  stack.jacobian(J_);
  
  for (int i = 0; i < Npts; ++i)
    R_[i] = R[i].value();
}
#endif


#ifdef HAS_FADBAD
void element_residue_fadbad(double u_[], double R_[], double J_[])
{
  typedef fadbad::F<double,Npts> adouble;

  adouble u[Npts]; // adept
  adouble R[Npts];

  for (int i = 0; i < Npts; ++i)
  {
    u[i] = u_[i]; // random values
    u[i].diff(i);
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
    R_[i] = R[i].x();
    for (int j = 0; j < Npts; ++j)
      J_[i*Npts + j] = R[i].d(j);
  }
}
#endif




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
  
  // ============================================================
#ifdef HAS_ADEPT

  begin = clock();
  
  for (int i = 0; i < N_iterations; ++i)
    element_residue_adept(u, R, J);

  end = clock();
  double elapsed_adept = double(end - begin) / CLOCKS_PER_SEC;
  
  cout << "ADEPT:\n";
  print_jac(J);

#endif

  // ============================================================
#ifdef HAS_FADBAD

  begin = clock();
  
  for (int i = 0; i < N_iterations; ++i)
    element_residue_fadbad(u, R, J);

  end = clock();
  double elapsed_fadbad = double(end - begin) / CLOCKS_PER_SEC;
  
  cout << "FADBAD:\n";
  print_jac(J);

#endif


  // ====================== TIMES ============================

  cout << "Time elapsed:\n";
  cout << "EAD: "; cout << std::setprecision(9) << elapsed_ad;
  cout << endl << endl;

#ifdef HAS_ADEPT
  cout << "Time elapsed:\n";
  cout << "ADEPT: "; cout << std::setprecision(9) << elapsed_adept;
  cout << endl << endl;  
#endif

#ifdef HAS_FADBAD
  cout << "Time elapsed:\n";
  cout << "FADBAD: "; cout << std::setprecision(9) << elapsed_fadbad;
  cout << endl << endl;  
#endif

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






