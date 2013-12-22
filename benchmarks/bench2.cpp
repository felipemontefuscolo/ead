#include <iostream>
//#define EAD_DEBUG       // to debug
#include "Ead/ead2.hpp"
#include <cstdlib>
#include <ctime>
#include <iomanip>      // std::setprecision

using namespace std;
using namespace ead;

typedef ead::D2Fad<double, 30> adouble;

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

void print_jac(double G[], double J[])
{
  for (int i = 0; i < Npts; ++i)
  {
    for (int j = 0; j < Npts; ++j)
      cout << J[i*Npts + j] << " ";
    cout << endl;
  }
  for (int i = 0; i < Npts; ++i)
  {
    cout << G[i] << " ";
  }
  
  cout << "\n\n";
}

double element_residue_ad(double x_[], double G_[], double H_[])
{
  adouble x[Npts],  y; // ead
  
  y.setNumVars(Npts);
  for (int i = 0; i < Npts; ++i)
  {
    x[i].val() = x_[i]; // random values
    x[i].setDiff(i, Npts);
  }
  
  y = 0;
  
  for (int i = 0; i < Npts; i+=3)
  {
    y += pow(x[i],2) + pow(x[i+1],2) + pow(x[i+2],2);
  }

  y = 1./sqrt(y);

  
  for (int i = 0; i < Npts; ++i)
  {
    for (int j = 0; j < Npts; ++j)
      H_[i*Npts + j] = y.d2x(i,j);
    G_[i] = y.dx(i);
  }
  
  
  return y.val();
}


double element_residue_exact(double x_[], double G_[], double H_[])
{
  double y;
  double *x = x_;
  double *H = H_;

  y = 0;
  for (int i = 0; i < Npts; i+=3)
  {
    y += pow(x[i],2) + pow(x[i+1],2) + pow(x[i+2],2);
  }
  y = 1./sqrt(y);

  double y2 (pow(y,2));
  double y3 (y2*y);
  double y5 (y2*y3);

  for (int i = 0; i < Npts; ++i)
  {
    for (int j = 0; j < Npts; ++j)
      if (i==j)
        H[i*Npts + j] = y3*(3*x[i]*x[j]*y2 - 1);
      else
        H[i*Npts + j] = y5*3*x[i]*x[j];
    G_[i] = -x[i]*y3;
  }
  
  
  return y;
}




int main(int argc, char *argv[])
{
  int N_iterations = int(1e+5);
  
  double u[Npts], J[Npts*Npts], G[Npts];
  
  if (argc > 1)
  {
    N_iterations = atoi(argv[1]);
  }
  
  srand(1234);
  

  for (int i = 0; i < Npts; ++i)
  {
    u[i] = double(rand())/RAND_MAX; // random values
    X[i] = 10*double(rand())/RAND_MAX; // random values
  }



  // ============================================================
  
  clock_t begin = clock();
  
  for (int i = 0; i < N_iterations; ++i)
    element_residue_ad(u, G, J);

  clock_t end = clock();
  double elapsed_ad = double(end - begin) / CLOCKS_PER_SEC;
  
  cout << "EAD:\n";
  print_jac(G,J);
  
  // ============================================================

  begin = clock();
  
  for (int i = 0; i < N_iterations; ++i)
    element_residue_exact(u, G, J);

  end = clock();
  double elapsed_ex = double(end - begin) / CLOCKS_PER_SEC;
  
  cout << "EXACT:\n";
  print_jac(G,J);
  
  
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

