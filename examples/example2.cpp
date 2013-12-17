#include <iostream>
#define EAD_DEBUG       // to debug
#include "Ead/ead.hpp"
#include <vector>

using namespace std;

// def AD number type with (30 is the maximum num of components)
typedef ead::DFad<double, 30> adouble;

template<class Vec>
void F1 (Vec const& x, Vec &y) {
  y[0] = sin(x[0]);
  y[1] = exp(2*x[0]*x[1]);
  y[2] = sqrt(pow(x[0],2) + 2.*pow(x[1],2));
}

int main()
{
  int n_eqs = 3;
  int n_unk = 2;

  std::cout.setf(std::ios::scientific);
  std::cout.precision(5);

  vector<adouble> x(n_unk, adouble(0,n_unk)),
                  y(n_eqs, adouble(0,n_unk)),
                  dydx(n_unk*n_eqs, adouble(0,n_unk));
  
  for (int i = 0; i < n_unk; ++i)
    x[i].setDiff(i, n_unk);

  // x.val() = ..   set the value
  // x       = ..   set the value set to zero all x components
  x[0].val() = 1./3.;
  x[1].val() = 8./3.;

  // compute function y and your derivatives (automatically)
  F1(x,y);

  // print jacobian
  std::cout << "\nJacobian computed by AD:\n\n";
  for (int i = 0; i < n_eqs; ++i)
  {
    for (int j = 0; j < n_unk; ++j)
      std::cout << y[i].dx(j) << " ";
    std::cout << std::endl;
  }
  std::cout << std::endl;
  
  std::cout << "Exact Jacobian:\n\n";
  std::cout << cos(x[0].val()) << " " << 0. << std::endl
            << 2*x[1].val()*exp(2*x[0].val()*x[1].val()) << " " << 2*x[0].val()*exp(2*x[0].val()*x[1].val())<< std::endl
            << x[0].val()/sqrt(pow(x[0].val(),2) + 2.*pow(x[1].val(),2)) << " " << 2.*x[1].val()/sqrt(pow(x[0].val(),2) + 2.*pow(x[1].val(),2))
            << std::endl << std::endl;
  
  
}
