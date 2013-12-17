#include <iostream>
#define EAD_DEBUG       // to debug
#include "Ead/ead2.hpp"
#include <vector>

using namespace std;

double const pi = 3.14159265359;

// computing the hessian of a function
template<class T>
void F(T const* x, T & y) {

  y = ead::pow(x[0],pi) + sin(x[1]*x[0]);

}


// Definition for syntax sugar.
// 10 is the maximum num of components (independet variables),
// not the actual size.
typedef ead::D2Fad<double, 10> adouble;


int main()
{
  
  int n_unk = 2;
  
  std::vector<adouble> x(n_unk, adouble(0,n_unk));
  adouble y(n_unk);
  
  double X[2] = {1./3., 8./3.}; // some random values
  
  for (int i = 0; i < n_unk; ++i)
  {
    x[i].setDiff(i);
    x[i].val() = X[i];
  }

  y.setNumVars(2);

  F(x.data(), y);

  std::cout << "\nfunction value:\n"
            << "AD   : " << y.val() << std::endl
            << "exact: " << pow(X[0],pi) + sin(X[1]*X[0]) << std::endl << std::endl;
  
  std::cout << "function derivatives:\n"
            << "AD   : " << y.dx(0) << " " << y.dx(1) << std::endl
            << "exact: " << pow(X[0],(pi-1))*pi+X[1]*cos(X[0]*X[1]) << " " << X[0]*cos(X[0]*X[1]) << std::endl << std::endl;
 
  std::cout << "hessian:\n"
            << "AD   : " << y.d2x(0,0) << " " << y.d2x(0,1) << " " << y.d2x(1,1) << std::endl
            << "exact: " << pow(X[0],(pi-2))*(pi-1)*pi-X[1]*X[1]*sin(X[0]*X[1])
            << " "       << cos(X[0]*X[1])-X[0]*X[1]*sin(X[0]*X[1])
            << " "       << -X[0]*X[0]*sin(X[0]*X[1]) << std::endl << std::endl;


  


}

