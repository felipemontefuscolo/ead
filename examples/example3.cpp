#include <iostream>
#define EAD_DEBUG       // to debug
#include "Ead/ead2.hpp"


// deriving a simple polynomial twice


// Definition for syntax sugar.
// 10 is the maximum num of components (independet variables),
// not the actual size.
typedef ead::D2Fad<double, 10> adouble;


int main()
{

  const double pi = 3.14159265;
  
  int const deg = 3; // polynomial degree
  
  double a[deg+1]; // Polynomial coefficients. Since they are constants, they
                   // can be double.
  
  // set coefficients values
  for (int i = 0; i < deg+1; ++i)
    a[i] = 1.0;
  
  
  adouble x,y;



  x.setDiff(0, 1); // first argument set x as an zeroth independent variable (same as x.dx(0) = 1)
                   // second argument says that the problem has one independent variable

  x.val() = pi;    // the polynomail will be evaluated at x = pi
                   // NOTE: do not use x = pi ... this way you make x a constant

  y.setNumVars(1); // the argument says that y has one independent variable


  // ----- compute the polynomial
  y = a[deg];
  for (int i = deg+2; i-- > 2 ;)
    y = y*x + a[i-2];


  // check the result
  std::cout << "function value: \n";
  std::cout << "* AD:    " << y.val() << std::endl;
  std::cout << "* exact: " << a[0] + a[1]*x.val() + a[1]*pow(x.val(),2) +  a[1]*pow(x.val(),3) << std::endl << std::endl;
  std::cout << "derivative value: \n";
  std::cout << "* AD:    " << y.dx() << std::endl;
  std::cout << "* exact: " << a[1] + 2*a[1]*x.val() +  3*a[1]*pow(x.val(),2) << std::endl << std::endl;
  std::cout << "hessian value: \n";
  std::cout << "* AD:    " << y.d2x() << std::endl;
  std::cout << "* exact: " << 2*a[1] +  6*a[1]*x.val() << std::endl << std::endl;

/*
*/  
}



