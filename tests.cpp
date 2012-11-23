#include <iostream> 
#define EAD_DEBUG
#include "Ead/ead.hpp"
#include <cmath>

using namespace std;
using namespace ead;

typedef unsigned long uint;

int main(int , char *[])
{
  const uint N = pow(10ul, 7ul)/4;
  const int nc = 30;
  
  DFad<double, nc> a(1.0, nc), b(1.0, nc), c(0.0, nc);
  
  for (int i = 0; i < nc; ++i)
  {
    a.dx(i) = i==0;
    b.dx(i) = i==1;
  }
  
  
  for (uint i=0; i<N; ++i)
    c += a*b*a*b*a*b*a*b*a*b*a*b*a*b*a*b*a*b*a*b*a*b*a*b*a*b*a*b*a*b*a*b*a*b*a*b*a*b*a*b;
    //c += a*b*a*b*a*b*a*b*a*b*a*b*a*b*a*b*a*b*a*b;
    //c += a*b*a*b*a*b*a*b*a*b;
    //c += a*b;
  
  cout << c.val()<<", ";
  for (int i=0; i<nc; ++i)
    cout << c.dx(i) << " ";
  cout << endl;

}

