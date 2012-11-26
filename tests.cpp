// This file is part of Ead, a lightweight C++ template library
// for automatic differentiation.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.



#include <iostream> 
#define EAD_DEBUG
#include "Ead/ead.hpp"
#include <cmath>
#include <typeinfo>
#include <complex>

using namespace std;
using namespace tr1;
using namespace ead;

typedef unsigned long uint;

const int nc = 30;

typedef DFad<double, nc>  ddouble;
typedef DFad<ddouble, nc> d2double;
typedef DFad<complex<double>, nc>  dcomplex;

int main(int , char *[])
{
  const uint N = 1;//pow(10ul, 7ul)/4;

  
  ddouble a(1.0, nc), b(1.0, nc), c(0.0, nc);
  d2double w;
  dcomplex z(1.0,nc);

  for (int i = 0; i < nc; ++i)
  {
    a.dx(i) = i==0;
    b.dx(i) = i==1;
  }
  
  //c = a*b;
  for (uint i=0; i<N; ++i)
    c += 1*a*2*b*2*a*b*2*a*2*b*2*a*2*b*a*b*a*b*a*b*a*b*a*b*a*b*a*b*a*b*a*b*a*b*a*b*a*b*a*b*a*b*a*b*a*b;
    //c += a*b*a*b*a*b*a*b*a*b*a*b*a*b*a*b*a*b*a*b;
    //c += a*b*a*b*a*b*a*b*a*b;
    //c += a*b;
  
  cout << c.val()<<", ";
  for (int i=0; i<nc; ++i)
    cout << c.dx(i) << " ";
  cout << endl;

}

