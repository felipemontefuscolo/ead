// This file is part of Ead, a lightweight C++ template library
// for automatic differentiation.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#include <gtest/gtest.h>
#include <iostream>
#define EAD_DEBUG       // to debug
#include "Ead/ead.hpp"
#include <limits>       // for std::numeric_limits<Real>::epsilon()
#include <cmath>
#include <typeinfo>
#include <vector>
#include <complex>

using namespace ead;
using namespace std;

#define MAX(a,b) ((a) > (b) ? (a) : (b))

double const EAD_EPS = std::numeric_limits<double>::epsilon();
double const EAD_TOL = 500000*EAD_EPS; // ~ 1.1e-10 for double

int const max_comps = 30;

typedef ead::DFad<double,  max_comps> adouble;
typedef ead::DFad<adouble, max_comps> a2double;
typedef ead::DFad<complex<double>,  max_comps> cdouble;



// calculates the derivative a function f(x, i) with F.D.
// f: callable object in form  "void f(X,Y)"
// where X is a input vector and Y and output vector

int main(int argc, char **argv)
{
  cout << "\nTolerance: " << EAD_TOL << endl;
  cout << "Comparison with second order finite differentiation.\n\n";
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

// --------- TESTING FUNDAMENTAL OPERATORS ---------//

TEST(EADTest, FundamentalOpsTest)
{
  double const c = 3.;
  double const a = 4./3.;
  adouble x(c,1),y(0,1);

  x.setDiff(0,1);

  ASSERT_EQ(c, x.val());

#define EAD_SINGLE_EXPR(Expr, ExactVal, ExactDiff) \
  Expr;                            \
  EXPECT_NEAR(ExactVal,  y.val(), EAD_TOL); \
  EXPECT_NEAR(ExactDiff, y.dx(), EAD_TOL);

  //              y(x)     val  dydx
  EAD_SINGLE_EXPR(y = x   , c  , 1.0)
  EAD_SINGLE_EXPR(y =+x   , c  , 1.0)
  EAD_SINGLE_EXPR(y =-x   ,-c  ,-1.0)
  EAD_SINGLE_EXPR(y = a   , a  , 0.0)
  EAD_SINGLE_EXPR(y = x*x , c*c, 2*c)
  EAD_SINGLE_EXPR(y =+x*x , c*c, 2*c)
  EAD_SINGLE_EXPR(y =-x*x ,-c*c,-2*c)
  EAD_SINGLE_EXPR(y = a*x , a*c, a  )
  EAD_SINGLE_EXPR(y = x*a , a*c, a  )
  EAD_SINGLE_EXPR(y = x+x , c+c, 2.0)
  EAD_SINGLE_EXPR(y = x+a , a+c, 1.0)
  EAD_SINGLE_EXPR(y = a+x , a+c, 1.0)
  EAD_SINGLE_EXPR(y = x-x , 0.0, 0.0)
  EAD_SINGLE_EXPR(y = x-a , c-a, 1.0)
  EAD_SINGLE_EXPR(y = a-x , a-c,-1.0)
  EAD_SINGLE_EXPR(y = x/x , 1.0, 0.0)
  EAD_SINGLE_EXPR(y = x/a , c/a, 1./a)
  EAD_SINGLE_EXPR(y = a/x , a/c,-a/c/c)

  // Dont change this order
  y = x;
  EAD_SINGLE_EXPR(y += x   , 2.*c    , 2.0  ) // 2x
  EAD_SINGLE_EXPR(y -= x   , c       , 1.0  ) // x
  EAD_SINGLE_EXPR(y += a*x , c*(a+1.), a+1  ) // x*(a+1)
  EAD_SINGLE_EXPR(y -= a*x , c       , 1.0  ) // x
  EAD_SINGLE_EXPR(y *= x   , c*c     , 2.*c ) // x*x
  EAD_SINGLE_EXPR(y /= x   , c       , 1.0  ) // x
  EAD_SINGLE_EXPR(y *= a*x , c*c*a   , 2*a*c) // ax^2
  EAD_SINGLE_EXPR(y /= a*x , c       , 1.0  ) // x

#undef EAD_SINGLE_EXPR
}

struct F1 {
  template<class Vec>
  void operator() (Vec const& x, Vec &y) const {
    y[0] = sin(x[0]);
    y[1] = exp(2*x[0]*x[1]);
    y[2] = sqrt(pow(x[0],2) + 2.*pow(x[1],2));
  }
};

// Second order finite difference
template<class F, typename Real>
void fd_diff(F const& f, vector<Real> const& x, vector<Real> & dydx)
{
  unsigned x_dim = x.size();
  unsigned n_eqs = dydx.size() / x_dim;

  for (unsigned j = 0; j < x_dim; ++j)
  {
    Real h = MAX(std::fabs(x[j]), 1.)*pow(EAD_EPS, 1./3.);
    volatile Real t = h + x[j];
    h = t - x[j];

    vector<Real> x0(x); // backward x
    vector<Real> x1(x); // forward x
    vector<Real> x00(x); // backward x (High order)
    vector<Real> x11(x); // forward x  (High order)
    vector<Real> y0(n_eqs); // backward y
    vector<Real> y1(n_eqs); // forward y
    vector<Real> y00(n_eqs); // backward y  (High order)
    vector<Real> y11(n_eqs); // forward y   (High order)

    x0[j]  -= h;
    x1[j]  += h;
    x00[j] -= 2.*h; //(High order)
    x11[j] += 2.*h; //(High order)

    f(x0, y0);
    f(x1, y1);
    f(x00, y00);  //(High order)
    f(x11, y11);  //(High order)

    for (unsigned i = 0; i < n_eqs; ++i)
      //dydx[i*x_dim + j] = (y1[i]-y0[i])/(2.*h);
      dydx[i*x_dim + j] = (-y11[i] + 8.*(y1[i]-y0[i]) + y00[i])/(12.*h);

  }


}

TEST(EADTest, FiniteDifferenceF1Test)
{
  int n_eqs = 3;
  int n_unk = 2;

  std::cout.setf(std::ios::scientific);
  std::cout.precision(5);

  vector<double> dydx_exact(n_unk*n_eqs);
  double xvals[] = {1./3., 8./3.};

  // compute exact solution
  {
    vector<double> x(n_unk);
    x[0] = xvals[0];
    x[1] = xvals[1];

    dydx_exact[0*n_unk + 0] = cos(x[0])                                 ;
    dydx_exact[0*n_unk + 1] = 0.0                                       ;
    dydx_exact[1*n_unk + 0] = 2.*x[1]*exp(2*x[0]*x[1])                  ;
    dydx_exact[1*n_unk + 1] = 2.*x[0]*exp(2*x[0]*x[1])                  ;
    dydx_exact[2*n_unk + 0] =    x[0]/sqrt(pow(x[0],2) + 2.*pow(x[1],2));
    dydx_exact[2*n_unk + 1] = 2.*x[1]/sqrt(pow(x[0],2) + 2.*pow(x[1],2));
  }

  // Finite difference test
  {
    vector<double> x(n_unk), y(n_eqs), dydx(n_unk*n_eqs);

    x[0] = xvals[0];
    x[1] = xvals[1];
    F1 f;
    f(x,y);
    fd_diff(f, x, dydx);

    for (int i = 0; i < n_eqs; ++i)
      for (int j = 0; j < n_unk; ++j) {
        EXPECT_NEAR(dydx_exact[i*n_unk + j] , dydx[i*n_unk + j], EAD_TOL);
        cout << "F1(x) FD error in dydx("<<i<<", "<<j<<") :"<< fabs(dydx_exact[i*n_unk + j]-dydx[i*n_unk + j]) << endl;
      }

  }

}


// ~~                                                                 ~~
// ~~~~~~~~                                                     ~~~~~~~~
// ~~~~~~~~~~~~~~~~~~                                 ~~~~~~~~~~~~~~~~~~
// =====================================================================
//                                                                    ::
//                        UNARY OPERATORS TEST                        ::
//                                                                    ::
// =====================================================================
// ~~~~~~~~~~~~~~~~~~                                 ~~~~~~~~~~~~~~~~~~
// ~~~~~~~~                                                     ~~~~~~~~
// ~~                                                                 ~~


#define EAD_DEFINE_UN_FUNC(FunName)                                \
  void unary_##FunName(vector<double> const& x, vector<double> &y) \
  { y[0] = FunName(x[0]); }

EAD_DEFINE_UN_FUNC(cos  )
EAD_DEFINE_UN_FUNC(sin  )
EAD_DEFINE_UN_FUNC(tan  )
EAD_DEFINE_UN_FUNC(acos )
EAD_DEFINE_UN_FUNC(asin )
EAD_DEFINE_UN_FUNC(atan )
EAD_DEFINE_UN_FUNC(cosh )
EAD_DEFINE_UN_FUNC(sinh )
EAD_DEFINE_UN_FUNC(tanh )
EAD_DEFINE_UN_FUNC(exp  )
EAD_DEFINE_UN_FUNC(log  )
EAD_DEFINE_UN_FUNC(log10)
EAD_DEFINE_UN_FUNC(sqrt )
EAD_DEFINE_UN_FUNC(ceil )
EAD_DEFINE_UN_FUNC(floor)

#undef EAD_DEFINE_UN_FUNC


TEST(EADTest, CmathUnaryFuncTest)
{
  //double const d = 8./7.;
  vector<double> X(1), Y(1), DY(1);
  adouble x(0,1,0);
  adouble y(0,1);

#define EAD_UNA_FUN_TEST(FunName)              \
  y = FunName(x);                              \
  fd_diff(unary_##FunName, X, DY);             \
  EXPECT_NEAR(FunName(c), y.val(), EAD_TOL);   \
  EXPECT_NEAR(DY[0]     , y.dx() , EAD_TOL);

  // function tests at various points
  for(double c=0.15; c<0.9; c+= 0.1201057010193)
  {
    x.val() = c;
    X[0]    = c;
    EAD_UNA_FUN_TEST(cos  )
    EAD_UNA_FUN_TEST(sin  )
    EAD_UNA_FUN_TEST(tan  )
    EAD_UNA_FUN_TEST(acos )
    EAD_UNA_FUN_TEST(asin )
    EAD_UNA_FUN_TEST(atan )
    EAD_UNA_FUN_TEST(cosh )
    EAD_UNA_FUN_TEST(sinh )
    EAD_UNA_FUN_TEST(tanh )
    EAD_UNA_FUN_TEST(exp  )
    EAD_UNA_FUN_TEST(log  )
    EAD_UNA_FUN_TEST(log10)
    EAD_UNA_FUN_TEST(sqrt )
    EAD_UNA_FUN_TEST(ceil )
    EAD_UNA_FUN_TEST(floor)
  }

#undef EAD_UNA_FUN_TEST
}



// ~~                                                                 ~~
// ~~~~~~~~                                                     ~~~~~~~~
// ~~~~~~~~~~~~~~~~~~                                 ~~~~~~~~~~~~~~~~~~
// =====================================================================
//                                                                    ::
//                       BINARY OPERATORS TEST                        ::
//                                                                    ::
// =====================================================================
// ~~~~~~~~~~~~~~~~~~                                 ~~~~~~~~~~~~~~~~~~
// ~~~~~~~~                                                     ~~~~~~~~
// ~~                                                                 ~~


#define EAD_DEFINE_BIN_FUNC(FunName)                                \
  void binary_##FunName(vector<double> const& x, vector<double> &y) \
  { y[0] = FunName(x[0], x[1]); }

EAD_DEFINE_BIN_FUNC(max )
EAD_DEFINE_BIN_FUNC(min )
EAD_DEFINE_BIN_FUNC(pow )
EAD_DEFINE_BIN_FUNC(fmod)

#undef EAD_DEFINE_BIN_FUNC



TEST(EADTest, CmathBinaryFuncTest)
{
  vector<double> X(2), DY(2);
  adouble x1(0,2,0), x2(0,2,1);
  x1.setDiff(0,2);
  x2.setDiff(1,2);
  adouble y(0,2);


#define EAD_BIN_FUN_TEST(FunName)                  \
  y = ead::FunName(x1,x2);                         \
  fd_diff(binary_##FunName, X, DY);                \
  EXPECT_NEAR(FunName(c,d), y.val(),  10*EAD_TOL); \
  EXPECT_NEAR(DY[0]       , y.dx(0) , 10*EAD_TOL); \
  EXPECT_NEAR(DY[1]       , y.dx(1) , 10*EAD_TOL) << c << " " << d << endl << endl;

  // function tests at various points
  for(double c=-0.94125124124; c<0.94125124124; c+= 2*0.1201057010193)
  //for (double c=-0.9523452345; c<1.0; c+= 5 )
  {
    for(double d=-0.94125124124; d<0.94125124124; d+= 2*0.1201057010193)
    {
      x1.val() = c;
      x2.val() = d;
      X[0]     = c;
      X[1]     = d;
      EAD_BIN_FUN_TEST(max )
      EAD_BIN_FUN_TEST(min )
      if (c>0) {
        EAD_BIN_FUN_TEST(pow )
      }
      if (c!=d)
      {
        EAD_BIN_FUN_TEST(fmod) // WARNING: THIS FUNCTION DERIV. HAS A LOT OF SINGULARITIES
      }
    }
  }
  
#undef EAD_BIN_FUN_TEST
}


// ~~                                                                 ~~
// ~~~~~~~~                                                     ~~~~~~~~
// ~~~~~~~~~~~~~~~~~~                                 ~~~~~~~~~~~~~~~~~~
// =====================================================================
//                                                                    ::
//                          ALIAS CHECK                     ::
//                                                                    ::
// =====================================================================
// ~~~~~~~~~~~~~~~~~~                                 ~~~~~~~~~~~~~~~~~~
// ~~~~~~~~                                                     ~~~~~~~~
// ~~                                                                 ~~


TEST(EADTest, RectangularJacobianTestF1)
{
  int n_eqs = 3;
  int n_unk = 2;

  std::cout.setf(std::ios::scientific);
  std::cout.precision(5);

  vector<double> dydx_exact(n_unk*n_eqs);
  double xvals[] = {1./3., 8./3.};

  // compute exact solution
  {
    vector<double> x(n_unk);
    x[0] = xvals[0];
    x[1] = xvals[1];

    dydx_exact[0*n_unk + 0] = cos(x[0])                                 ;
    dydx_exact[0*n_unk + 1] = 0.0                                       ;
    dydx_exact[1*n_unk + 0] = 2.*x[1]*exp(2*x[0]*x[1])                  ;
    dydx_exact[1*n_unk + 1] = 2.*x[0]*exp(2*x[0]*x[1])                  ;
    dydx_exact[2*n_unk + 0] =    x[0]/sqrt(pow(x[0],2) + 2.*pow(x[1],2));
    dydx_exact[2*n_unk + 1] = 2.*x[1]/sqrt(pow(x[0],2) + 2.*pow(x[1],2));
  }

  // Automatic test
  {
    vector<adouble> x(n_unk, adouble(0,n_unk)),
                    y(n_eqs, adouble(0,n_unk)),
                    dydx(n_unk*n_eqs, adouble(0,n_unk));

    for (int i = 0; i < n_unk; ++i)
      x[i].setDiff(i, n_unk);

    x[0].val() = xvals[0];
    x[1].val() = xvals[1];
    F1 f;
    f(x,y);


    for (int i = 0; i < n_eqs; ++i)
      for (int j = 0; j < n_unk; ++j) {
        EXPECT_NEAR(dydx_exact[i*n_unk + j] , y[i].dx(j), EAD_TOL);
        cout << "F1(x) AD error in dydx("<<i<<", "<<j<<") :"<< fabs(dydx_exact[i*n_unk + j]-y[i].dx(j)) << endl;
      }

  }
}

struct F2 {
  template<class Vec>
  void operator() (Vec const& x, Vec &y) const {
    y[0] = -x[0] + x[1]*x[2] + x[3]/x[4] - x[5];
    y[0] = sin(tan(cos(y[0])));
    y[0] += x[1];
    y[0] *= x[2];
    y[0] -= x[3];
    y[0] /= x[4];
  }
};

TEST(EADTest, CorrectValuesF2)
{
  double x_[] = {1,2,3,4,5,6};
  vector<double> xx(x_,x_+6);
  vector<double> dydx_fd(6);
  F2 f;
  fd_diff(f, xx, dydx_fd);

  vector<adouble> x(x_,x_+6);
  vector<adouble> y(1);

  for (uint i = 0; i < x.size(); ++i)
    x[i].setDiff(i, x.size());
  y[0].setDiff(-1, x.size());

  f(x,y);

  for (int i = 0; i < 6; ++i) {
    EXPECT_NEAR(dydx_fd[i], y[0].dx(i), EAD_TOL);
  }

}


template<class T>
T F3(T const& x)
{
  T y = x<1 ? x : x*x;
  return y;
}


TEST(EADTest, NonSmoothROP)
{
  adouble x(0,1),y(0,1);
  x.setDiff(0,1);
  y = F3(x);

  EXPECT_NEAR(y.val(), 0.0,EAD_TOL);
  EXPECT_NEAR(y.dx(), 1.0 ,EAD_TOL);

  x.val() = -1;
  y = F3(x);
  EXPECT_NEAR(y.val(),-1.0,EAD_TOL);
  EXPECT_NEAR(y.dx(), 1.0 ,EAD_TOL);

  x.val() = 2;
  y = F3(x);
  EXPECT_NEAR(y.val(), 4.0,EAD_TOL);
  EXPECT_NEAR(y.dx(), 4.0 ,EAD_TOL);

  x.val() = 1;
  y = F3(x);
  EXPECT_NEAR(y.val(), 1.0,EAD_TOL);
  EXPECT_NEAR(y.dx(), 2.0 ,EAD_TOL);
}











