// This file is part of Ead, a lightweight C++ template library
// for automatic differentiation.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#include <gtest/gtest.h>
#include <iostream>
#define EAD_DEBUG       // to debug
#include "Ead/ead2.hpp"
#include <limits>       // for std::numeric_limits<Real>::epsilon()
#include <cmath>
#include <typeinfo>
#include <vector>
#include <complex>

using namespace ead;
using namespace std;

#define MAX(a,b) ((a) > (b) ? (a) : (b))

double const EAD_EPS = std::numeric_limits<double>::epsilon();
//double const EAD_TOL = 50000000*EAD_EPS; // ~ 1.1e-10 for double
double const EAD_TOL  = 500000*EAD_EPS; // ~ 1.1e-10 for double
double const EAD_TOL2 = 5.e-7;

int const max_comps = 30;

typedef ead::D2Fad<double,  max_comps> adouble;
typedef ead::D2Fad<adouble, max_comps> a2double;
typedef ead::D2Fad<complex<double>,  max_comps> cdouble;



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

#define EAD_SINGLE_EXPR(Expr, ExactVal, ExactDiff, ExactDiff2)  \
  Expr;                                                         \
  EXPECT_NEAR(ExactVal,  y.val(), EAD_TOL) << #Expr << endl;    \
  EXPECT_NEAR(ExactDiff, y.dx(), EAD_TOL) << #Expr << endl;     \
  EXPECT_NEAR(ExactDiff2, y.d2x(), EAD_TOL) << #Expr << endl;

  //              y(x)     val  dydx   d2ydx2
  EAD_SINGLE_EXPR(y = x   , c  , 1.0  , 0.0         )
  EAD_SINGLE_EXPR(y =+x   , c  , 1.0  , 0.0         )
  EAD_SINGLE_EXPR(y =-x   ,-c  ,-1.0  , 0.0         )
  EAD_SINGLE_EXPR(y = a   , a  , 0.0  , 0.0         )
  EAD_SINGLE_EXPR(y = x*x , c*c, 2*c  , 2.0         )
  EAD_SINGLE_EXPR(y =+x*x , c*c, 2*c  , 2.0         )
  EAD_SINGLE_EXPR(y =-x*x ,-c*c,-2*c  ,-2.0         )
  EAD_SINGLE_EXPR(y = a*x , a*c, a    , 0.0         )
  EAD_SINGLE_EXPR(y = x*a , a*c, a    , 0.0         )
  EAD_SINGLE_EXPR(y = x+x , c+c, 2.0  , 0.0         )
  EAD_SINGLE_EXPR(y = x+a , a+c, 1.0  , 0.0         )
  EAD_SINGLE_EXPR(y = a+x , a+c, 1.0  , 0.0         )
  EAD_SINGLE_EXPR(y = x-x , 0.0, 0.0  , 0.0         )
  EAD_SINGLE_EXPR(y = x-a , c-a, 1.0  , 0.0         )
  EAD_SINGLE_EXPR(y = a-x , a-c,-1.0  , 0.0         )
  EAD_SINGLE_EXPR(y = x/x , 1.0, 0.0  , 0.0         )
  EAD_SINGLE_EXPR(y = x/a , c/a, 1./a , 0.0         )
  EAD_SINGLE_EXPR(y = a/x , a/c,-a/c/c, 2.*a/(c*c*c))

  // Dont change this order
  y = x;
  EAD_SINGLE_EXPR(y += x   , 2.*c    , 2.0  , 0.0 ) // 2x
  EAD_SINGLE_EXPR(y -= x   , c       , 1.0  , 0.0 ) // x
  EAD_SINGLE_EXPR(y += a*x , c*(a+1.), a+1  , 0.0 ) // x*(a+1)
  EAD_SINGLE_EXPR(y -= a*x , c       , 1.0  , 0.0 ) // x
  EAD_SINGLE_EXPR(y *= x   , c*c     , 2.*c , 2.0 ) // x*x
  EAD_SINGLE_EXPR(y /= x   , c       , 1.0  , 0.0 ) // x
  EAD_SINGLE_EXPR(y *= a*x , c*c*a   , 2*a*c, 2.*a) // ax^2
  EAD_SINGLE_EXPR(y /= a*x , c       , 1.0  , 0.0 ) // x

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
    vector<Real> x000(x); // backward x (High order)
    vector<Real> x111(x); // forward x  (High order)
    vector<Real> y0(n_eqs); // backward y
    vector<Real> y1(n_eqs); // forward y
    vector<Real> y00(n_eqs); // backward y  (High order)
    vector<Real> y11(n_eqs); // forward y   (High order)
    vector<Real> y000(n_eqs); // backward y  (High order)
    vector<Real> y111(n_eqs); // forward y   (High order)

    x0[j]  -= h;
    x1[j]  += h;
    x00[j] -= 2.*h; //(High order)
    x11[j] += 2.*h; //(High order)
    x000[j] -= 3.*h; //(High order)
    x111[j] += 3.*h; //(High order)

    f(x0, y0);
    f(x1, y1);
    f(x00, y00);  //(High order)
    f(x11, y11);  //(High order)
    f(x000, y000);  //(High order)
    f(x111, y111);  //(High order)

    for (unsigned i = 0; i < n_eqs; ++i)
      //dydx[i*x_dim + j] = (y1[i]-y0[i])/(2.*h);
      //dydx[i*x_dim + j] = (y111[i]-y000[i] -9.*(y11[i]-y00[i]) + 45.*(y1[i]-y0[i]) )/(60.*h);
      dydx[i*x_dim + j] = (-y11[i] + 8.*(y1[i]-y0[i]) + y00[i])/(12.*h);

  }


}

template<class F, typename Real>
void fd2_diff(F const& f, vector<Real> const& x, vector<Real> & dydx,  vector<Real> & d2ydx2)
{
  unsigned x_dim = x.size();
  unsigned n_eqs = dydx.size() / x_dim;

  for (unsigned i = 0; i < n_eqs; ++i)
  {
    for (unsigned k = 0; k < x_dim; ++k)
    {
      Real h = 500.*MAX(std::fabs(x[k]), 1.)*pow(EAD_EPS, 1./3.);
      volatile Real t = h + x[k];
      h = t - x[k];

      vector<Real> x0(x); // backward x
      vector<Real> x1(x); // forward x
      vector<Real> x00(x); // backward x (High order)
      vector<Real> x11(x); // forward x  (High order)
      vector<Real> x000(x); // backward x (High order)
      vector<Real> x111(x); // forward x  (High order)
      vector<Real> dydx0(dydx); // backward y
      vector<Real> dydx1(dydx); // forward y
      vector<Real> dydx00(dydx); // backward y  (High order)
      vector<Real> dydx11(dydx); // forward y   (High order)
      vector<Real> dydx000(dydx); // backward y  (High order)
      vector<Real> dydx111(dydx); // forward y   (High order)

      x0[k]  -= h;
      x1[k]  += h;
      x00[k] -= 2.*h; //(High order)
      x11[k] += 2.*h; //(High order)
      x000[k] -= 3.*h; //(High order)
      x111[k] += 3.*h; //(High order)

      fd_diff(f, x0, dydx0);
      fd_diff(f, x1, dydx1);
      fd_diff(f, x00, dydx00);  //(High order)
      fd_diff(f, x11, dydx11);  //(High order)
      fd_diff(f, x000, dydx000);  //(High order)
      fd_diff(f, x111, dydx111);  //(High order)

      for (unsigned j = 0; j < x_dim; ++j)
      {
        unsigned S = i*x_dim + j;
        //d2ydx2[i*x_dim*x_dim + j*x_dim + k] = (-dydx11[S] + 8.*(dydx1[S]-dydx0[S]) + dydx00[S])/(12.*h);
        d2ydx2[i*x_dim*x_dim + j*x_dim + k] = (dydx111[S]-dydx000[S] - 9.*(dydx11[S]-dydx00[S]) + 45.*(dydx1[S]-dydx0[S])   )/(60.*h);
      }
    }
  }

  fd_diff(f, x, dydx);

}

TEST(EADTest, FiniteDifferenceF1Test)
{
  int n_eqs = 3;
  int n_unk = 2;

  std::cout.setf(std::ios::scientific);
  std::cout.precision(5);

  vector<double> dydx_exact(n_unk*n_eqs), d2ydx2_exact(n_unk*n_unk*n_eqs);
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

    d2ydx2_exact.at(0*n_unk*n_unk + 0*n_unk + 0) = -sin(x[0]);
    d2ydx2_exact.at(0*n_unk*n_unk + 0*n_unk + 1) = 0.0       ;
    d2ydx2_exact.at(0*n_unk*n_unk + 1*n_unk + 0) = 0.0       ;
    d2ydx2_exact.at(0*n_unk*n_unk + 1*n_unk + 1) = 0.0       ;

    d2ydx2_exact.at(1*n_unk*n_unk + 0*n_unk + 0) = 4*exp(2*x[0]*x[1])*x[1]*x[1]                                                                  ;
    d2ydx2_exact.at(1*n_unk*n_unk + 0*n_unk + 1) = 4*exp(2*x[0]*x[1])*x[0]*x[1]+2*exp(2*x[0]*x[1])                                               ;
    d2ydx2_exact.at(1*n_unk*n_unk + 1*n_unk + 0) = 4*exp(2*x[0]*x[1])*x[0]*x[1]+2*exp(2*x[0]*x[1])                                               ;
    d2ydx2_exact.at(1*n_unk*n_unk + 1*n_unk + 1) = 4*exp(2*x[0]*x[1])*x[0]*x[0]                                                                  ;

    d2ydx2_exact.at(2*n_unk*n_unk + 0*n_unk + 0) = (2*pow(x[1],2)*sqrt(2*pow(x[1],2)+pow(x[0],2)))/(4*pow(x[1],4)+4*pow(x[0],2)*pow(x[1],2)+pow(x[0],4));
    d2ydx2_exact.at(2*n_unk*n_unk + 0*n_unk + 1) = -(2.*x[0]*x[1])/pow(2.*pow(x[1],2)+pow(x[0],2),1.5)                                                 ;
    d2ydx2_exact.at(2*n_unk*n_unk + 1*n_unk + 0) = -(2.*x[0]*x[1])/pow(2.*pow(x[1],2)+pow(x[0],2),1.5)                                                 ;
    d2ydx2_exact.at(2*n_unk*n_unk + 1*n_unk + 1) = (2*pow(x[0],2)*sqrt(2*pow(x[1],2)+pow(x[0],2)))/(4*pow(x[1],4)+4*pow(x[0],2)*pow(x[1],2)+pow(x[0],4));


  }

  // Finite difference test
  {
    vector<double> x(n_unk), y(n_eqs), dydx(n_unk*n_eqs), d2ydx2(n_unk*n_unk*n_eqs);

    x[0] = xvals[0];
    x[1] = xvals[1];
    F1 f;
    f(x,y);
//    fd_diff(f, x, dydx);
    fd2_diff(f, x, dydx, d2ydx2);

    for (int i = 0; i < n_eqs; ++i)
      for (int j = 0; j < n_unk; ++j) {
        EXPECT_NEAR(dydx_exact[i*n_unk + j] , dydx[i*n_unk + j], EAD_TOL);
        cout << "F1(x) FD error in dydx("<<i<<", "<<j<<") :"<< fabs(dydx_exact[i*n_unk + j]-dydx[i*n_unk + j]) << endl;
      }

    for (int i = 0; i < n_eqs; ++i)
      for (int j = 0; j < n_unk; ++j) {
        for (int k = 0; k < n_unk; ++k) {
          double d2ydx2_exact_ = d2ydx2_exact[i*n_unk*n_unk + j*n_unk + k];
          double d2ydx2_ = d2ydx2[i*n_unk*n_unk + j*n_unk + k];
          EXPECT_NEAR( d2ydx2_exact_, d2ydx2_, EAD_TOL2* MAX(1.,fabs(d2ydx2_exact_)));
          cout << "F1(x) FD error in d2yd2x("<<i<<", "<<j<<", "<<k<<") :"<< fabs(d2ydx2_exact_- d2ydx2_)/MAX(1.,fabs(d2ydx2_exact_)) << endl;
        }
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
EAD_DEFINE_UN_FUNC(fabs )

#undef EAD_DEFINE_UN_FUNC


TEST(EADTest, CmathUnaryFuncTest)
{
  //double const d = 8./7.;
  vector<double> X(1), Y(1), DY(1), D2Y(1);
  adouble x(0,1,0);
  adouble y(0,1);

#define EAD_UNA_FUN_TEST(FunName)              \
  y = FunName(x);                              \
  fd2_diff(unary_##FunName, X, DY, D2Y);       \
  EXPECT_NEAR(FunName(c), y.val(), EAD_TOL *MAX(1., fabs(y.val())));   \
  EXPECT_NEAR(DY[0]     , y.dx() , EAD_TOL *MAX(1., fabs(y.dx())));    \
  EXPECT_NEAR(D2Y[0]    , y.d2x(), EAD_TOL2*MAX(1., fabs(y.d2x())));

  // function tests at various points
  for(double c=-0.6347834; c<0.9; c+= 0.1201057010193)
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
    if ( c > 0) {
      EAD_UNA_FUN_TEST(log  )
      EAD_UNA_FUN_TEST(log10)
      EAD_UNA_FUN_TEST(sqrt )
    }
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
//                    PSEUDO-UNARY OPERATORS TEST                     ::
//                                                                    ::
// =====================================================================
// ~~~~~~~~~~~~~~~~~~                                 ~~~~~~~~~~~~~~~~~~
// ~~~~~~~~                                                     ~~~~~~~~
// ~~                                                                 ~~


#define EAD_DEFINE_PUN_FUNC(FunName)                           \
  struct unaryL_##FunName {                                     \
    double a;                                                  \
    unaryL_##FunName(double a_) : a(a_) {}                       \
    void operator() (vector<double> const& x, vector<double> &y) const \
    { y[0] = FunName(x[0],a); }                                   \
  };                                                             \
                                                                \
  struct unaryR_##FunName {                                     \
    double a;                                                    \
    unaryR_##FunName(double a_) : a(a_) {}                       \
    void operator() (vector<double> const& x, vector<double> &y) const \
    { y[0] = FunName(a,x[0]); }                                   \
  };

EAD_DEFINE_PUN_FUNC(max )
EAD_DEFINE_PUN_FUNC(min )
EAD_DEFINE_PUN_FUNC(pow )
EAD_DEFINE_PUN_FUNC(fmod)

#undef EAD_DEFINE_PUN_FUNC


TEST(EADTest, CmathPunaryFuncTest)
{
  vector<double> X(1), DY(1), D2Y(1);
  adouble x(0,1,0);
  adouble y(0,1);

#define EAD_PUNA_FUN_TEST(FunName)              \
  y = ead::FunName(x,a);                        \
  fd2_diff(unaryL_##FunName(a), X, DY, D2Y);          \
  EXPECT_NEAR(FunName(c,a), y.val(), EAD_TOL);   \
  EXPECT_NEAR(DY[0]       , y.dx() , EAD_TOL*MAX(1.,fabs(y.dx()))) << #FunName << ", 1st, c=" <<c << ", a= "<< a<< endl;    \
  EXPECT_NEAR(D2Y[0]      , y.d2x() , EAD_TOL2) << #FunName << ", 1st, c=" <<c << ", a= "<< a<< endl;    \
                                                \
  y = ead::FunName(a,x);                        \
  fd2_diff(unaryR_##FunName(a), X, DY, D2Y);          \
  EXPECT_NEAR(FunName(a,c), y.val(), EAD_TOL);   \
  EXPECT_NEAR(DY[0]       , y.dx() , EAD_TOL*MAX(1.,fabs(y.dx()))) << "2nd" << endl;    \
  EXPECT_NEAR(D2Y[0]      , y.d2x() , EAD_TOL2*MAX(1.,fabs(y.d2x()))) << "2nd" << endl;    \


  // function tests at various points
  for(double c=-0.94125124124; c<0.94125124124; c+= 2*0.1201057010193)
  {
    for(double a=-0.94125124124; a<0.94125124124; a+= 2*0.1201057010193)
    {
      x.val() = c;
      X[0]    = c;
      if (c != a)
      {
        EAD_PUNA_FUN_TEST(max )
        EAD_PUNA_FUN_TEST(min )
      }
      if (c!=a) {
//        EAD_PUNA_FUN_TEST(fmod)
      }
    }

  }

  // function tests at various points
  for(double c=.5; c<5; c+= .5)
  {
    for(double a=.5; a<5; a+= .5)
    {
      x.val() = c;
      X[0]    = c;
      EAD_PUNA_FUN_TEST(pow )
    }
  }


#undef EAD_PUNA_FUN_TEST
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
  vector<double> X(2), DY(2), D2Y(4);
  adouble x1(0,2,0), x2(0,2,1);
  x1.setDiff(0,2);
  x2.setDiff(1,2);
  adouble y(0,2);


#define EAD_BIN_FUN_TEST(FunName)                  \
  y = ead::FunName(x1,x2);                         \
  fd2_diff(binary_##FunName, X, DY, D2Y);          \
  EXPECT_NEAR(FunName(c,d), y.val(),  10*EAD_TOL); \
  EXPECT_NEAR(DY[0]       , y.dx(0) , 10*EAD_TOL); \
  EXPECT_NEAR(DY[1]       , y.dx(1) , 10*EAD_TOL) << c << " " << d << endl << endl; \
  EXPECT_NEAR(D2Y[0]      , y.d2x(0,0) , MAX(1.,fabs(y.d2x(0,0)))*EAD_TOL2) << c << " " << d << endl << endl; \
  EXPECT_NEAR(D2Y[1]      , y.d2x(0,1) , MAX(1.,fabs(y.d2x(0,1)))*EAD_TOL2) << c << " " << d << endl << endl; \
  EXPECT_NEAR(D2Y[3]      , y.d2x(1,1) , MAX(1.,fabs(y.d2x(1,1)))*EAD_TOL2) << c << " " << d << endl << endl;

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
      if (c != d)
      {
        EAD_BIN_FUN_TEST(max )
        EAD_BIN_FUN_TEST(min )
      }
      //if (c>0) {
      //  EAD_BIN_FUN_TEST(pow )
      //}
      if (c!=d)
      {
      //  EAD_BIN_FUN_TEST(fmod) // WARNING: THIS FUNCTION DERIV. HAS A LOT OF SINGULARITIES
      }
    }
  }

  for(double c=.25; c<5; c+= .25)
  {
    for(double d=-2.5; d<=2.5; d+= .5)
    {
      x1.val() = c;
      x2.val() = d;
      X[0]     = c;
      X[1]     = d;
      EAD_BIN_FUN_TEST(pow )
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


TEST(EADTest, AliasTest1)
{
  double const c = 8./7.;
  adouble x(c,1,0);
  adouble y(0,1);

  y = x;
  y = x - sin(x+x);
  y = fabs( cos(2*y + x) );
  y = exp(sin(cos(sqrt(y))));
  y = -y + 1.;
  
  EXPECT_NEAR(-1.097967159421815, y.val(),  1e-4*EAD_TOL);
  EXPECT_NEAR(3.516004803580704,  y.dx(),   1e-4*EAD_TOL);
  EXPECT_NEAR(-2.762497313468641, y.d2x(),  1e-4*EAD_TOL2);
  
  y = y;
  
  EXPECT_NEAR(-1.097967159421815, y.val(),  1e-4*EAD_TOL);
  EXPECT_NEAR(3.516004803580704,  y.dx(),   1e-4*EAD_TOL);
  EXPECT_NEAR(-2.762497313468641, y.d2x(),  1e-4*EAD_TOL2);
}

TEST(EADTest, LongTreeTest)
{
  int N = 10;
  double const c = 8./7.;
  vector<adouble> x(N);
  adouble y(0,N);

  for (int i = 0; i < N; ++i)
  {
    x[i].val() = 1. + (double(i)-4.5)/10.;
    x[i].setDiff(i, N);
  }

  y = x[0]*x[1]*x[2]*x[3]*x[4]*x[5]*x[6]*x[7]*x[8]*x[9];

  EXPECT_NEAR(0.63970355423584, y.val(),  1e-4*EAD_TOL);
  
  {
    double dx_exact[]  = {1.163097371337891,0.98415931420898,0.85293807231445,0.75259241674805,0.67337216235352,0.60924148022461,0.55626396020508,0.51176284338867,0.47385448461914,0.44117486499023};
    double d2x_exact[] = {0,1.789380571289062,1.550796495117188,1.368349848632813,1.224313022460938,1.107711782226563,1.011389018554688,0.93047789707031,0.86155360839844,0.80213611816406,0,1.312212418945312,1.157834487304688,1.035957172851562,0.93729458496094,0.85579070800781,0.78732745136719,0.72900689941406,0.67873056152344,0,1.003456555664063,0.89782954980469,0.81232197363281,0.74168528027344,0.68235045785156,0.63180597949219,0.58823315332031,0,0.79220254394531,0.71675468261719,0.65442818847656,0.60207393339844,0.55747586425781,0.51902925292969,0,0.64130682128906,0.58554101074219,0.53869772988281,0.49879419433594,0.46439459472656,0,0.52977520019531,0.48739318417969,0.45128998535156,0.42016653808594,0,0.44501116816406,0.41204737792969,0.38363031738281,0,0.37908358769531,0.35293989199219,0,0.32679619628906,0};
  
    for (int i = 0; i < N; ++i)
    {
      EXPECT_NEAR(dx_exact[i],  y.dx(i),   1e-14);
      
      for (int j = i; j < N; ++j)
        EXPECT_NEAR(d2x_exact[i*N - i*(i+1)/2 + j], y.d2x(i,j),  1e-14) << "\n" << "i=" << i << ", j=" << j << "\n";
    }  
  
  }


  y = x[0]*x[1]*x[2]*x[3]*x[4]*x[5]*x[6]*x[7] + x[8]*x[9];
  
  EXPECT_NEAR(2.284296196289063, y.val(),  1e-4*EAD_TOL);

  {
    double dx_exact[]  = {0.59417490234375,0.50276337890625,0.43572826171875,0.38446611328125,0.34399599609375,0.31123447265625,0.28417060546875,0.26143695703125,1.45,1.35};
    double d2x_exact[] = {0,0.914115234375,0.792233203125,0.699029296875,0.625447265625,0.565880859375,0.516673828125,0.475339921875,0,0,0,0.670351171875,0.591486328125,0.529224609375,0.478822265625,0.437185546875,0.402210703125,0,0,0,0.512621484375,0.458661328125,0.414979296875,0.378894140625,0.348582609375,0,0,0,0.404701171875,0.366158203125,0.334318359375,0.307572890625,0,0,0,0.327615234375,0.299126953125,0.275196796875,0,0,0,0.270638671875,0.248987578125,0,0,0,0.227336484375,0,0,0,0,0,0,1,0};
  
    for (int i = 0; i < N; ++i)
    {
      EXPECT_NEAR(dx_exact[i],  y.dx(i),   1e-14);
      
      for (int j = i; j < N; ++j)
        EXPECT_NEAR(d2x_exact[i*N - i*(i+1)/2 + j], y.d2x(i,j),  1e-14) << "\n" << "i=" << i << ", j=" << j << "\n";
    }  
  
  }

  y = x[0]*x[1]*x[2] + x[3]*x[4]*x[5]*x[6]*x[7] + x[8]*x[9];
  
  EXPECT_NEAR(3.4444453125, y.val(),  1e-4*EAD_TOL);

  {
    double dx_exact[]  = {0.4875,0.4125,0.3575,1.43390625,1.28296875,1.16078125,1.05984375,0.97505625,1.45,1.35};
    double d2x_exact[] = {0,0.75,0.65,0,0,0,0,0,0,0,0,0.55,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.509375,1.365625,1.246875,1.147125,0,0,0,1.221875,1.115625,1.026375,0,0,0,1.009375,0.928625,0,0,0,0.847875,0,0,0,0,0,0,1,0};
  
    for (int i = 0; i < N; ++i)
    {
      EXPECT_NEAR(dx_exact[i],  y.dx(i),   1e-14);
      
      for (int j = i; j < N; ++j)
        EXPECT_NEAR(d2x_exact[i*N - i*(i+1)/2 + j], y.d2x(i,j),  1e-14) << "\n" << "i=" << i << ", j=" << j << "\n";
    }  
  
  }

}

TEST(EADTest, LongTreeTest2)
{
  int N = 25;
  double const c = 8./7.;
  vector<adouble> x(N);
  adouble y(0,N);
  
  for (int i = 0; i < N; ++i)
  {
    x[i].val() = c;
    x[i].setDiff(i, N);
  }
  
  y = x[0]*x[0]+x[1]*x[1]+x[2]*x[2]+x[3]*x[3]+x[4]*x[4]+
      x[5]*x[5]+x[6]*x[6]+x[7]*x[7]+x[8]*x[8]+x[9]*x[9]+
      x[10]*x[10]+x[11]*x[11]+x[12]*x[12]+x[13]*x[13]+x[14]*x[14]+
      x[15]*x[15]+x[16]*x[16]+x[17]*x[17]+x[18]*x[18]+x[19]*x[19]+
      x[20]*x[20]+x[21]*x[21]+x[22]*x[22]+x[23]*x[23]+x[24]*x[24];

  for (int i = 0; i < N; ++i)
  {
    EXPECT_NEAR(2.*c,  y.dx(i),   1e-14);
    
    EXPECT_NEAR(2.,  y.d2x(i,i),   1e-14);
    
    for (int j = i+1; j < N; ++j)
      EXPECT_NEAR(0., y.d2x(i,j),  1e-14) << "\n" << "i=" << i << ", j=" << j << "\n";
  } 

   
  
}

// ~~                                                                 ~~
// ~~~~~~~~                                                     ~~~~~~~~
// ~~~~~~~~~~~~~~~~~~                                 ~~~~~~~~~~~~~~~~~~
// =====================================================================
//                                                                    ::
//                          RELATIONAL OPS                            ::
//                                                                    ::
// =====================================================================
// ~~~~~~~~~~~~~~~~~~                                 ~~~~~~~~~~~~~~~~~~
// ~~~~~~~~                                                     ~~~~~~~~
// ~~                                                                 ~~


TEST(EADTest,RelationalOpsTest1)
{
  double const c = 8./7.;
  adouble x(c,3,0);
  adouble y(c,3,1);
  adouble z(2*c,3,2);
  adouble w(-c,3,0);

  EXPECT_TRUE (x==y);
  EXPECT_FALSE(x==w);
  EXPECT_TRUE (x!=z);
  EXPECT_FALSE(x!=y);
  EXPECT_TRUE (x <z);
  EXPECT_FALSE(x <w);
  EXPECT_TRUE (z >x);
  EXPECT_FALSE(w >x);
  EXPECT_TRUE (x<=y);
  EXPECT_FALSE(x<=w);
  EXPECT_TRUE (y>=x);
  EXPECT_FALSE(w>=x);

  EXPECT_TRUE (x==c  );  EXPECT_TRUE (c==x  );
  EXPECT_FALSE(x==-c );  EXPECT_FALSE(-c==x );
  EXPECT_TRUE (x!=2*c);  EXPECT_TRUE (2*c!=x);
  EXPECT_FALSE(x!=c  );  EXPECT_FALSE(c!=x  );
  EXPECT_TRUE (x <2*c);  EXPECT_TRUE (2*c>x);
  EXPECT_FALSE(x <-c );  EXPECT_FALSE(-c>x );
  EXPECT_TRUE (z >c  );  EXPECT_TRUE (c< z  );
  EXPECT_FALSE(w >c  );  EXPECT_FALSE(c< w  );
  EXPECT_TRUE (x<=c  );  EXPECT_TRUE (c>=x );
  EXPECT_FALSE(x<=-c );  EXPECT_FALSE(-c>=x );
  EXPECT_TRUE (y>=c  );  EXPECT_TRUE (c<=y  );
  EXPECT_FALSE(w>=c  );  EXPECT_FALSE(c<=w  );


}



TEST(EADTest, RectangularJacobianTestF1)
{
  int n_eqs = 3;
  int n_unk = 2;

  std::cout.setf(std::ios::scientific);
  std::cout.precision(5);

  vector<double> dydx_exact(n_unk*n_eqs), d2ydx2_exact(n_unk*n_unk*n_eqs);
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
    
    d2ydx2_exact.at(0*n_unk*n_unk + 0*n_unk + 0) = -sin(x[0]);
    d2ydx2_exact.at(0*n_unk*n_unk + 0*n_unk + 1) = 0.0       ;
    d2ydx2_exact.at(0*n_unk*n_unk + 1*n_unk + 0) = 0.0       ;
    d2ydx2_exact.at(0*n_unk*n_unk + 1*n_unk + 1) = 0.0       ;

    d2ydx2_exact.at(1*n_unk*n_unk + 0*n_unk + 0) = 4*exp(2*x[0]*x[1])*x[1]*x[1]                                                                  ;
    d2ydx2_exact.at(1*n_unk*n_unk + 0*n_unk + 1) = 4*exp(2*x[0]*x[1])*x[0]*x[1]+2*exp(2*x[0]*x[1])                                               ;
    d2ydx2_exact.at(1*n_unk*n_unk + 1*n_unk + 0) = 4*exp(2*x[0]*x[1])*x[0]*x[1]+2*exp(2*x[0]*x[1])                                               ;
    d2ydx2_exact.at(1*n_unk*n_unk + 1*n_unk + 1) = 4*exp(2*x[0]*x[1])*x[0]*x[0]                                                                  ;

    d2ydx2_exact.at(2*n_unk*n_unk + 0*n_unk + 0) = (2*pow(x[1],2)*sqrt(2*pow(x[1],2)+pow(x[0],2)))/(4*pow(x[1],4)+4*pow(x[0],2)*pow(x[1],2)+pow(x[0],4));
    d2ydx2_exact.at(2*n_unk*n_unk + 0*n_unk + 1) = -(2.*x[0]*x[1])/pow(2.*pow(x[1],2)+pow(x[0],2),1.5)                                                 ;
    d2ydx2_exact.at(2*n_unk*n_unk + 1*n_unk + 0) = -(2.*x[0]*x[1])/pow(2.*pow(x[1],2)+pow(x[0],2),1.5)                                                 ;
    d2ydx2_exact.at(2*n_unk*n_unk + 1*n_unk + 1) = (2*pow(x[0],2)*sqrt(2*pow(x[1],2)+pow(x[0],2)))/(4*pow(x[1],4)+4*pow(x[0],2)*pow(x[1],2)+pow(x[0],4));

    
  }

  // AD test
  {
    vector<adouble> x(n_unk, adouble(0,n_unk)),
                    y(n_eqs, adouble(0,n_unk));

    for (int i = 0; i < n_unk; ++i)
      x[i].setDiff(i, n_unk);

    x[0].val() = xvals[0];
    x[1].val() = xvals[1];
    F1 f;
    f(x,y);


    for (int i = 0; i < n_eqs; ++i)
      for (int j = 0; j < n_unk; ++j) {
        EXPECT_NEAR(dydx_exact[i*n_unk + j] , y[i].dx(j), 1e-4*EAD_TOL);
        cout << "F1(x) AD error in dydx("<<i<<", "<<j<<") :"<< fabs(dydx_exact[i*n_unk + j]-y[i].dx(j)) << endl;
      }

    for (int i = 0; i < n_eqs; ++i)
      for (int j = 0; j < n_unk; ++j) {
        for (int k = 0; k < n_unk; ++k) {
          EXPECT_NEAR(d2ydx2_exact[i*n_unk*n_unk + j*n_unk + k] , y[i].d2x(j,k), 1e-3*EAD_TOL);
          cout << "F1(x) AD error in dydx("<<i<<", "<<j<<") :"<< fabs(d2ydx2_exact[i*n_unk*n_unk + j*n_unk + k] - y[i].d2x(j,k)) << endl;
        }
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
  vector<double> d2ydx2_fd(36);
  F2 f;
  fd2_diff(f, xx, dydx_fd, d2ydx2_fd);

  vector<adouble> x(x_,x_+6);
  vector<adouble> y(1);

  for (uint i = 0; i < x.size(); ++i)
    x[i].setDiff(i, x.size());
  y[0].setNumVars(x.size());

  f(x,y);

  for (int i = 0; i < 6; ++i) {
    EXPECT_NEAR(dydx_fd[i], y[0].dx(i), EAD_TOL);
    for (int j = 0; j < 6; ++j)
      EXPECT_NEAR(d2ydx2_fd[i*6 + j], y[0].d2x(i,j), EAD_TOL2);
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
  EXPECT_NEAR(y.d2x(), 0.0 ,EAD_TOL);

  x.val() = -1;
  y = F3(x);
  EXPECT_NEAR(y.val(),-1.0,EAD_TOL);
  EXPECT_NEAR(y.dx(), 1.0 ,EAD_TOL);
  EXPECT_NEAR(y.d2x(), 0.0 ,EAD_TOL);

  x.val() = 2;
  y = F3(x);
  EXPECT_NEAR(y.val(), 4.0,EAD_TOL);
  EXPECT_NEAR(y.dx(), 4.0 ,EAD_TOL);
  EXPECT_NEAR(y.d2x(), 2.0 ,EAD_TOL);

  x.val() = 1;
  y = F3(x);
  EXPECT_NEAR(y.val(), 1.0,EAD_TOL);
  EXPECT_NEAR(y.dx(), 2.0 ,EAD_TOL);
  EXPECT_NEAR(y.d2x(), 2.0 ,EAD_TOL);
}


TEST(EADTest, DeepTree)
{
  adouble x(0,1,0),y(0,1);

  y = 1;
  x.val() = 1;

  y += cosh(sinh(sqrt(exp(sin(cos(tan(x)))))));


  EXPECT_NEAR(y.val(), 2.789093925730607, 1e-3*EAD_TOL);
  EXPECT_NEAR(y.dx(), -3.966758435906108 ,1e-3*EAD_TOL);
  EXPECT_NEAR(y.d2x(), 12.09191423016362 ,1e-3*EAD_TOL);
}





