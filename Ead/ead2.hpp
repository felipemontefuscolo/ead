// This file is part of Ead, a lightweight C++ template library
// for automatic differentiation.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef EAD2_HPP
#define EAD2_HPP


#include <functional>
#include <ostream>
#include "ead_mpl.hpp"
#include "ead_check.hpp"
#include <cmath>

//#include <iostream>

namespace ead
{




// all fad numbers and expressions inherits from this class.
template<class A>
struct ExprWrapper2
{
  inline
  operator A const& () const
  { return *static_cast<A const*>(this);}
};

// AD type
template<typename T_, unsigned Mnc_>
class D2Fad : public ExprWrapper2<D2Fad<T_, Mnc_> >
{
  typedef D2Fad Self;
public:
  typedef T_   ValueT; // can be a base type or Dfad itself.
  typedef typename GetField<ValueT>::type FieldT;// FieldT is the base
                                                 // type, e.g., double,
                                                 // float, complex, etc.
  // Returns ValueT if it has trivial constructor,
  // and returns ValueT const& otherwise.
  typedef typename RetTrivial<ValueT>::const_type ValueT_CR;

  typedef D2Fad LeafType;

  typedef LeafData2_<Self, ValueT> LeafData;

  static const unsigned max_n_comps = Mnc_;
private:

  // ------------ ATTRIBUTES --------------------------
  ValueT   m_val;             // value
  unsigned m_n_vars;
  ValueT   m_dx[max_n_comps]; // df/dui
  ValueT   m_d2x[max_n_comps*(max_n_comps+1)/2]; // hessian
  // ---------------------------------------------------


  void resize(unsigned s)
  { m_n_vars = s;}

public:

  static int const n_leaves = 1;
  static int const dtmp_size = 1;

  inline explicit
  D2Fad(ValueT_CR val=0, unsigned n_comps=0) : m_val(val), m_n_vars(n_comps), m_dx(), m_d2x()
  {
    EAD_CHECK(n_comps <= max_n_comps, "num comps > max num comps");
  }

  inline
  D2Fad(ValueT_CR val, unsigned n_comps, unsigned ith) : m_val(val), m_n_vars(n_comps), m_dx(), m_d2x()
  {
    EAD_CHECK(n_comps <= max_n_comps && ith < n_comps, "num comps > max num comps or ith >= n_comps");
    m_dx[ith] = 1.0;
  }

  // When ExprWrapper2 is D2Fad it uses the default constructor, as expected
  template<typename ExprT>
  D2Fad(ExprWrapper2<ExprT> const& e_)
  {
    ExprT const& e (e_);
    m_n_vars = e.numVars();
    *this = e;
  }

  ValueT&    val()            { return m_val; }
  ValueT&    dx(unsigned i=0) { return m_dx[i]; }
  ValueT&    d2x(unsigned i=0,unsigned j=0)
  {
    if (j < i)
      std::swap(i,j);
    return m_d2x[i*numVars()-i*(i+1)/2+j];
  }
  ValueT&    d2xFast(unsigned i) { return m_d2x[i];}

  ValueT_CR  val()            const { return m_val; }
  ValueT_CR  dx(unsigned i=0) const { return m_dx[i]; }
  ValueT_CR  d2x(unsigned i=0,unsigned j=0) const
  {
    if (j < i)
      std::swap(i,j);
    return m_d2x[i*numVars()-i*(i+1)/2+j];
  }
  ValueT_CR  d2xFast(unsigned i) const { return m_d2x[i];}

  unsigned  numVars() const {return m_n_vars;}
  unsigned  hessianSize() const {return m_n_vars*(1+m_n_vars)/2;}

  void setDiff(int ith, unsigned n_comps)
  {
    m_n_vars = n_comps;
    resetDerivatives();
    dx(ith) = ValueT(1.0);
  }

  void setDiff(int ith)
  {
    resetDerivatives();
    dx(ith) = ValueT(1.0);
  }

  void setNumVars(unsigned n_comps)
  {
    m_n_vars = n_comps;
  }

  void resetDerivatives()
  {
    for (unsigned i = 0; i < m_n_vars; ++i)
      m_dx[i] = ValueT(0.0);
    for (unsigned i = 0, N = hessianSize(); i < N; ++i)
      m_d2x[i] = ValueT(0.0);
  }

  // TODO: not for users, put it in a private area
  // bar = df/dterminal
  // csize = hessian's number of columns
  void computeHessianPartials(ValueT_CR bar, ValueT_CR bar2, LeafData leaves[], ValueT */*dtmp[]*/, ValueT /*hessian_off_diag*/[], int /*csize*/) const
  {
    leaves[0].partial  = bar;
    leaves[0].hes_dig  = bar2;
    //std::cout << " hessian_off_diag[0] leaf = " << hessian_off_diag << std::endl;
  }

  void getLeafsAndTempPartials(ValueT dtmp[], LeafData leaves[]) const
  {
    dtmp[0] = 1.0;
    leaves[0].ptr = this;
    //std::cout << "dtmp = " << dtmp << std::endl;
  }

  void computeTempPartials(ValueT_CR bar, ValueT dtmp[]) const
  {
    dtmp[0] = bar;
  }

//         ----------------------------------------------
//     -------------------------------------------------------
//------------------ ASSIGNS OPERATORS | EXPR VERSION ---------------
//     -------------------------------------------------------
//         ----------------------------------------------

  template<class ExprT>
  inline
  Self& operator= (ExprWrapper2<ExprT> const& e_)
  {
    const int n_leaves = ExprT::n_leaves;
    ExprT const& e (e_);
    EAD_CHECK(numVars()==e.numVars(), "incompatible dimension");
    LeafData leaves[n_leaves];
    ValueT hessian_off_diag[n_leaves*(n_leaves-1)/2 + (n_leaves==1?1:0)]; /* hessian off-diagonal */
    /* partial of the temporaries; it doesn't store for the last temporary */
    ValueT dtmp[ExprT::dtmp_size - n_leaves];
    e.getLeafsAndTempPartials(dtmp - n_leaves, leaves);
    e.computeHessianPartials(1.0, 0.0, leaves, dtmp-n_leaves, hessian_off_diag, n_leaves);
    //for (int i = 0; i < ExprT::n_leaves*(ExprT::n_leaves+1)/2; ++i)
    //  std::cout << " hessian_off_diag operator= [" <<i<<"] " << &hessian_off_diag[i] << std::endl;
    //for (int i = 0; i < ExprT::dtmp_size - ExprT::n_leaves; ++i)
    //  std::cout << " dtmp operator= [" <<i<<"] " << &dtmp[i] << std::endl;
    //for (int i = 0; i < ExprT::n_leaves*(ExprT::n_leaves+1)/2; ++i)
    //  std::cout << " val hessian_off_diag operator= [" <<i<<"] " << hessian_off_diag[i] << std::endl;
    //for (int k = 0; k < ExprT::n_leaves; ++k)
    //  std::cout << "leaves["<<k<<"]->d2x(0) = " << leaves[k]->d2x() << std::endl;
    ValueT e_val = e.val();
    ValueT e_dxi;
    ValueT e_dxij;
    for (int i = 0; i<(int)m_n_vars; ++i)
    {
      for (int j = i; j<(int)m_n_vars; ++j)
      {
        e_dxij = 0.;
        for (int k = 0; k< n_leaves ; ++k)
        {
          ValueT const leaf_k_dxi = leaves[k].ptr->dx(i);
          ValueT const leaf_k_dxj = leaves[k].ptr->dx(j);

          e_dxij += leaf_k_dxi*leaf_k_dxj*leaves[k].hes_dig + leaves[k].partial*leaves[k].ptr->d2x(i,j);
          for (int l = k+1; l< n_leaves; ++l)
          {
            int kl = k*(n_leaves-1)-k*(k+1)/2+l-1;

            e_dxij += hessian_off_diag[kl]*(leaves[k].ptr->dx(i)*leaves[l].ptr->dx(j) + leaves[k].ptr->dx(j)*leaves[l].ptr->dx(i));
          }
        }
        d2x(i,j)  = e_dxij;
      }
    }
    for (unsigned i = 0; i<m_n_vars; ++i)
    {
      e_dxi = ExprDxi<Self, ExprT::n_leaves>(leaves, i).result;;
      dx(i) =  e_dxi;
    }
    this->val() = e_val;
    return *this;
  }

#define EAD_ASSIGN_OPS(OP_NAME, OP)                                 \
  template<class ExprT>                                             \
  Self& operator OP_NAME (ExprWrapper2<ExprT> const& e_)            \
  {                                                                 \
    ExprT const& e (e_);                                            \
    EAD_CHECK(numVars()==e.numVars(), "incompatible dimension");  \
    *this = *this OP e;                                             \
    return *this;                                                   \
  }

  EAD_ASSIGN_OPS(+= , +) /* += */
  EAD_ASSIGN_OPS(-= , -) /* -= */
  EAD_ASSIGN_OPS(*= , *) /* *= */
  EAD_ASSIGN_OPS(/= , /) /* /= */

#undef EAD_ASSIGN_OPS



//         ----------------------------------------------
//     -------------------------------------------------------
//------------------ ASSIGNS OPERATORS | FIELD VERSION ---------------
//     -------------------------------------------------------
//         ----------------------------------------------

  template<typename T>
  inline
  typename EnableIf<IsField<T,Self>::value, Self&>::type
  operator= (T const& z)
  {
    resetDerivatives();
    this->val() = z;
    return *this;
  }

  template<typename T>
  inline
  typename EnableIf<IsField<T,Self>::value, Self&>::type
  operator+= (T const& z)
  {
    this->val() += z;
    return *this;
  }

  template<typename T>
  inline
  typename EnableIf<IsField<T,Self>::value, Self&>::type
  operator-= (T const& z)
  {
    this->val() -= z;
    return *this;
  }

  template<typename T>
  inline
  typename EnableIf<IsField<T,Self>::value, Self&>::type
  operator*= (T const& z)
  {
    this->val() *= z;
    for (unsigned i = 0; i < m_n_vars; ++i)
      m_dx[i] *= z;
    for (unsigned i = 0, N = hessianSize(); i < N; ++i)
      m_d2x[i] *= z;
    return *this;
  }

  template<typename T>
  inline
  typename EnableIf<IsField<T,Self>::value, Self&>::type
  operator/= (T const& z)
  {
    this->val() /= z;
    for (int i = 0; i < m_n_vars; ++i)
      m_dx[i] /= z;
    for (unsigned i = 0, N = hessianSize(); i < N; ++i)
      m_d2x[i] /= z;
    return *this;
  }


}; // ------------ end class D2Fad ------------------




// ~~                                                                 ~~
// ~~~~~~~~                                                     ~~~~~~~~
// ~~~~~~~~~~~~~~~~~~                                 ~~~~~~~~~~~~~~~~~~
// =====================================================================
//                                                                    ::
//                        RELATIONAL OPERATORS                        ::
//                                                                    ::
// =====================================================================
// ~~~~~~~~~~~~~~~~~~                                 ~~~~~~~~~~~~~~~~~~
// ~~~~~~~~                                                     ~~~~~~~~
// ~~                                                                 ~~

#define EAD_RELAT_OP(OP)                                            \
template<typename L, typename R>                                    \
inline                                                              \
bool operator OP (ExprWrapper2<L> const& l_, ExprWrapper2<R> const& r_) \
{                                                                       \
  L const& l (l_);                                                      \
  R const& r (r_);                                                      \
  return l.val() OP r.val();                                            \
}                                                                       \
                                                                        \
template<typename L, typename T>                                        \
inline                                                                  \
typename EnableIf<IsField<T,L>::value, bool >::type                     \
operator OP (ExprWrapper2<L> const& l_, T const& r)                     \
{                                                                       \
  L const& l (l_);                                                      \
  return l.val() OP r;                                                  \
}                                                                       \
                                                                        \
template<typename T, typename R>                                        \
inline                                                                  \
typename EnableIf<IsField<T,R>::value, bool >::type                     \
operator OP (T const& l, ExprWrapper2<R> const& r_)                     \
{                                                                       \
  R const& r (r_);                                                      \
  return l OP r.val();                                                  \
}

EAD_RELAT_OP(==)
EAD_RELAT_OP(!=)
EAD_RELAT_OP(<)
EAD_RELAT_OP(>)
EAD_RELAT_OP(<=)
EAD_RELAT_OP(>=)
EAD_RELAT_OP(<<=)
EAD_RELAT_OP(>>=)
EAD_RELAT_OP(&)
EAD_RELAT_OP(|)

#undef EAD_RELAT_OP

// ~~                                                                 ~~
// ~~~~~~~~                                                     ~~~~~~~~
// ~~~~~~~~~~~~~~~~~~                                 ~~~~~~~~~~~~~~~~~~
// =====================================================================
//                                                                    ::
//                         BINARY OPERATORS                           ::
//                                                                    ::
// =====================================================================
// ~~~~~~~~~~~~~~~~~~                                 ~~~~~~~~~~~~~~~~~~
// ~~~~~~~~                                                     ~~~~~~~~
// ~~                                                                 ~~

template <typename T>
std::ostream& operator << (std::ostream& os, ExprWrapper2<T> const& x_) {

  T const& x = T(x_);

  os << "(" << x.val() << " | " << x.dx(0);

  for (unsigned i=1; i< x.numVars(); i++) {
    os  << ", " << x.dx(i);
  }

  os << " | " << x.d2xFast(0);

  for (unsigned i=1; i< x.hessianSize(); i++) {
    os  << ", " << x.d2xFast(i);
  }

  os << " )";
  return os;
}


// DUMMY_S is not used for anything
#define EAD_BINARY_OP(OP_FUN_NAME, OP_CLASS_NAME, VAL_RET, DEDL, DEDR, D2EDL, D2EDR, D2EDLR)           \
template<typename ExprL, typename ExprR>                                                               \
class OP_CLASS_NAME : public ExprWrapper2<OP_CLASS_NAME<ExprL,ExprR> >                                  \
{                                                                                                      \
public:                                                                                                \
  typedef typename ExprL::ValueT ValueT;                                                               \
  typedef typename ExprL::FieldT FieldT;                                                               \
  typedef typename ExprL::ValueT_CR ValueT_CR;                                                         \
  typedef typename ExprL::LeafType LeafType;                                                           \
  typedef typename ExprL::LeafData LeafData;                                                               \
                                                                                                       \
private:                                                                                               \
                                                                                                       \
  ExprL const& m_expL;                                                                                 \
  ExprR const& m_expR;                                                                                 \
                                                                                                       \
  ValueT const x;    /* left value  */                                                                 \
  ValueT const y;    /* right value */                                                                 \
                                                                                                       \
public:                                                                                                \
                                                                                                       \
  static int const n_leaves1  = ExprL::n_leaves;                                                         \
  static int const n_leaves2  = ExprR::n_leaves;                                                         \
  static int const n_leaves   = n_leaves1 + n_leaves2;                                                    \
  static int const dtmp_size = n_leaves + ExprL::dtmp_size + ExprR::dtmp_size;                          \
                                                                                                       \
  OP_CLASS_NAME(ExprL const& lhs, ExprR const& rhs) : m_expL(lhs),                                     \
                                                      m_expR(rhs),                                     \
                                                      x(lhs.val()),                                    \
                                                      y(rhs.val())                                     \
  { }                                                                                                  \
                                                                                                       \
  ValueT val() const                                                                                   \
  { return VAL_RET;}                                                                                   \
                                                                                                       \
  unsigned numVars() const                                                                            \
  { return m_expL.numVars(); }                                                                        \
                                                                                                       \
                                                                                                       \
  void getLeafsAndTempPartials(ValueT dtmp[], LeafData leaves[]) const                           \
  {                                                                                                    \
    m_expL.computeTempPartials(1.0, dtmp + n_leaves                    );                                \
    m_expR.computeTempPartials(1.0, dtmp + n_leaves + ExprL::dtmp_size );                                \
                                                                                                       \
    m_expL.getLeafsAndTempPartials(dtmp + n_leaves                    , leaves);                         \
    m_expR.getLeafsAndTempPartials(dtmp + n_leaves + ExprL::dtmp_size , leaves + n_leaves1);              \
  }                                                                                                    \
                                                                                                       \
  void computeTempPartials(ValueT_CR bar, ValueT dtmp[]) const                                         \
  {                                                                                                    \
    m_expL.computeTempPartials((DEDL)*bar, dtmp);                                                      \
    m_expR.computeTempPartials((DEDR)*bar, dtmp + n_leaves1);                                           \
  }                                                                                                    \
                                                                                                       \
  void computeHessianPartials(ValueT_CR bar, ValueT_CR bar2, LeafData leaves[], ValueT dtmp[], ValueT hessian_off_diag[], int csize) const  \
  {                                                                                                                                         \
    ValueT const dedl   = DEDL;                                                                                                             \
    ValueT const dedr   = DEDR;                                                                                                             \
    ValueT const d2edlr = D2EDLR;                                                                                                           \
                                                                                                                                            \
    ValueT const diag = bar2*dedl*dedr + bar*d2edlr;                                                                                        \
                                                                                                                                            \
    for (int i = 0; i < n_leaves1; ++i)                                                                                                     \
    {                                                                                                                                       \
      ValueT const dldai = dtmp[n_leaves + i];                                                                                              \
                                                                                                                                            \
      ValueT const aux = dldai*diag;                                                                                                        \
                                                                                                                                            \
      for (int j = n_leaves1; j < n_leaves; ++j)                                                                                            \
      {                                                                                                                                     \
        ValueT const drdaj = dtmp[n_leaves + ExprL::dtmp_size + j-n_leaves1];                                                               \
                                                                                                                                            \
        hessian_off_diag[i*(csize-1)-i*(i+1)/2+j-1] = drdaj*aux;                                                                            \
      }                                                                                                                                     \
    }                                                                                                                                       \
                                                                                                                                            \
    ValueT const d2edl = D2EDL;                                                                                                             \
    ValueT const d2edr = D2EDR;                                                                                                             \
                                                                                                                                            \
    m_expL.computeHessianPartials(dedl*bar, d2edl*bar + dedl*dedl*bar2, leaves, dtmp+n_leaves, hessian_off_diag, csize);             \
    m_expR.computeHessianPartials(dedr*bar, d2edr*bar + dedr*dedr*bar2, leaves + n_leaves1,                                          \
                                                                       dtmp+n_leaves+ExprL::dtmp_size,                                          \
                                                                       hessian_off_diag + n_leaves1*(csize-1) - n_leaves1*(1+n_leaves1)/2 + n_leaves1, \
                                                                       csize - n_leaves1 );                                                     \
                                                                                                                                               \
                                                                                                       \
  }                                                                                                    \
                                                                                                       \
};                                                                                                     \
                                                                                                       \
template<typename L, typename R>                                                                       \
inline                                                                                                 \
OP_CLASS_NAME<L, R>                                                                                    \
OP_FUN_NAME (ExprWrapper2<L> const& l, ExprWrapper2<R> const& r)                                       \
{                                                                                                      \
  return OP_CLASS_NAME<L, R>(l,r);                                                                     \
}
// END EAD_BINARY_OP MACRO ----------------------------------------------------------------------------

/*                                    value  dfdx        dfdy       d2fdx2   d2fdy2         d2fdxdy*/

EAD_BINARY_OP(operator+, BinAddiExpr_2, x + y, 1.0       , 1.0       , 0.0   , 0.0          , 0.0      )
EAD_BINARY_OP(operator-, BinSubtExpr_2, x - y, 1.0       ,-1.0       , 0.0   , 0.0          , 0.0      )
EAD_BINARY_OP(operator*, BinMultExpr_2, x*y  , y         , x         , 0.0   , 0.0          , 1.0      )
EAD_BINARY_OP(operator/, BinDiviExpr_2, x/y  , 1.0/y     ,(-x/(y*y)) , 0.0   , 2.0*x/(y*y*y), -1./(y*y))

EAD_BINARY_OP(max,  BinMaxExpr_2 , (x<y)?y:x     , (x==y? .5 :  (x<y)?0:1),  (x==y? .5 :  (x<y)?1:0) , 0.0 , 0.0 , 0.0 )
EAD_BINARY_OP(min,  BinMinExpr_2 ,!(x<y)?y:x     , (x==y? .5 : !(x<y)?0:1),  (x==y? .5 : !(x<y)?1:0) , 0.0 , 0.0 , 0.0 )
EAD_BINARY_OP(pow,  BinPowExpr_2 , std::pow(x,y) , std::pow(x,y-1)*y    , std::pow(x,y)*std::log(x) , std::pow(x,y-2)*y*(y-1) , std::pow(x,y)*std::pow(std::log(x),2) , std::pow(x,(y-1))*std::log(x)*y+std::pow(x,(y-1)) )
EAD_BINARY_OP(fmod, BinFmodExpr_2, std::fmod(x,y), 1.0  ,-(y==0.?0.: ((y<0)^(x<0)? std::ceil(x/y) : std::floor(x/y)) )  , 0.0,  0.0 , 0.0 )

#undef EAD_BINARY_OP








// ~~                                                                 ~~
// ~~~~~~~~                                                     ~~~~~~~~
// ~~~~~~~~~~~~~~~~~~                                 ~~~~~~~~~~~~~~~~~~
// =====================================================================
//                                                                    ::
//                        PSEUDO-UNARY OPERATORS                      ::
//                                                                    ::
// =====================================================================
// ~~~~~~~~~~~~~~~~~~                                 ~~~~~~~~~~~~~~~~~~
// ~~~~~~~~                                                     ~~~~~~~~
// ~~                                                                 ~~


// OBS: Binary operators involving scalars is considered unary operators
// in the sense that only one Expr class is involved, e.g., X+scalar,
// scalar+X, scalar*X, X*scalar, pow(X, scalar), pow(scalar, X), etc.

#define X m_exp.val()
#define EAD_PSEUDO_UNARY_OP_CLASS_TYPE(OP_FUN_NAME, OP_CLASS_NAME, VAL_RET, DEDX, D2EDX)                     \
  template<typename T, typename ExprT>                                                                       \
  class OP_CLASS_NAME : public ExprWrapper2<OP_CLASS_NAME<T,ExprT> >                                          \
  {                                                                                                          \
  public:                                                                                                    \
    typedef typename ExprT::ValueT ValueT;                                                                   \
    typedef typename ExprT::FieldT FieldT;                                                                   \
    typedef typename ExprT::ValueT_CR ValueT_CR;                                                             \
    typedef typename ExprT::LeafType LeafType;                                                               \
    typedef typename ExprT::LeafData LeafData;                                                               \
                                                                                                             \
  private:                                                                                                   \
    ExprT const& m_exp;                                                                                      \
                                                                                                             \
    T const& a; /* scalar value */                                                                           \
                                                                                                             \
  public:                                                                                                    \
                                                                                                             \
    static int const n_leaves  = ExprT::n_leaves;                                                              \
    static int const dtmp_size = n_leaves + ExprT::dtmp_size;                                                 \
                                                                                                             \
    OP_CLASS_NAME(T const& s_, ExprT const& e_) : m_exp(e_),                                                 \
                                                  a(s_)                                                      \
    { }                                                                                                      \
                                                                                                             \
    ValueT val() const                                                                                       \
    {return VAL_RET;}                                                                                        \
                                                                                                             \
    unsigned numVars() const                                                                                \
    { return m_exp.numVars(); }                                                                             \
                                                                                                             \
                                                                                                             \
    void getLeafsAndTempPartials(ValueT dtmp[], LeafData leaves[]) const                               \
    {                                                                                                        \
      m_exp.computeTempPartials(1.0, dtmp + n_leaves );                                                       \
      m_exp.getLeafsAndTempPartials(dtmp + n_leaves, leaves);                                                  \
    }                                                                                                        \
                                                                                                             \
    void computeTempPartials(ValueT_CR bar, ValueT dtmp[]) const                                             \
    {                                                                                                        \
      m_exp.computeTempPartials((DEDX)*bar, dtmp);                                                           \
    }                                                                                                        \
                                                                                                             \
    void computeHessianPartials(ValueT_CR bar, ValueT_CR bar2, LeafData leaves[], ValueT dtmp[], ValueT hessian_off_diag[], int csize) const \
    {                                                                                                                                        \
      ValueT const dedx  = DEDX;                                                                                                             \
      ValueT const d2edx = D2EDX;                                                                                                            \
      m_exp.computeHessianPartials(dedx*bar, d2edx*bar + dedx*dedx*bar2, leaves, dtmp+n_leaves, hessian_off_diag, csize);       \
    }                                                                                                                                        \
                                                                                                                                             \
  };

#define EAD_PSEUDO_UNARY_OP_FUNCTION_L(OP_FUN_NAME, OP_CLASS_NAME)                                                \
  template<typename Expr, typename T>                                                                             \
  inline                                                                                                          \
  typename EnableIf<IsField<T,Expr>::value,OP_CLASS_NAME<T,Expr> >::type                                          \
  OP_FUN_NAME (ExprWrapper2<Expr> const& l, T const& r)                                                           \
  {                                                                                                               \
    return OP_CLASS_NAME<T,Expr>(r,l);                                                                            \
  }

#define EAD_PSEUDO_UNARY_OP_FUNCTION_R(OP_FUN_NAME, OP_CLASS_NAME)                                                \
  template<typename Expr, typename T>                                                                             \
  inline                                                                                                          \
  typename EnableIf<IsField<T,Expr>::value,OP_CLASS_NAME<T,Expr> >::type                                          \
  OP_FUN_NAME (T const& l, ExprWrapper2<Expr> const& r)                                                           \
  {                                                                                                               \
    return OP_CLASS_NAME<T,Expr>(l,r);                                                                            \
  }

EAD_PSEUDO_UNARY_OP_CLASS_TYPE(operator+, UnaAddiExpr_2, X+a, 1.0 , 0.0) // case:
EAD_PSEUDO_UNARY_OP_FUNCTION_L(operator+, UnaAddiExpr_2)                 // X + scalar
EAD_PSEUDO_UNARY_OP_FUNCTION_R(operator+, UnaAddiExpr_2)                 // scalar + X

EAD_PSEUDO_UNARY_OP_CLASS_TYPE(operator*, UnaMultExpr_2, X*a, a, 0.0) // case:
EAD_PSEUDO_UNARY_OP_FUNCTION_L(operator*, UnaMultExpr_2)              // X*scalar
EAD_PSEUDO_UNARY_OP_FUNCTION_R(operator*, UnaMultExpr_2)              // scalar*X

EAD_PSEUDO_UNARY_OP_CLASS_TYPE(max, UnaMaxExpr_2, (X<a)?a:X,   X==a ? 0.5 : ((X<a)?0.:1.) , 0.0) // case:
EAD_PSEUDO_UNARY_OP_FUNCTION_L(max, UnaMaxExpr_2)                                                // max(X,scalar)
EAD_PSEUDO_UNARY_OP_FUNCTION_R(max, UnaMaxExpr_2)                                                // max(scalar,X)

EAD_PSEUDO_UNARY_OP_CLASS_TYPE(min, UnaMinExpr_2, !(X<a)?a:X,  X==a ? 0.5 : (!(X<a)?0.:1.), 0.0) // case:
EAD_PSEUDO_UNARY_OP_FUNCTION_L(min, UnaMinExpr_2)                                                // min(X,scalar)
EAD_PSEUDO_UNARY_OP_FUNCTION_R(min, UnaMinExpr_2)                                                // min(scalar,X)


// UnaDiviExpr and UnaSubtExpr are not symmetric,
// therefore must be implemented separately.

EAD_PSEUDO_UNARY_OP_CLASS_TYPE(operator-, UnaSubtExprL_2, X-a,  1.0, 0.0) // expr at left
EAD_PSEUDO_UNARY_OP_CLASS_TYPE(operator-, UnaSubtExprR_2, a-X, -1.0, 0.0) // expr at right
EAD_PSEUDO_UNARY_OP_FUNCTION_L(operator-, UnaSubtExprL_2)                 // X - scalar
EAD_PSEUDO_UNARY_OP_FUNCTION_R(operator-, UnaSubtExprR_2)                 // scalar - X

EAD_PSEUDO_UNARY_OP_CLASS_TYPE(operator/, UnaDiviExprL_2, X/a,  1.0/a        , 0.0)       // expr at left
EAD_PSEUDO_UNARY_OP_CLASS_TYPE(operator/, UnaDiviExprR_2, a/X, -(a/(X*X)), 2.0*a/(X*X*X)) // expr at right
EAD_PSEUDO_UNARY_OP_FUNCTION_L(operator/, UnaDiviExprL_2)                                 // X / scalar
EAD_PSEUDO_UNARY_OP_FUNCTION_R(operator/, UnaDiviExprR_2)                                 // scalar / X

EAD_PSEUDO_UNARY_OP_CLASS_TYPE(pow,  UnaPowExprL_2, std::pow(X,a), a*std::pow(X,a-1)         , a*(a-1.)*std::pow(X,a-2)             ) // expr at left
EAD_PSEUDO_UNARY_OP_CLASS_TYPE(pow,  UnaPowExprR_2, std::pow(a,X), std::pow(a,X)*std::log(a) , std::pow(a,X)*std::pow(std::log(a),2)) // expr at right
EAD_PSEUDO_UNARY_OP_FUNCTION_L(pow,  UnaPowExprL_2)                                                                                   // pow(X , scalar)
EAD_PSEUDO_UNARY_OP_FUNCTION_R(pow,  UnaPowExprR_2)                                                                                   // pow(scalar , X)

EAD_PSEUDO_UNARY_OP_CLASS_TYPE(fmod, UnaFmodExprL_2, std::fmod(X,a), 1.0 , 0.0 )                                                             // expr at left
EAD_PSEUDO_UNARY_OP_CLASS_TYPE(fmod, UnaFmodExprR_2, std::fmod(a,X), -(X==0.?0.: ((X<0)^(a<0)? std::ceil(a/X) : std::floor(a/X)) )*bar, 0.0) // expr at right
EAD_PSEUDO_UNARY_OP_FUNCTION_L(fmod, UnaFmodExprL_2)                                                                                         // fmod(X , scalar)
EAD_PSEUDO_UNARY_OP_FUNCTION_R(fmod, UnaFmodExprR_2)                                                                                         // fmod(scalar , X)


#undef EAD_PSEUDO_UNARY_OP_CLASS_TYPE
#undef EAD_PSEUDO_UNARY_OP_FUNCTION_L
#undef EAD_PSEUDO_UNARY_OP_FUNCTION_R
#undef X





// ~~                                                                 ~~
// ~~~~~~~~                                                     ~~~~~~~~
// ~~~~~~~~~~~~~~~~~~                                 ~~~~~~~~~~~~~~~~~~
// =====================================================================
//                                                                    ::
//                          UNARY OPERATORS                           ::
//                                                                    ::
// =====================================================================
// ~~~~~~~~~~~~~~~~~~                                 ~~~~~~~~~~~~~~~~~~
// ~~~~~~~~                                                     ~~~~~~~~
// ~~                                                                 ~~

#define X m_exp.val()
#define EAD_UNARY_OP_CLASS_TYPE(OP_FUN_NAME, OP_CLASS_NAME, VAL_RET, DEDX, D2EDX)                            \
  template<typename ExprT>                                                                                   \
  class OP_CLASS_NAME : public ExprWrapper2<OP_CLASS_NAME<ExprT> >                                           \
  {                                                                                                          \
  public:                                                                                                    \
    typedef typename ExprT::ValueT ValueT;                                                                   \
    typedef typename ExprT::FieldT FieldT;                                                                   \
    typedef typename ExprT::ValueT_CR ValueT_CR;                                                             \
    typedef typename ExprT::LeafType LeafType;                                                               \
    typedef typename ExprT::LeafData LeafData;                                                               \
                                                                                                             \
  private:                                                                                                   \
    ExprT const& m_exp;                                                                                      \
                                                                                                             \
  public:                                                                                                    \
                                                                                                             \
    static int const n_leaves  = ExprT::n_leaves;                                                              \
    static int const dtmp_size = n_leaves + ExprT::dtmp_size;                                                 \
                                                                                                             \
                                                                                                             \
    OP_CLASS_NAME(ExprT const& e_) : m_exp(e_)                                                               \
    { }                                                                                                      \
                                                                                                             \
    ValueT val() const                                                                                       \
    {return VAL_RET;}                                                                                        \
                                                                                                             \
    unsigned numVars() const                                                                                 \
    { return m_exp.numVars(); }                                                                              \
                                                                                                             \
    void getLeafsAndTempPartials(ValueT dtmp[], LeafData leaves[]) const                               \
    {                                                                                                        \
      m_exp.computeTempPartials(1.0, dtmp + n_leaves );                                                       \
      m_exp.getLeafsAndTempPartials(dtmp + n_leaves, leaves);                                                  \
    }                                                                                                        \
                                                                                                             \
    void computeTempPartials(ValueT_CR bar, ValueT dtmp[]) const                                             \
    {                                                                                                        \
      m_exp.computeTempPartials((DEDX)*bar, dtmp);                                                           \
    }                                                                                                        \
                                                                                                             \
    void computeHessianPartials(ValueT_CR bar, ValueT_CR bar2, LeafData leaves[], ValueT dtmp[], ValueT hessian_off_diag[], int csize) const \
    {                                                                                                                                        \
      ValueT const dedx  = DEDX;                                                                                                             \
      ValueT const d2edx = D2EDX;                                                                                                            \
      m_exp.computeHessianPartials(dedx*bar, d2edx*bar + dedx*dedx*bar2, leaves, dtmp+n_leaves, hessian_off_diag, csize);      \
    }                                                                                                                                        \
                                                                                                             \
  };                                                                                                         \
                                                                                                             \
  template<typename Expr>                                                                                    \
  inline                                                                                                     \
  OP_CLASS_NAME<Expr>                                                                                        \
  OP_FUN_NAME (ExprWrapper2<Expr> const& e_)                                                                 \
  {                                                                                                          \
    return OP_CLASS_NAME<Expr>(e_);                                                                          \
  }
//  ------------------------end EAD_UNARY_OP_CLASS_TYPE


EAD_UNARY_OP_CLASS_TYPE(operator+, UnaPlusExpr_2,   X,  1.0, 0.0)
EAD_UNARY_OP_CLASS_TYPE(operator-, UnaMinusExpr_2, -X, -1.0, 0.0)

EAD_UNARY_OP_CLASS_TYPE(cos  , UnaCosExpr_2  ,  std::cos(X)  , -std::sin(X)                 , -std::cos(X)                             )
EAD_UNARY_OP_CLASS_TYPE(sin  , UnaSinExpr_2  ,  std::sin(X)  ,  std::cos(X)                 , -std::sin(X)                             )
EAD_UNARY_OP_CLASS_TYPE(tan  , UnaTanExpr_2  ,  std::tan(X)  , (1./std::pow(std::cos(X),2)) , 2.0*std::sin(X)/std::pow(std::cos(X),3)  )
EAD_UNARY_OP_CLASS_TYPE(acos , UnaAcosExpr_2 ,  std::acos(X) , (-1./std::sqrt(1-X*X))       , -X/std::pow((1-X*X),1.5)                 )
EAD_UNARY_OP_CLASS_TYPE(asin , UnaAsinExpr_2 ,  std::asin(X) , ( 1./std::sqrt(1-X*X))       ,  X/std::pow((1-X*X),1.5)                 )
EAD_UNARY_OP_CLASS_TYPE(atan , UnaAtanExpr_2 ,  std::atan(X) , ( 1./(1+X*X))                , -(2.*X)/std::pow((X*X+1),2)              )

EAD_UNARY_OP_CLASS_TYPE(cosh , UnaCoshExpr_2 ,  std::cosh(X) ,  std::sinh(X)                , std::cosh(X)                             )
EAD_UNARY_OP_CLASS_TYPE(sinh , UnaSinhExpr_2 ,  std::sinh(X) ,  std::cosh(X)                , std::sinh(X)                             )
EAD_UNARY_OP_CLASS_TYPE(tanh , UnaTanhExpr_2 ,  std::tanh(X) , (1./std::pow(std::cosh(X),2)), -2.*std::tanh(X)/std::pow(std::cosh(X),2))

EAD_UNARY_OP_CLASS_TYPE(exp  , UnaExpExpr_2  ,  std::exp(X)  , std::exp(X)                  , std::exp(X)                              )
EAD_UNARY_OP_CLASS_TYPE(log  , UnaLogExpr_2  ,  std::log(X)  , 1.0/X                        , -1./std::pow(X,2)                        )
EAD_UNARY_OP_CLASS_TYPE(log10, UnaLog10Expr_2,  std::log10(X), 1.0/(X*std::log(10))         , -1./(std::pow(X,2)*std::log(10))         )

//EAD_UNARY_OP_CLASS_TYPE(sqrt , UnaSqrtExpr_2 ,  std::sqrt(X) , 1.0/(2.*std::sqrt(X))        , -1./(4.*std::pow(X,1.5))                 )

EAD_UNARY_OP_CLASS_TYPE(ceil , UnaCeilExpr_2 ,  std::ceil(X) , 0.                           , 0.                                       )
EAD_UNARY_OP_CLASS_TYPE(fabs , UnaFabsExpr_2 ,  (X<0)?-X:X   , (X==0?0.:((X<0.)?-1.:1.))    , 0.                                       )
EAD_UNARY_OP_CLASS_TYPE(abs  , UnaAbsExpr_2  ,  (X<0)?-X:X   , (X==0?0.:((X<0.)?-1.:1.))    , 0.                                       )
EAD_UNARY_OP_CLASS_TYPE(floor, UnaFloorExpr_2,  std::floor(X), 0.                           , 0.                                       )

#undef EAD_UNARY_OP_CLASS_TYPE







// Specialization
//EAD_UNARY_OP_CLASS_TYPE(sqrt , UnaSqrtExpr_2 ,  std::sqrt(X) , 1.0/(2.*std::sqrt(X))        , -1./(4.*std::pow(X,1.5))                 )
#define EAD_UNARY_OP_CLASS_TYPE(OP_FUN_NAME, OP_CLASS_NAME, VAL_RET, DEDX, D2EDX)
template<typename ExprT>
class UnaSqrtExpr_2 : public ExprWrapper2<UnaSqrtExpr_2<ExprT> >
{
public:
  typedef typename ExprT::ValueT ValueT;
  typedef typename ExprT::FieldT FieldT;
  typedef typename ExprT::ValueT_CR ValueT_CR;
  typedef typename ExprT::LeafType LeafType;
  typedef typename ExprT::LeafData LeafData;

private:
  ExprT const& m_exp;
  ValueT m_val;
public:

  static int const n_leaves  = ExprT::n_leaves;
  static int const dtmp_size = n_leaves + ExprT::dtmp_size;


  UnaSqrtExpr_2(ExprT const& e_) : m_exp(e_), m_val(std::sqrt(X))
  { }

  ValueT val() const
  {return m_val;}

  unsigned numVars() const
  { return m_exp.numVars(); }

  void getLeafsAndTempPartials(ValueT dtmp[], LeafData leaves[]) const
  {
    m_exp.computeTempPartials(1.0, dtmp + n_leaves );
    m_exp.getLeafsAndTempPartials(dtmp + n_leaves, leaves);
  }

  void computeTempPartials(ValueT_CR bar, ValueT dtmp[]) const
  {
    ValueT const dedx = 1.0/(2.*m_val);
    m_exp.computeTempPartials(dedx*bar, dtmp);
  }

  void computeHessianPartials(ValueT_CR bar, ValueT_CR bar2, LeafData leaves[], ValueT dtmp[], ValueT hessian_off_diag[], int csize) const
  {
    ValueT const dedx  =  1.0/(2.*m_val);
    ValueT const d2edx = -dedx/(2.*m_val*m_val);
    m_exp.computeHessianPartials(dedx*bar, d2edx*bar + dedx*dedx*bar2, leaves, dtmp+n_leaves, hessian_off_diag, csize);
  }

};

template<typename Expr>
inline
UnaSqrtExpr_2<Expr>
sqrt (ExprWrapper2<Expr> const& e_)
{
  return UnaSqrtExpr_2<Expr>(e_);
}



















#undef X





} // endnamespace


#endif // EAD2_HPP




