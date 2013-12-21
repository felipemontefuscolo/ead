// This file is part of Ead, a lightweight C++ template library
// for automatic differentiation.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef EAD_HPP
#define EAD_HPP


#include <functional>
#include <ostream>
#include "ead_mpl.hpp"
#include "ead_check.hpp"
#include <cmath>

namespace ead
{



// all fad numbers and expressions inherits from this class.
template<class A>
struct ExprWrapper1
{
  inline
  operator A const& () const
  { return *static_cast<A const*>(this);}
};

// AD type
template<typename T_, unsigned Mnc_>
class DFad : public ExprWrapper1<DFad<T_, Mnc_> >
{
  typedef DFad Self;
public:
  typedef T_   ValueT; // can be a base type or Dfad itself.
  typedef typename GetField<ValueT>::type FieldT;// FieldT is the base
                                                 // type, e.g., double,
                                                 // float, complex, etc.
  // Returns ValueT if it has trivial constructor,
  // and returns ValueT const& otherwise.
  typedef typename RetTrivial<ValueT>::const_type ValueT_CR;

  typedef DFad LeafType;

  typedef LeafData_<Self, ValueT> LeafData;

  static const unsigned max_n_comps = Mnc_;
private:

  // ------------ ATTRIBUTES --------------------------
  ValueT   m_val;             // value
  unsigned m_n_vars;
  ValueT   m_dx[max_n_comps]; // df/dui
  // ---------------------------------------------------


  void resize(unsigned s)
  { m_n_vars = s;}

public:

  static int const n_leafs = 1;

  inline explicit
  DFad(ValueT_CR val=0, unsigned n_comps=0) : m_val(val), m_n_vars(n_comps), m_dx()
  {
    EAD_CHECK(n_comps <= max_n_comps, "num comps > max num comps");
  }

  inline
  DFad(ValueT_CR val, unsigned n_comps, unsigned ith) : m_val(val), m_n_vars(n_comps), m_dx()
  {
    EAD_CHECK(n_comps <= max_n_comps && ith < n_comps, "num comps > max num comps or ith >= n_comps");
    m_dx[ith] = 1.0;
  }

  template<typename T>
  inline
  DFad(ExprWrapper1<T> const& e_)
  {
    T const& e (e_);
    m_val = e.val();
    m_n_vars = e.numVars();
    LeafData leaves[T::n_leafs];
    e.computePartialsAndGetLeaves(1.0, leaves);
    ValueT e_dxi;
    for (unsigned i = 0; i<m_n_vars; ++i)
    {
      e_dxi = ExprDxi<Self, T::n_leafs>(leaves, i).result;
      dx(i) = e_dxi;
    }
  }

  inline ValueT&    val()            { return m_val; }
  inline ValueT&    dx(unsigned i=0) { return m_dx[i]; }

  inline ValueT_CR  val()            const { return m_val; }
  inline ValueT_CR  dx(unsigned i=0) const { return m_dx[i]; }

  unsigned  numVars() const {return m_n_vars;}

  void setDiff(unsigned ith, unsigned n_comps)
  {
    m_n_vars = n_comps;
    resetDerivatives();
    dx(ith) = ValueT(1.0);
  }

  void setDiff(unsigned ith)
  {
    resetDerivatives();
    dx(ith) = ValueT(1.0);
  }

  void setNumVars(unsigned n_comps)
  {
    m_n_vars = n_comps;
  }

  inline void resetDerivatives()
  {
    for (unsigned i = 0; i < m_n_vars; ++i)
      m_dx[i] = ValueT(0.0);
  };

  // TODO: not for users, put it in a private area
  // bar = df/dterminal
  inline void computePartialsAndGetLeaves(ValueT_CR bar, LeafData leaves[]) const
  {
    leaves[0].partial = bar;
    leaves[0].ptr = this;
  }

//         ----------------------------------------------
//     -------------------------------------------------------
//------------------ ASSIGNS OPERATORS | EXPR VERSION ---------------
//     -------------------------------------------------------
//         ----------------------------------------------


#define EAD_ASSIGN_OPS(OP, IMPL)                                         \
  template<class ExprT>                                                  \
  inline                                                                 \
  Self& operator OP (ExprWrapper1<ExprT> const& e_)                      \
  {                                                                      \
    ExprT const& e (e_);                                                 \
    EAD_CHECK(numVars()==e.numVars(), "incompatible dimension");         \
    LeafData leaves[ExprT::n_leafs];                                     \
    e.computePartialsAndGetLeaves(1.0, leaves);                          \
    ValueT e_val = e.val();                                              \
    ValueT e_dxi;                                                        \
    for (unsigned i = 0; i<m_n_vars; ++i)                                \
    {                                                                    \
      e_dxi = ExprDxi<Self, ExprT::n_leafs>(leaves, i).result;           \
      IMPL                                                               \
    }                                                                    \
    this->val() OP e_val;                                                \
                                                                         \
    return *this;                                                        \
                                                                         \
  }

  EAD_ASSIGN_OPS( =, dx(i) =  e_dxi;)
  EAD_ASSIGN_OPS(+=, dx(i) += e_dxi;)
  EAD_ASSIGN_OPS(-=, dx(i) -= e_dxi;)
  EAD_ASSIGN_OPS(*=, dx(i) *= e_val;
                     dx(i) += e_dxi*val();)
  EAD_ASSIGN_OPS(/=, dx(i) = (dx(i)*e_val - val()*e_dxi)/(e_val*e_val); )

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
    this->val() = z;
    resetDerivatives();
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
      this->dx(i) *= z;
    return *this;
  }

  template<typename T>
  inline
  typename EnableIf<IsField<T,Self>::value, Self&>::type
  operator/= (T const& z)
  {
    this->val() /= z;
    for (int i = 0; i < m_n_vars; ++i)
      this->dx(i) /= z;
    return *this;
  }


}; // ------------ end class DFad ------------------




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
bool operator OP (ExprWrapper1<L> const& l_, ExprWrapper1<R> const& r_) \
{                                                                       \
  L const& l (l_);                                                      \
  R const& r (r_);                                                      \
  return l.val() OP r.val();                                            \
}                                                                       \
                                                                        \
template<typename L, typename T>                                        \
inline                                                                  \
typename EnableIf<IsField<T,L>::value, bool >::type                     \
operator OP (ExprWrapper1<L> const& l_, T const& r)                     \
{                                                                       \
  L const& l (l_);                                                      \
  return l.val() OP r;                                                  \
}                                                                       \
                                                                        \
template<typename T, typename R>                                        \
inline                                                                  \
typename EnableIf<IsField<T,R>::value, bool >::type                     \
operator OP (T const& l, ExprWrapper1<R> const& r_)                     \
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
std::ostream& operator << (std::ostream& os, ExprWrapper1<T> const& x_) {

  T const& x = T(x_);

  os << "(" << x.val() << " | " << x.dx(0);

  for (unsigned i=1; i< x.numVars(); i++) {
    os  << ", " << x.dx(i);
  }

  os << " )";
  return os;
}


// DUMMY_S is not used for anything
#define EAD_BINARY_OP(OP_FUN_NAME, OP_CLASS_NAME, VAL_RET, DEDL, DEDR)                                 \
template<typename ExprL, typename ExprR>                                                               \
class OP_CLASS_NAME : public ExprWrapper1<OP_CLASS_NAME<ExprL,ExprR> >                                 \
{                                                                                                      \
public:                                                                                                \
  typedef typename ExprL::ValueT ValueT;                                                               \
  typedef typename ExprL::FieldT FieldT;                                                               \
  typedef typename ExprL::ValueT_CR ValueT_CR;                                                         \
  typedef typename ExprL::LeafType LeafType;                                                           \
  typedef typename ExprL::LeafData LeafData;                                                           \
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
  static int const n_leafs1 = ExprL::n_leafs;                                                          \
  static int const n_leafs2 = ExprR::n_leafs;                                                          \
  static int const n_leafs  = n_leafs1 + n_leafs2;                                                     \
                                                                                                       \
  inline                                                                                               \
  OP_CLASS_NAME(ExprL const& lhs, ExprR const& rhs) : m_expL(lhs),                                     \
                                                      m_expR(rhs),                                     \
                                                      x(lhs.val()),                                    \
                                                      y(rhs.val())                                     \
  { }                                                                                                  \
                                                                                                       \
  inline                                                                                               \
  ValueT val() const                                                                                   \
  { return VAL_RET;}                                                                                   \
                                                                                                       \
  inline                                                                                               \
  unsigned numVars() const                                                                             \
  { return m_expL.numVars(); }                                                                         \
                                                                                                       \
  inline                                                                                               \
  void computePartialsAndGetLeaves(ValueT_CR bar, LeafData leaves[]) const                             \
  {                                                                                                    \
    m_expL.computePartialsAndGetLeaves(DEDL, leaves);                                                  \
    m_expR.computePartialsAndGetLeaves(DEDR, leaves + n_leafs1);                                       \
  }                                                                                                    \
                                                                                                       \
};                                                                                                     \
                                                                                                       \
template<typename L, typename R>                                                                       \
inline                                                                                                 \
OP_CLASS_NAME<L, R>                                                                                    \
OP_FUN_NAME (ExprWrapper1<L> const& l, ExprWrapper1<R> const& r)                                       \
{                                                                                                      \
  return OP_CLASS_NAME<L, R>(l,r);                                                                     \
}
// END EAD_BINARY_OP MACRO ----------------------------------------------------------------------------


EAD_BINARY_OP(operator+, BinAddiExpr, x + y, bar       , bar     )
EAD_BINARY_OP(operator-, BinSubtExpr, x - y, bar       ,-bar     )
EAD_BINARY_OP(operator*, BinMultExpr, x*y  , bar*y, x*bar        )
EAD_BINARY_OP(operator/, BinDiviExpr, x/y  , bar/y,(-x/(y*y))*bar)

EAD_BINARY_OP(max,  BinMaxExpr , (x<y)?y:x     , ((x==y? .5 :  (x<y)?0:1))*bar, ((x==y? .5 :  (x<y)?1:0))*bar)
EAD_BINARY_OP(min,  BinMinExpr ,!(x<y)?y:x     , ((x==y? .5 : !(x<y)?0:1))*bar, ((x==y? .5 : !(x<y)?1:0))*bar)
EAD_BINARY_OP(pow,  BinPowExpr , std::pow(x,y) , std::pow(x,y-1)*y*bar        , std::pow(x,y)*std::log(x)*bar)
EAD_BINARY_OP(fmod, BinFmodExpr, std::fmod(x,y), bar                          ,-(y==0.?0.: ((y<0)^(x<0)? std::ceil(x/y) : std::floor(x/y)) )*bar)

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
#define EAD_PSEUDO_UNARY_OP_CLASS_TYPE(OP_FUN_NAME, OP_CLASS_NAME, VAL_RET, DEDX)                            \
  template<typename T, typename ExprT>                                                                       \
  class OP_CLASS_NAME : public ExprWrapper1<OP_CLASS_NAME<T,ExprT> >                                         \
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
    static int const n_leafs  = ExprT::n_leafs;                                                              \
                                                                                                             \
    inline                                                                                                   \
    OP_CLASS_NAME(T const& s_, ExprT const& e_) : m_exp(e_),                                                 \
                                                  a(s_)                                                      \
    { }                                                                                                      \
                                                                                                             \
    inline                                                                                                   \
    ValueT val() const                                                                                       \
    {return VAL_RET;}                                                                                        \
                                                                                                             \
    inline                                                                                                   \
    unsigned numVars() const                                                                                 \
    { return m_exp.numVars(); }                                                                              \
                                                                                                             \
    inline                                                                                                   \
    void computePartialsAndGetLeaves(ValueT_CR bar, LeafData leaves[]) const                                 \
    {                                                                                                        \
      m_exp.computePartialsAndGetLeaves(DEDX, leaves);                                                       \
    }                                                                                                        \
                                                                                                             \
  };

#define EAD_PSEUDO_UNARY_OP_FUNCTION_L(OP_FUN_NAME, OP_CLASS_NAME)                                           \
  template<typename Expr, typename T>                                                                        \
  inline                                                                                                     \
  typename EnableIf<IsField<T,Expr>::value,OP_CLASS_NAME<T,Expr> >::type                                     \
  OP_FUN_NAME (ExprWrapper1<Expr> const& l, T const& r)                                                      \
  {                                                                                                          \
    return OP_CLASS_NAME<T,Expr>(r,l);                                                                       \
  }


#define EAD_PSEUDO_UNARY_OP_FUNCTION_R(OP_FUN_NAME, OP_CLASS_NAME)                                           \
  template<typename Expr, typename T>                                                                        \
  inline                                                                                                     \
  typename EnableIf<IsField<T,Expr>::value,OP_CLASS_NAME<T,Expr> >::type                                     \
  OP_FUN_NAME (T const& l, ExprWrapper1<Expr> const& r)                                                      \
  {                                                                                                          \
    return OP_CLASS_NAME<T,Expr>(l,r);                                                                       \
  }

EAD_PSEUDO_UNARY_OP_CLASS_TYPE(operator+, UnaAddiExpr, X+a, bar       ) // case:
EAD_PSEUDO_UNARY_OP_FUNCTION_L(operator+, UnaAddiExpr)                  // X + scalar
EAD_PSEUDO_UNARY_OP_FUNCTION_R(operator+, UnaAddiExpr)                  // scalar + X

EAD_PSEUDO_UNARY_OP_CLASS_TYPE(operator*, UnaMultExpr, X*a, bar*a) // case:
EAD_PSEUDO_UNARY_OP_FUNCTION_L(operator*, UnaMultExpr)             // X*scalar
EAD_PSEUDO_UNARY_OP_FUNCTION_R(operator*, UnaMultExpr)             // scalar*X

EAD_PSEUDO_UNARY_OP_CLASS_TYPE(max, UnaMaxExpr, (X<a)?a:X,  (  X==a ? 0.5 : ((X<a)?0.:1.) )*bar ) // case:
EAD_PSEUDO_UNARY_OP_FUNCTION_L(max, UnaMaxExpr)                                                  // max(X,scalar)
EAD_PSEUDO_UNARY_OP_FUNCTION_R(max, UnaMaxExpr)                                                  // max(scalar,X)

EAD_PSEUDO_UNARY_OP_CLASS_TYPE(min, UnaMinExpr, !(X<a)?a:X, (  X==a ? 0.5 : (!(X<a)?0.:1.) )*bar ) // case:
EAD_PSEUDO_UNARY_OP_FUNCTION_L(min, UnaMinExpr)                                                    // min(X,scalar)
EAD_PSEUDO_UNARY_OP_FUNCTION_R(min, UnaMinExpr)                                                    // min(scalar,X)


// UnaDiviExpr and UnaSubtExpr are not symmetric,
// therefore must be implemented separately.

EAD_PSEUDO_UNARY_OP_CLASS_TYPE(operator-, UnaSubtExprL, X-a,  bar) // expr at left
EAD_PSEUDO_UNARY_OP_CLASS_TYPE(operator-, UnaSubtExprR, a-X, -bar) // expr at right
EAD_PSEUDO_UNARY_OP_FUNCTION_L(operator-, UnaSubtExprL)            // X - scalar
EAD_PSEUDO_UNARY_OP_FUNCTION_R(operator-, UnaSubtExprR)            // scalar - X

EAD_PSEUDO_UNARY_OP_CLASS_TYPE(operator/, UnaDiviExprL, X/a,  bar/a        ) // expr at left
EAD_PSEUDO_UNARY_OP_CLASS_TYPE(operator/, UnaDiviExprR, a/X, -(a/(X*X))*bar) // expr at right
EAD_PSEUDO_UNARY_OP_FUNCTION_L(operator/, UnaDiviExprL)                      // X / scalar
EAD_PSEUDO_UNARY_OP_FUNCTION_R(operator/, UnaDiviExprR)                      // scalar / X

EAD_PSEUDO_UNARY_OP_CLASS_TYPE(pow,  UnaPowExprL, std::pow(X,a), a*std::pow(X,a-1)*bar        ) // expr at left
EAD_PSEUDO_UNARY_OP_CLASS_TYPE(pow,  UnaPowExprR, std::pow(a,X), std::pow(a,X)*std::log(a)*bar) // expr at right
EAD_PSEUDO_UNARY_OP_FUNCTION_L(pow,  UnaPowExprL)                                               // pow(X , scalar)
EAD_PSEUDO_UNARY_OP_FUNCTION_R(pow,  UnaPowExprR)                                               // pow(scalar , X)

EAD_PSEUDO_UNARY_OP_CLASS_TYPE(fmod, UnaFmodExprL, std::fmod(X,a), bar                 ) // expr at left
EAD_PSEUDO_UNARY_OP_CLASS_TYPE(fmod, UnaFmodExprR, std::fmod(a,X), -(X==0.?0.: ((X<0)^(a<0)? std::ceil(a/X) : std::floor(a/X)) )*bar) // expr at right
EAD_PSEUDO_UNARY_OP_FUNCTION_L(fmod, UnaFmodExprL)                                       // fmod(X , scalar)
EAD_PSEUDO_UNARY_OP_FUNCTION_R(fmod, UnaFmodExprR)                                       // fmod(scalar , X)


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
#define EAD_UNARY_OP_CLASS_TYPE(OP_FUN_NAME, OP_CLASS_NAME, VAL_RET, DEDX)                                   \
  template<typename ExprT>                                                                                   \
  class OP_CLASS_NAME : public ExprWrapper1<OP_CLASS_NAME<ExprT> >                                           \
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
    static int const n_leafs  = ExprT::n_leafs;                                                              \
                                                                                                             \
    inline                                                                                                   \
    OP_CLASS_NAME(ExprT const& e_) : m_exp(e_)                                                               \
    { }                                                                                                      \
                                                                                                             \
    inline                                                                                                   \
    ValueT val() const                                                                                       \
    {return VAL_RET;}                                                                                        \
                                                                                                             \
    inline                                                                                                   \
    unsigned numVars() const                                                                                 \
    { return m_exp.numVars(); }                                                                              \
                                                                                                             \
    inline                                                                                                   \
    void computePartialsAndGetLeaves(ValueT_CR bar, LeafData leaves[]) const                                 \
    {                                                                                                        \
      m_exp.computePartialsAndGetLeaves(DEDX, leaves);                                                       \
    }                                                                                                        \
                                                                                                             \
  };                                                                                                         \
                                                                                                             \
  template<typename Expr>                                                                                    \
  inline                                                                                                     \
  OP_CLASS_NAME<Expr>                                                                                        \
  OP_FUN_NAME (ExprWrapper1<Expr> const& e_)                                                                 \
  {                                                                                                          \
    return OP_CLASS_NAME<Expr>(e_);                                                                          \
  }
//  ------------------------end EAD_UNARY_OP_CLASS_TYPE


EAD_UNARY_OP_CLASS_TYPE(operator+, UnaPlusExpr,   X,  bar)
EAD_UNARY_OP_CLASS_TYPE(operator-, UnaMinusExpr, -X, -bar)

EAD_UNARY_OP_CLASS_TYPE(cos  , UnaCosExpr  ,  std::cos(X)  , -std::sin(X)*bar                 )
EAD_UNARY_OP_CLASS_TYPE(sin  , UnaSinExpr  ,  std::sin(X)  ,  std::cos(X)*bar                 )
EAD_UNARY_OP_CLASS_TYPE(tan  , UnaTanExpr  ,  std::tan(X)  , (1./std::pow(std::cos(X),2))*bar )
EAD_UNARY_OP_CLASS_TYPE(acos , UnaAcosExpr ,  std::acos(X) , (-1./std::sqrt(1-X*X))*bar       )
EAD_UNARY_OP_CLASS_TYPE(asin , UnaAsinExpr ,  std::asin(X) , ( 1./std::sqrt(1-X*X))*bar       )
EAD_UNARY_OP_CLASS_TYPE(atan , UnaAtanExpr ,  std::atan(X) , ( 1./(1+X*X))*bar                )

EAD_UNARY_OP_CLASS_TYPE(cosh , UnaCoshExpr ,  std::cosh(X) ,  std::sinh(X)*bar                )
EAD_UNARY_OP_CLASS_TYPE(sinh , UnaSinhExpr ,  std::sinh(X) ,  std::cosh(X)*bar                )
EAD_UNARY_OP_CLASS_TYPE(tanh , UnaTanhExpr ,  std::tanh(X) , (1./std::pow(std::cosh(X),2))*bar)

//EAD_UNARY_OP_CLASS_TYPE(exp  , UnaExpExpr  ,  std::exp(X)  , std::exp(X)*bar                  ) // specialized
EAD_UNARY_OP_CLASS_TYPE(log  , UnaLogExpr  ,  std::log(X)  , bar/X                            )
EAD_UNARY_OP_CLASS_TYPE(log10, UnaLog10Expr,  std::log10(X), bar/(X*std::log(10))             )

//EAD_UNARY_OP_CLASS_TYPE(sqrt , UnaSqrtExpr ,  std::sqrt(X) , bar/(2.*std::sqrt(X))            )

EAD_UNARY_OP_CLASS_TYPE(ceil , UnaCeilExpr ,  std::ceil(X) , 0.*bar                           )
EAD_UNARY_OP_CLASS_TYPE(fabs , UnaFabsExpr ,  (X<0)?-X:X   , (X==0?0.:((X<0.)?-1.:1.))*bar    )
EAD_UNARY_OP_CLASS_TYPE(abs  , UnaAbsExpr  ,  (X<0)?-X:X   , (X==0?0.:((X<0.)?-1.:1.))*bar    )
EAD_UNARY_OP_CLASS_TYPE(floor, UnaFloorExpr,  std::floor(X), 0.*bar                           )

#undef EAD_UNARY_OP_CLASS_TYPE
//#undef X

// These class are implemented separately because they need some optimization
template<typename ExprT>
class UnaExpExpr : public ExprWrapper1<UnaExpExpr<ExprT> >
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

  static int const n_leafs  = ExprT::n_leafs;

  inline
  UnaExpExpr(ExprT const& e_) : m_exp(e_), m_val(std::exp(X))
  { }

  inline
  ValueT val() const
  {return m_val;}

  inline
  unsigned numVars() const
  { return m_exp.numVars(); }

  inline
  void computePartialsAndGetLeaves(ValueT_CR bar, LeafData leaves[]) const
  {
    m_exp.computePartialsAndGetLeaves(m_val*bar, leaves);
  }

};

template<typename Expr>
inline
UnaExpExpr<Expr>
exp (ExprWrapper1<Expr> const& e_)
{
  return UnaExpExpr<Expr>(e_);
}



template<typename ExprT>
class UnaSqrtExpr : public ExprWrapper1<UnaSqrtExpr<ExprT> >
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

  static int const n_leafs  = ExprT::n_leafs;

  inline
  UnaSqrtExpr(ExprT const& e_) : m_exp(e_), m_val(std::sqrt(X))
  { }

  inline
  ValueT val() const
  {return m_val;}

  inline
  unsigned numVars() const
  { return m_exp.numVars(); }

  inline
  void computePartialsAndGetLeaves(ValueT_CR bar, LeafData leaves[]) const
  {
    m_exp.computePartialsAndGetLeaves(bar/(2*m_val), leaves);
  }

};

template<typename Expr>
inline
UnaSqrtExpr<Expr>
sqrt (ExprWrapper1<Expr> const& e_)
{
  return UnaSqrtExpr<Expr>(e_);
}





#undef X


} // endnamespace


#endif // EAD_HPP




