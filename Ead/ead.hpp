// This file is part of Ead, a lightweight C++ template library
// for automatic differentiation.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef EAD_HPP
#define EAD_HPP


#include <functional>
#include <vector>
#include <ostream>
#include "ead_mpl.hpp"
#include "ead_check.hpp"
#include <cmath>

namespace ead
{


// Essa foi a forma mais rapida de realizar o produto escalar
// This class compute the ith derivative of an expression
template<typename Expr, int Nleafs>
struct ExprDxi
{
  typedef typename Expr::ValueT ValueT;
  typedef typename Expr::LeafType LeafType;
  ValueT result; // = exp.dx(i)
  inline
  ExprDxi(ValueT partials[], LeafType const* leafs[], int i)
  {
    result  = partials[0] * leafs[0]->dx(i);
    result += ExprDxi<Expr, Nleafs-1>(partials+1, leafs+1, i).result;
  }
};

template<typename Expr>
struct ExprDxi<Expr,1>
{
  typedef typename Expr::ValueT ValueT;
  typedef typename Expr::LeafType LeafType;
  ValueT result;
  inline
  ExprDxi(ValueT partials[], LeafType const* leafs[], int i) : result(partials[0] * leafs[0]->dx(i))
  { }
};

// all fad numbers and expressions inherits from this class.
template<class A>
struct ExprWrapper
{
  inline
  operator A const& () const
  { return *static_cast<A const*>(this);}
};

// AD type
template<typename T_, unsigned Mnc_>
class DFad : public ExprWrapper<DFad<T_, Mnc_> >
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
  static const unsigned max_n_comps = Mnc_;
private:

  // ------------ ATTRIBUTES --------------------------
  ValueT   m_val;             // value
  unsigned m_n_comps;
  ValueT   m_dx[max_n_comps]; // df/dui
  // ---------------------------------------------------


  void resize(unsigned s)
  { m_n_comps = s;}

  void setDxZeros()
  {
    for (unsigned i = 0; i < m_n_comps; ++i)
      m_dx[i] = ValueT(0.0);
  };

public:

  static int const n_leafs = 1;

  inline explicit
  DFad(ValueT_CR val=0, unsigned n_comps=0) : m_val(val), m_n_comps(n_comps), m_dx()
  {
    EAD_CHECK(n_comps <= max_n_comps, "num comps > max num comps");
  }

  inline
  DFad(ValueT_CR val, unsigned n_comps, unsigned ith) : m_val(val), m_n_comps(n_comps), m_dx()
  {
    EAD_CHECK(n_comps <= max_n_comps && ith < n_comps, "num comps > max num comps or ith >= n_comps");
    m_dx[ith] = 1.0;
  }

  template<typename T>
  DFad(ExprWrapper<T> const& e_)
  {
    T const& e (e_);
    m_val = e.val();
    m_n_comps = e.numComps();
    ValueT partials[T::n_leafs];
    DFad const* leafs[T::n_leafs];
    e.computePartialsAndGetLeafs(1.0, partials, leafs);
    ValueT e_dxi;
    for (unsigned i = 0; i<m_n_comps; ++i)
    {
      e_dxi = ExprDxi<T, T::n_leafs>(partials, leafs, i).result;
      dx(i) = e_dxi;
    }
  }

  ValueT&    val()            { return m_val; }
  ValueT&    dx(unsigned i=0) { return m_dx[i]; }

  ValueT_CR  val()            const { return m_val; }
  ValueT_CR  dx(unsigned i=0) const { return m_dx[i]; }

  unsigned numComps() const {return m_n_comps;}

  /// ith = -1 to set dx to zero
  void setDiff(int ith, unsigned n_comps)
  {
    m_n_comps = n_comps;
    setDxZeros();
    if (unsigned(ith) < n_comps)
      dx(ith) = ValueT(1.0);
  }

  // bar = df/dterminal
  void computePartialsAndGetLeafs(ValueT_CR bar, ValueT partials[], DFad const* leafs[]) const
  {
    partials[0] = bar;
    leafs[0] = this;
  }

//         ----------------------------------------------
//     -------------------------------------------------------
//------------------ ASSIGNS OPERATORS | EXPR VERSION ---------------
//     -------------------------------------------------------
//         ----------------------------------------------


#define EAD_ASSIGN_OPS(OP, IMPL)                         \
  template<class ExprT>                                  \
  Self& operator OP (ExprWrapper<ExprT> const& e_)          \
  {                                                      \
    ExprT const& e (e_);                                 \
    EAD_CHECK(numComps()==e.numComps(), "incompatible dimension"); \
    ValueT partials[ExprT::n_leafs];                     \
    DFad const* leafs[ExprT::n_leafs];                   \
    e.computePartialsAndGetLeafs(1.0, partials, leafs);  \
    ValueT e_val = e.val();                              \
    ValueT e_dxi;                                        \
    for (unsigned i = 0; i<m_n_comps; ++i)               \
    {                                                    \
      e_dxi = ExprDxi<ExprT, ExprT::n_leafs>(partials, leafs, i).result; \
      IMPL                                               \
    }                                                    \
    this->val() OP e_val;                                \
                                                         \
    return *this;                                        \
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
    for (unsigned i = 0; i < m_n_comps; ++i)
      this->dx(i) = 0;
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
    for (unsigned i = 0; i < m_n_comps; ++i)
      this->dx(i) *= z;
    return *this;
  }

  template<typename T>
  inline
  typename EnableIf<IsField<T,Self>::value, Self&>::type
  operator/= (T const& z)
  {
    this->val() /= z;
    for (int i = 0; i < m_n_comps; ++i)
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

#define EAD_RELAT_OP(OP)                                      \
template<typename L, typename R>                              \
inline                                                        \
bool operator OP (ExprWrapper<L> const& l, ExprWrapper<R> const& r) \
{                                                             \
  return L(l).val() OP R(r).val();                            \
}                                                             \
                                                              \
template<typename L, typename T>                              \
inline                                                        \
typename EnableIf<IsField<T,L>::value, bool >::type           \
operator OP (ExprWrapper<L> const& l, T const& r)                \
{                                                             \
  return L(l).val() OP r;                                     \
}                                                             \
                                                              \
template<typename T, typename R>                              \
inline                                                        \
typename EnableIf<IsField<T,R>::value, bool >::type           \
operator OP (T const& l, ExprWrapper<R> const& r)                \
{                                                             \
  return l OP R(r).val();                                     \
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
std::ostream& operator << (std::ostream& os, ExprWrapper<T> const& x_) {

  T const& x = T(x_);

  os << "(" << x.val() << " | " << x.dx(0);

  for (unsigned i=1; i< x.numComps(); i++) {
    os  << ", " << x.dx(i);
  }

  os << " )";
  return os;
}


// DUMMY_S is not used for anything
#define EAD_BINARY_OP(OP_FUN_NAME, OP_CLASS_NAME, VAL_RET, DEDL, DEDR)                                 \
template<typename ExprL, typename ExprR>                                                               \
class OP_CLASS_NAME : public ExprWrapper<OP_CLASS_NAME<ExprL,ExprR> >                                  \
{                                                                                                      \
public:                                                                                                \
  typedef typename ExprL::ValueT ValueT;                                                               \
  typedef typename ExprL::FieldT FieldT;                                                               \
  typedef typename ExprL::ValueT_CR ValueT_CR;                                                         \
  typedef typename ExprL::LeafType LeafType;                                                           \
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
  OP_CLASS_NAME(ExprL const& lhs, ExprR const& rhs) : m_expL(lhs),                                     \
                                                      m_expR(rhs),                                     \
                                                      x(lhs.val()),                                    \
                                                      y(rhs.val())                                     \
  { }                                                                                                  \
                                                                                                       \
  ValueT val() const                                                                                   \
  { return VAL_RET;}                                                                                   \
                                                                                                       \
  unsigned numComps() const                                                                            \
  { return m_expL.numComps(); }                                                                        \
                                                                                                       \
  void computePartialsAndGetLeafs(ValueT_CR bar, ValueT partials[], LeafType const* leafs[]) const     \
  {                                                                                                    \
    m_expL.computePartialsAndGetLeafs(DEDL, partials, leafs);                                          \
    m_expR.computePartialsAndGetLeafs(DEDR, partials + n_leafs1, leafs + n_leafs1);                    \
  }                                                                                                    \
                                                                                                       \
};                                                                                                     \
                                                                                                       \
template<typename L, typename R>                                                                       \
inline                                                                                                 \
OP_CLASS_NAME<L, R>                                                                                    \
OP_FUN_NAME (ExprWrapper<L> const& l, ExprWrapper<R> const& r)                                         \
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
  class OP_CLASS_NAME : public ExprWrapper<OP_CLASS_NAME<T,ExprT> >                                          \
  {                                                                                                          \
  public:                                                                                                    \
    typedef typename ExprT::ValueT ValueT;                                                                   \
    typedef typename ExprT::FieldT FieldT;                                                                   \
    typedef typename ExprT::ValueT_CR ValueT_CR;                                                             \
    typedef typename ExprT::LeafType LeafType;                                                               \
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
    OP_CLASS_NAME(T const& s_, ExprT const& e_) : m_exp(e_),                                                 \
                                                  a(s_)                                                      \
    { }                                                                                                      \
                                                                                                             \
    ValueT val() const                                                                                       \
    {return VAL_RET;}                                                                                        \
                                                                                                             \
    unsigned numComps() const                                                                                \
    { return m_exp.numComps(); }                                                                             \
                                                                                                             \
    void computePartialsAndGetLeafs(ValueT_CR bar, ValueT partials[], LeafType const* leafs[]) const         \
    {                                                                                                        \
      m_exp.computePartialsAndGetLeafs(DEDX, partials, leafs);                                               \
    }                                                                                                        \
                                                                                                             \
  };

#define EAD_PSEUDO_UNARY_OP_FUNCTION_L(OP_FUN_NAME, OP_CLASS_NAME)                                           \
  template<typename Expr, typename T>                                                                        \
  inline                                                                                                     \
  typename EnableIf<IsField<T,Expr>::value,OP_CLASS_NAME<T,Expr> >::type                                     \
  OP_FUN_NAME (ExprWrapper<Expr> const& l, T const& r)                                                       \
  {                                                                                                          \
    return OP_CLASS_NAME<T,Expr>(r,l);                                                                       \
  }


#define EAD_PSEUDO_UNARY_OP_FUNCTION_R(OP_FUN_NAME, OP_CLASS_NAME)                                           \
  template<typename Expr, typename T>                                                                        \
  inline                                                                                                     \
  typename EnableIf<IsField<T,Expr>::value,OP_CLASS_NAME<T,Expr> >::type                                     \
  OP_FUN_NAME (T const& l, ExprWrapper<Expr> const& r)                                                       \
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
  class OP_CLASS_NAME : public ExprWrapper<OP_CLASS_NAME<ExprT> >                                            \
  {                                                                                                          \
  public:                                                                                                    \
    typedef typename ExprT::ValueT ValueT;                                                                   \
    typedef typename ExprT::FieldT FieldT;                                                                   \
    typedef typename ExprT::ValueT_CR ValueT_CR;                                                             \
    typedef typename ExprT::LeafType LeafType;                                                               \
                                                                                                             \
  private:                                                                                                   \
    ExprT const& m_exp;                                                                                      \
                                                                                                             \
  public:                                                                                                    \
                                                                                                             \
    static int const n_leafs  = ExprT::n_leafs;                                                              \
                                                                                                             \
    OP_CLASS_NAME(ExprT const& e_) : m_exp(e_)                                                               \
    { }                                                                                                      \
                                                                                                             \
    ValueT val() const                                                                                       \
    {return VAL_RET;}                                                                                        \
                                                                                                             \
    unsigned numComps() const                                                                                \
    { return m_exp.numComps(); }                                                                             \
                                                                                                             \
    void computePartialsAndGetLeafs(ValueT_CR bar, ValueT partials[], LeafType const* leafs[]) const         \
    {                                                                                                        \
      m_exp.computePartialsAndGetLeafs(DEDX, partials, leafs);                                               \
    }                                                                                                        \
                                                                                                             \
  };                                                                                                         \
                                                                                                             \
  template<typename Expr>                                                                                    \
  inline                                                                                                     \
  OP_CLASS_NAME<Expr>                                                                                        \
  OP_FUN_NAME (ExprWrapper<Expr> const& e_)                                                                  \
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

EAD_UNARY_OP_CLASS_TYPE(exp  , UnaExpExpr  ,  std::exp(X)  , std::exp(X)*bar                  )
EAD_UNARY_OP_CLASS_TYPE(log  , UnaLogExpr  ,  std::log(X)  , bar/X                            )
EAD_UNARY_OP_CLASS_TYPE(log10, UnaLog10Expr,  std::log10(X), bar/(X*std::log(10))             )

EAD_UNARY_OP_CLASS_TYPE(sqrt , UnaSqrtExpr ,  std::sqrt(X) , bar/(2.*std::sqrt(X))            )

EAD_UNARY_OP_CLASS_TYPE(ceil , UnaCeilExpr ,  std::ceil(X) , 0.*bar                           )
EAD_UNARY_OP_CLASS_TYPE(fabs , UnaFabsExpr ,  (X<0)?-X:X   , (X==0?0.:((X<0.)?-1.:1.))*bar    )
EAD_UNARY_OP_CLASS_TYPE(abs  , UnaAbsExpr  ,  (X<0)?-X:X   , (X==0?0.:((X<0.)?-1.:1.))*bar    )
EAD_UNARY_OP_CLASS_TYPE(floor, UnaFloorExpr,  std::floor(X), 0.*bar                           )

#undef EAD_UNARY_OP_CLASS_TYPE
#undef X





} // endnamespace


#endif // EAD_HPP




