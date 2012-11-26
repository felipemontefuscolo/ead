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
#include <iostream>
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
struct ExprWrap
{
  inline
  operator A const& () const
  { return *static_cast<A const*>(this);}
};

// AD type
template<typename T_, unsigned Mnc_>
class DFad : public ExprWrap<DFad<T_, Mnc_> >
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
    for (int i = 0; i < m_n_comps; ++i)
      m_dx[i] = ValueT(0.0);
  };

public:

  static int const n_leafs = 1;

  inline explicit
  DFad(ValueT_CR val=ValueT(0), unsigned n_comps=0) : m_val(val), m_n_comps(n_comps), m_dx()
  {
    EAD_CHECK(n_comps <= max_n_comps, "num comps > max num comps");
  }

  inline
  DFad(ValueT_CR val, unsigned n_comps, unsigned ith) : m_val(val), m_n_comps(n_comps), m_dx()
  {
    EAD_CHECK(n_comps <= max_n_comps && ith < n_comps, "num comps > max num comps or ith >= n_comps");
    m_dx(ith) = ValueT(1.0);
  }

  ValueT&    val()          { return m_val; }
  ValueT&    dx(unsigned i) { return m_dx[i]; }

  ValueT_CR  val()          const { return m_val; }
  ValueT_CR  dx(unsigned i) const { return m_dx[i]; }

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
  Self& operator OP (ExprWrap<ExprT> const& e_)          \
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

  Self& operator= (FieldT const& z)
  {
    this->val() = z;
    return *this;
  }

  Self& operator+= (FieldT const& z)
  {
    this->val() += z;
    return *this;
  }

  Self& operator-= (FieldT const& z)
  {
    this->val() -= z;
    return *this;
  }

  Self& operator*= (FieldT const& z)
  {
    this->val() *= z;
    for (int i = 0; i < m_n_comps; ++i)
      this->dx(i) *= z;
    return *this;
  }

  Self& operator/= (FieldT const& z)
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
//                         BINARY OPERATORS                           ::
//                                                                    ::
// =====================================================================
// ~~~~~~~~~~~~~~~~~~                                 ~~~~~~~~~~~~~~~~~~
// ~~~~~~~~                                                     ~~~~~~~~
// ~~                                                                 ~~
                                                     


// DUMMY_S is not used for anything
#define EAD_BINARY_OP(OP_FUN_NAME, OP_CLASS_NAME, VAL_RET, DEDL, DEDR)                                 \
template<typename ExprL, typename ExprR>                                                               \
class OP_CLASS_NAME : public ExprWrap<OP_CLASS_NAME<ExprL,ExprR> >                                     \
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
                                                m_expR(rhs),                                           \
                                                x(lhs.val()),                                          \
                                                y(rhs.val())                                           \
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
OP_FUN_NAME (ExprWrap<L> const& l, ExprWrap<R> const& r)                                               \
{                                                                                                      \
  return OP_CLASS_NAME<L, R>(l,r);                                                                     \
}
// END EAD_BINARY_OP MACRO ----------------------------------------------------------------------------


EAD_BINARY_OP(operator+, BinAddiExpr, x + y, bar       , bar     )
EAD_BINARY_OP(operator-, BinSubtExpr, x - y, bar       ,-bar     )
EAD_BINARY_OP(operator*, BinMultExpr, x*y  , bar*y, x*bar        )
EAD_BINARY_OP(operator/, BinDiviExpr, x/y  , bar/y,(-x/(y*y))*bar)

EAD_BINARY_OP(max, BinMaxExpr, (x<y)?y:x, (x<y)?ValueT(0):bar, (x<y)?bar:ValueT(0))
EAD_BINARY_OP(min, BinMinExpr,!(x<y)?y:x,!(x<y)?ValueT(0):bar,!(x<y)?bar:ValueT(0))

EAD_BINARY_OP(pow, BinPowExpr, std::pow(x,y), std::pow(x,y-1)*y, std::pow(x,y)*log(x))

#undef EAD_BINARY_OP


// ~~                                                                 ~~
// ~~~~~~~~                                                     ~~~~~~~~
// ~~~~~~~~~~~~~~~~~~                                 ~~~~~~~~~~~~~~~~~~
// =====================================================================
//                                                                    ::
//                         UNARY OPERATORS                            ::
//                                                                    ::
// =====================================================================
// ~~~~~~~~~~~~~~~~~~                                 ~~~~~~~~~~~~~~~~~~
// ~~~~~~~~                                                     ~~~~~~~~
// ~~                                                                 ~~


// OBS: Binary operators involving scalars is considered unary operators
// here, e.g., X+scalar, scalar+X, scalar*X, X*scalar, etc.


#define EAD_PSEUDO_UNARY_OP_CLASS_TYPE(OP_SYMBOL, DUMMY_S, OP_NAME, VAL_RET, DEDX)                           \
  template<typename ExprT>                                                                                   \
  class OP_NAME : public ExprWrap<OP_NAME<ExprT> >                                                           \
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
    FieldT const& m_sval; /* scalar value */                                                                 \
                                                                                                             \
  public:                                                                                                    \
                                                                                                             \
    static int const n_leafs  = ExprT::n_leafs;                                                              \
                                                                                                             \
    OP_NAME(FieldT const& s_, ExprT const& e_) : m_exp(e_),                                                  \
                                                 m_sval(s_)                                                  \
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

#define EAD_PSEUDO_UNARY_OP_OPERATOR_L(OP_SYMBOL, DUMMY_S, OP_NAME)                                          \
  template<typename Expr>                                                                                    \
  inline                                                                                                     \
  OP_NAME<Expr>                                                                                              \
  operator OP_SYMBOL (ExprWrap<Expr> const& l, typename Expr::FieldT const& r)                               \
  {                                                                                                          \
    return OP_NAME<Expr>(r,l);                                                                               \
  }


#define EAD_PSEUDO_UNARY_OP_OPERATOR_R(OP_SYMBOL, DUMMY_S, OP_NAME)                                          \
  template<typename Expr>                                                                                    \
  inline                                                                                                     \
  OP_NAME<Expr>                                                                                              \
  operator OP_SYMBOL (typename Expr::FieldT const& l, ExprWrap<Expr> const& r)                               \
  {                                                                                                          \
    return OP_NAME<Expr>(l,r);                                                                               \
  }

EAD_PSEUDO_UNARY_OP_CLASS_TYPE(+, +, UnaAddiExpr, m_exp.val()+m_sval, bar       ) // case:
EAD_PSEUDO_UNARY_OP_OPERATOR_L(+, +, UnaAddiExpr)                                 // X + scalar
EAD_PSEUDO_UNARY_OP_OPERATOR_R(+, +, UnaAddiExpr)                                 // scalar + X

EAD_PSEUDO_UNARY_OP_CLASS_TYPE(*, *, UnaMultExpr, m_exp.val()*m_sval, bar*m_sval) // case:
EAD_PSEUDO_UNARY_OP_OPERATOR_L(*, *, UnaMultExpr)                                 // X*scalar
EAD_PSEUDO_UNARY_OP_OPERATOR_R(*, *, UnaMultExpr)                                 // scalar*X

// UnaDiviExpr and UnaSubtExpr are not symmetric,
// therefore must be implemented separately.

EAD_PSEUDO_UNARY_OP_CLASS_TYPE(-, -, UnaSubtExprL, m_exp.val()-m_sval,  bar) // expr at left
EAD_PSEUDO_UNARY_OP_CLASS_TYPE(-, -, UnaSubtExprR, m_sval-m_exp.val(), -bar) // expr at right
EAD_PSEUDO_UNARY_OP_OPERATOR_L(-, -, UnaSubtExprL)                           // X - scalar
EAD_PSEUDO_UNARY_OP_OPERATOR_R(-, -, UnaSubtExprR)                           // scalar - X

EAD_PSEUDO_UNARY_OP_CLASS_TYPE(/, /, UnaDiviExprL, m_exp.val()/m_sval,  bar/m_sval                            ) // expr at left
EAD_PSEUDO_UNARY_OP_CLASS_TYPE(/, /, UnaDiviExprR, m_sval/m_exp.val(), -(m_sval/(m_exp.val()*m_exp.val()))*bar) // expr at right
EAD_PSEUDO_UNARY_OP_OPERATOR_L(/, /, UnaDiviExprL)                                                              // X / scalar
EAD_PSEUDO_UNARY_OP_OPERATOR_R(/, /, UnaDiviExprR)                                                              // scalar / X

#undef EAD_PSEUDO_UNARY_OP_CLASS_TYPE
#undef EAD_PSEUDO_UNARY_OP_OPERATOR_L
#undef EAD_PSEUDO_UNARY_OP_OPERATOR_R

} // endnamespace
#endif // EAD_HPP




