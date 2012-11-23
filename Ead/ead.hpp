#include <functional>
#include <vector>
#include <iostream>
//

namespace ead
{

// Essa foi a forma mais rapida de realizar o produto escalar
template<typename Expr, int Nleafs>
struct ExprAccum
{
  typedef typename Expr::ValueT ValueT;
  typedef typename Expr::LeafType LeafType;
  ValueT product; // = exp.dx(i)
  inline
  ExprAccum(ValueT partials[], LeafType const* leafs[], int i)
  {
    product  = partials[0] * leafs[0]->dx(i);
    product += ExprAccum<Expr, Nleafs-1>(partials+1, leafs+1, i).product;
  }
};

template<typename Expr>
struct ExprAccum<Expr,1>
{
  typedef typename Expr::ValueT ValueT;
  typedef typename Expr::LeafType LeafType;
  ValueT product;
  inline
  ExprAccum(ValueT partials[], LeafType const* leafs[], int i) : product(partials[0] * leafs[0]->dx(i))
  { }
};

// template<class A>
struct Expression
{
//  const A& cast() const {return static_cast<const A&>(*this);}
};

template<class T_, int Mnc_>
class DFad : public Expression
{
  typedef DFad Self;
public:
  typedef T_   ValueT;
  typedef DFad LeafType;
  static const int max_n_comps = Mnc_;
private:
  ValueT   m_val;             // value
  unsigned m_n_comps;
  ValueT   m_dx[max_n_comps]; // df/dui

  void resize(unsigned s)
  { m_n_comps = s;}

  void setZeros()
  {
    for (int i = 0; i < m_n_comps; ++i)
      m_dx[i] = ValueT(0.0);
  };

public:

  static int const n_leafs = 1;

  inline
  DFad(ValueT val, int n_comps) : m_val(val), m_n_comps(n_comps), m_dx()
  {
#ifdef EAD_DEBUG
    if (n_comps > max_n_comps)
      std::cout << "WARNING: num comps > max num comps\n";
#endif
  }

  ValueT  val() const { return m_val; }
  ValueT& val()       { return m_val; }
  ValueT  dx(unsigned i)  const { return m_dx[i]; }
  ValueT& dx(unsigned i)        { return m_dx[i]; }
  unsigned numComps() const {return m_n_comps;}
  void setDiff(int ith, int n_comps)
  {
    m_n_comps = n_comps;
    setZeros();
    dx(ith) = ValueT(1.0);
  }

  // bar = df/dterminal
  void computePartialsAndGetLeafs(ValueT bar, ValueT partials[], DFad const* leafs[]) const
  {
    partials[0] = bar;
    leafs[0] = this;
  }

  //////////////////////////////////////////////////////////////////////
  ////////////////// OPERATORS /////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////

#ifdef EAD_DEBUG
  #define EAD_OP_DEBUG                                                                           \
    if (numComps() != e.numComps())                                                             \
    {                                                                                           \
      std::cout << "ERROR: " <<numComps()<<" x "<< e.numComps()<<std::endl;     \
      throw;                                                                                    \
    }
#else
  #define EAD_OP_DEBUG
#endif


#define EAD_ASSIGN_OPS(OP, IMPL)                         \
  template<class Expr>                                   \
  Self& operator OP (Expr const& e)                      \
  {                                                      \
    EAD_OP_DEBUG                                         \
    ValueT partials[Expr::n_leafs];                      \
    DFad const* leafs[Expr::n_leafs];                    \
    e.computePartialsAndGetLeafs(1.0, partials, leafs);  \
    ValueT e_val = e.val();                              \
    ValueT e_dxi;                                        \
    for (unsigned i = 0; i<m_n_comps; ++i)               \
    {                                                    \
      e_dxi = ExprAccum<Expr, Expr::n_leafs>(partials, leafs, i).product; \
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
#undef EAD_OP_DEBUG
}; // end class DFad


template<typename ExprL, typename ExprR>
class MultExpr : public Expression
{

  ExprL const& m_expL;
  ExprR const& m_expR;

  const double m_valL;
  const double m_valR;

public:

  typedef typename ExprL::ValueT ValueT;
  typedef typename ExprL::LeafType LeafType;

  static int const n_leafs1 = ExprL::n_leafs;
  static int const n_leafs2 = ExprR::n_leafs;
  static int const n_leafs  = n_leafs1 + n_leafs2;

  MultExpr(ExprL const& lhs, ExprR const& rhs) : m_expL(lhs),
                                                 m_expR(rhs),
                                                 m_valL(lhs.val()),
                                                 m_valR(rhs.val())
  { }

  ValueT val() const
  { return m_valL*m_valR;}

  unsigned numComps() const
  { return m_expL.numComps(); }

  void computePartialsAndGetLeafs(ValueT bar, ValueT partials[], LeafType const* leafs[]) const
  {
    m_expL.computePartialsAndGetLeafs(bar*m_valR, partials, leafs);
    m_expR.computePartialsAndGetLeafs(bar*m_valL, partials + n_leafs1, leafs + n_leafs1);
  }

};

template<typename ExprL, typename ExprR>
MultExpr<ExprL, ExprR>
operator* (ExprL const& l, ExprR const& r)
{
  return MultExpr<ExprL, ExprR>(l,r);
}





} // endnamespace



