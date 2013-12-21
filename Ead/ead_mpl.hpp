// This file is part of Ead, a lightweight C++ template library
// for automatic differentiation.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.



#ifndef EAD_MPL_HPP
#define EAD_MPL_HPP

#include <tr1/type_traits>

namespace ead
{

template<bool, typename T = void>
struct EnableIf {};

template<typename T>
struct EnableIf<true, T> {
  typedef T type;
};

template< class T >
struct is_type {
  static const bool value = true;
};


// Usage: GetField<Foo>::type is Foo itself if Foo::FieldT doesn't exist, and
// is GetField<Foo::FieldT>::type otherwise
template<class U, class V = void>
struct GetField
{
  typedef U type;
};

template<class U>
struct GetField<U,
    typename EnableIf<is_type<typename U::ValueT>::value, void>::type>
{
  typedef typename GetField<typename U::ValueT>::type type;
};



template<class U, class V = void>
struct RetTrivial
{
  typedef U type;
  typedef U const_type;
  //static char const id = 'A';
};

template<class U>
struct RetTrivial<U,
    typename EnableIf<!std::tr1::has_trivial_constructor<U>::value || (sizeof(U)>sizeof(double)), void>::type>
{
  typedef U&       type;
  typedef U const& const_type;
  //static char const id = 'B';
};


template<typename T, typename Expr>
struct IsField
{
  typedef typename Expr::FieldT FieldT;
  static const bool value = std::tr1::is_arithmetic<T>::value ||
                            std::tr1::is_same<T,FieldT>::value;
};



template <typename T> int sign(T val) {
    return (0 < val) - (val < 0);
}

template<class Fad, class Value>
struct LeafData_
{
  Value partial;
  Fad const* ptr;
};

template<class Fad, class Value>
struct LeafData2_
{
  Value partial;
  Fad const* ptr;
  Value hes_dig;
};

// Essa foi a forma mais rapida de realizar o produto escalar
// This class compute the ith derivative of an expression
template<typename Fad, int Nleaves>
struct ExprDxi
{
  typedef typename Fad::ValueT ValueT;
  typedef typename Fad::LeafType LeafType;
  typedef typename Fad::LeafData LeafData;
  ValueT result; // = exp.dx(i)
  inline
  ExprDxi(LeafData const* leaves, int i)
  {
    result  = leaves[0].partial * leaves[0].ptr->dx(i);
    result += ExprDxi<Fad, Nleaves-1>(leaves+1, i).result;
  }
};

template<typename Fad>
struct ExprDxi<Fad,1>
{
  typedef typename Fad::ValueT ValueT;
  typedef typename Fad::LeafType LeafType;
  typedef typename Fad::LeafData LeafData;
  ValueT result;
  inline
  ExprDxi(LeafData const* leaves, int i) : result(leaves[0].partial * leaves[0].ptr->dx(i))
  { }
};




// Essa foi a forma mais rapida de realizar o produto escalar
// This class compute the ith derivative of an expression
template<typename Expr, int Nleaves>
struct Dot
{
  typedef typename Expr::ValueT ValueT;
  typedef typename Expr::LeafType LeafType;
  ValueT result; // = exp.dx(i)
  inline
  Dot(ValueT partials[], ValueT dxi[])
  {
    result  = partials[0] * dxi[0];
    result += Dot<Expr, Nleaves-1>(partials+1, dxi+1).result;
  }
};

template<typename Expr>
struct Dot<Expr,1>
{
  typedef typename Expr::ValueT ValueT;
  typedef typename Expr::LeafType LeafType;
  ValueT result;
  inline
  Dot(ValueT partials[],ValueT dxi[]) : result(partials[0] * dxi[0])
  { }
};





// Essa foi a forma mais rapida de realizar o produto escalar
// This class compute the ith derivative of an expression
template<typename Expr, int Nleaves>
struct Getter
{
  typedef typename Expr::ValueT ValueT;
  typedef typename Expr::LeafType LeafType;
  inline
  Getter(ValueT partials[], LeafType const* leaves[], int i)
  {
    partials[0] = leaves[0]->dx(i);
    Getter<Expr, Nleaves-1>(partials+1, leaves+1, i);
  }
};

template<typename Expr>
struct Getter<Expr,1>
{
  typedef typename Expr::ValueT ValueT;
  typedef typename Expr::LeafType LeafType;
  inline
  Getter(ValueT partials[], LeafType const* leaves[], int i)
  { partials[0] = leaves[0]->dx(i); }
};


} // endnamespace



#endif // EAD_MPL_HPP


