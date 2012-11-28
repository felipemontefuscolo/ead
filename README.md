EAD: Easy Automatic Differentiation
Author: Felipe Montefuscolo
contact: felipe.mt87 (at) gmail.com
location: Brazil, SP

===================================

EAD is a compact and efficient automatic differentiation package
written in C++ language. It's based on operator overload to compute
the gradient of a function in forward mode.
For more information about AD, see [this link](http://en.wikipedia.org/wiki/Automatic_differentiation).

The implementation follow the idea of the paper
["Efficient Expression Templates for Operator Overloading-based Automatic Differentiation"](http://arxiv.org/abs/1205.3506),
which is implemented in the Trilinos package [Sacado](http://trilinos.sandia.gov/packages/sacado/).

The main reasons that have led me to develop a new package are:
* a less restrictive license;
* an independent library;
* a library specialized in forward differentiation;

EAD also tries to be efficient, compact and easy to use.

This library has been tested only under Linux with gcc v4.6.

Any help to improve, extend, maintain,  ..., is Welcome!

