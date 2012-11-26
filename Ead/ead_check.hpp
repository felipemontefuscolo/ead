// This file is part of Ead, a lightweight C++ template library
// for automatic differentiation.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.




#ifndef EAD_CHECK_HPP
#define EAD_CHECK_HPP


#include <stdexcept>
#include <string>
#include <sstream>

void assertion_failed(std::string const& expr, std::string const& msg);


#define EAD_ASSERT(ok, mesg) if(!(ok)) \
                    assertion_failed(#ok, mesg, __LINE__, __FILE__, __PRETTY_FUNCTION__)

#ifdef EAD_DEBUG
  #define EAD_CHECK(ok, mesg) EAD_ASSERT(ok, mesg)
#else
  #define EAD_CHECK(ok, mesg) ((void)0)
#endif

void assertion_failed(std::string const& expr, std::string const& msg, int Line, const char File[], const char PrettyFunction[])
{
  std::stringstream what_arg;
  what_arg << "\nEAD_ERROR: "<< File << ": "<< Line <<": "<<msg<<"\n" <<
                "EAD_ERROR: "<< File << ": "<< Line <<": assertion '"<<expr<<"' failed\n" <<
                "EAD_ERROR: "<< File << ": in '"<< PrettyFunction <<"'"<<std::endl;
  
  throw std::runtime_error(what_arg.str());
}

#endif // EAD_CHECK
