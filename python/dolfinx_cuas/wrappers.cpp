// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <iostream>
#include <pybind11/pybind11.h>
#include <dolfinx.h>
#include <dolfinx_cuas/assemble.hpp>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;


PYBIND11_MODULE(cpp, m)
{
  // Create module for C++ wrappers
  m.doc() = "DOLFINX Custom Assemblers Python interfacess";
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
  m.def("assemble_matrix", &dolfinx_cuas::assemble_matrix);

}
