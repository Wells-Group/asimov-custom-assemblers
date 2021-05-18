// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
#include "array.h"
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx_cuas/Contact.hpp>
#include <dolfinx_cuas/build_gap_function.hpp>
#include <dolfinx_cuas/utils.hpp>
#include <iostream>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <xtl/xspan.hpp>

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
  m.def("test_func", &dolfinx_cuas::test_func);
  m.def("facet_master_puppet_relation",
        [](const std::shared_ptr<dolfinx::mesh::Mesh>& Mesh,
           const py::array_t<std::int32_t, py::array::c_style>& puppet_facets,
           const py::array_t<std::int32_t, py::array::c_style>& candidate_facets,
           int quadrature_degree) {
          return dolfinx_cuas::contact::facet_master_puppet_relation(
              Mesh, xtl::span<const std::int32_t>(puppet_facets.data(), puppet_facets.size()),
              xtl::span<const std::int32_t>(candidate_facets.data(), candidate_facets.size()),
              quadrature_degree);
        });
  py::class_<dolfinx_cuas::contact::Contact, std::shared_ptr<dolfinx_cuas::contact::Contact>>(
      m, "Contact", "Contact object")
      .def(py::init<std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>>, int, int>(),
           py::arg("marker"), py::arg("suface_0"), py::arg("surface_1"));
}
