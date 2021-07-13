// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
#include "array.h"
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx_cuas/assembly.hpp>
#include <dolfinx_cuas/contact/Contact.hpp>
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
  py::class_<dolfinx_cuas::contact::Contact, std::shared_ptr<dolfinx_cuas::contact::Contact>>(
      m, "Contact", "Contact object")
      .def(py::init<std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>>, int, int,
                    std::shared_ptr<dolfinx::fem::FunctionSpace>>(),
           py::arg("marker"), py::arg("suface_0"), py::arg("surface_1"), py::arg("V"))
      .def("create_distance_map",
           [](dolfinx_cuas::contact::Contact& self, int origin_meshtag)
           {
             self.create_distance_map(origin_meshtag);
             return;
           })
      .def("map_0_to_1", &dolfinx_cuas::contact::Contact::map_0_to_1)
      .def("map_1_to_0", &dolfinx_cuas::contact::Contact::map_1_to_0)
      .def("facet_0", &dolfinx_cuas::contact::Contact::facet_0)
      .def("facet_1", &dolfinx_cuas::contact::Contact::facet_1)
      .def("generate_surface_kernel", &dolfinx_cuas::contact::Contact::generate_surface_kernel);
  m.def("assemble_exterior_facets", &dolfinx_cuas::assemble_exterior_facets);
  m.def("assemble_cells", &dolfinx_cuas::assemble_cells);
  py::enum_<dolfinx_cuas::Kernel>(m, "Kernel")
      .value("Mass", dolfinx_cuas::Kernel::Mass)
      .value("Stiffness", dolfinx_cuas::Kernel::Stiffness)
      .value("Contact_Jac", dolfinx_cuas::Kernel::Contact_Jac);
}
