// Copyright (C) 2021 Sarah Roggendorf
//
// This file is part of DOLFINx_CUAS
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "../array.h"
#include "../caster_petsc.h"
#include "../kernelwrapper.h"
#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx_cuas/contact/Contact.hpp>
#include <dolfinx_cuas/contact/contact_kernels.hpp>
#include <iostream>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <xtl/xspan.hpp>
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace dolfinx_cuas_wrappers
{
void contact(py::module& m)
{
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
      .def("facet_1", &dolfinx_cuas::contact::Contact::facet_1);
  m.def("generate_rhs_kernel",
        [](std::shared_ptr<const dolfinx::fem::FunctionSpace> V, dolfinx_cuas::contact::Kernel type,
           int quad_degree,
           std::vector<std::shared_ptr<const dolfinx::fem::Function<PetscScalar>>> coeffs)
        {
          return cuas_wrappers::KernelWrapper(
              dolfinx_cuas::contact::generate_rhs_kernel(V, type, quad_degree, coeffs));
        });
  py::enum_<dolfinx_cuas::contact::Kernel>(m, "Kernel")
      .value("NitscheRigidSurface", dolfinx_cuas::contact::Kernel::NitscheRigidSurface);
}
} // namespace dolfinx_cuas_wrappers