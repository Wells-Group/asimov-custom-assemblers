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
      .def("pack_gap_plane", [](dolfinx_cuas::contact::Contact& self, int origin_meshtag, double g)
           { return dolfinx_cuas_wrappers::as_pyarray2d(self.pack_gap_plane(origin_meshtag, g)); })
      .def("pack_gap", [](dolfinx_cuas::contact::Contact& self, int origin_meshtag)
           { return dolfinx_cuas_wrappers::as_pyarray2d(self.pack_gap(origin_meshtag)); })
      .def("map_0_to_1", &dolfinx_cuas::contact::Contact::map_0_to_1)
      .def("map_1_to_0", &dolfinx_cuas::contact::Contact::map_1_to_0)
      .def("facet_0", &dolfinx_cuas::contact::Contact::facet_0)
      .def("facet_1", &dolfinx_cuas::contact::Contact::facet_1)
      .def("set_quadrature_degree", &dolfinx_cuas::contact::Contact::set_quadrature_degree);
  m.def(
      "generate_contact_kernel",
      [](std::shared_ptr<const dolfinx::fem::FunctionSpace> V, dolfinx_cuas::contact::Kernel type,
         dolfinx_cuas::QuadratureRule& q_rule,
         std::vector<std::shared_ptr<const dolfinx::fem::Function<PetscScalar>>> coeffs,
         bool constant_normal)
      {
        return cuas_wrappers::KernelWrapper(dolfinx_cuas::contact::generate_contact_kernel(
            V, type, q_rule, coeffs, constant_normal));
      },
      py::arg("V"), py::arg("kernel_type"), py::arg("quadrature_rule"), py::arg("coeffs"),
      py::arg("constant_normal") = true);
  py::enum_<dolfinx_cuas::contact::Kernel>(m, "Kernel")
      .value("NitscheRigidSurfaceRhs", dolfinx_cuas::contact::Kernel::NitscheRigidSurfaceRhs)
      .value("NitscheRigidSurfaceJac", dolfinx_cuas::contact::Kernel::NitscheRigidSurfaceJac);
}
} // namespace dolfinx_cuas_wrappers