// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINx_CUAS
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "array.h"
#include "caster_petsc.h"
#include "kernelwrapper.h"
#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx_cuas/kernels_non_const_coefficient.hpp>
#include <dolfinx_cuas/matrix_assembly.hpp>
#include <dolfinx_cuas/surface_kernels.hpp>
#include <dolfinx_cuas/utils.hpp>
#include <dolfinx_cuas/vector_assembly.hpp>
#include <dolfinx_cuas/vector_kernels.hpp>
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
void contact(py::module& m);
}

PYBIND11_MODULE(cpp, m)
{
  // Create module for C++ wrappers
  m.doc() = "DOLFINX Custom Assemblers Python interface";
#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  // Create contact submodule [contact]
  py::module contact = m.def_submodule("contact", "contact module");
  dolfinx_cuas_wrappers::contact(contact);

  py::class_<cuas_wrappers::KernelWrapper, std::shared_ptr<cuas_wrappers::KernelWrapper>>(
      m, "KernelWrapper", "Wrapper for C++ integration kernels");

  m.def("generate_surface_kernel",
        [](std::shared_ptr<const dolfinx::fem::FunctionSpace> V, dolfinx_cuas::Kernel type,
           int quadrature_degree)
        {
          return cuas_wrappers::KernelWrapper(
              dolfinx_cuas::generate_surface_kernel(V, type, quadrature_degree));
        });
  m.def("generate_kernel", [](dolfinx_cuas::Kernel type, int p, int bs)
        { return cuas_wrappers::KernelWrapper(dolfinx_cuas::generate_kernel(type, p, bs)); });
  m.def("generate_vector_kernel",
        [](std::shared_ptr<const dolfinx::fem::FunctionSpace> V, dolfinx_cuas::Kernel type,
           int quad_degree)
        {
          return cuas_wrappers::KernelWrapper(
              dolfinx_cuas::generate_vector_kernel(V, type, quad_degree));
        });
  m.def("generate_coeff_kernel",
        [](dolfinx_cuas::Kernel type,
           std::vector<std::shared_ptr<const dolfinx::fem::Function<PetscScalar>>> coeffs, int p,
           int q) {
          return cuas_wrappers::KernelWrapper(
              dolfinx_cuas::generate_coeff_kernel(type, coeffs, p, q));
        });
  m.def("generate_surface_vector_kernel",
        [](std::shared_ptr<const dolfinx::fem::FunctionSpace> V, dolfinx_cuas::Kernel type,
           int quad_degree)
        {
          return cuas_wrappers::KernelWrapper(
              dolfinx_cuas::generate_surface_vector_kernel(V, type, quad_degree));
        });

  m.def("assemble_matrix",
        [](Mat A, std::shared_ptr<dolfinx::fem::FunctionSpace> V,
           const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<PetscScalar>>>& bcs,
           const py::array_t<std::int32_t, py::array::c_style>& active_cells,
           cuas_wrappers::KernelWrapper& kernel,
           const py::array_t<PetscScalar, py::array::c_style>& coeffs,
           const py::array_t<PetscScalar, py::array::c_style>& constants,
           dolfinx::fem::IntegralType type)
        {
          dolfinx::array2d<PetscScalar> _coeffs(coeffs.shape()[0], coeffs.shape()[1]);
          std::copy_n(coeffs.data(), coeffs.size(), _coeffs.data());
          auto ker = kernel.get();
          dolfinx_cuas::assemble_matrix(
              dolfinx::la::PETScMatrix::set_block_fn(A, ADD_VALUES), V, bcs,
              xtl::span<const std::int32_t>(active_cells.data(), active_cells.size()), ker, _coeffs,
              xtl::span(constants.data(), constants.shape(0)), type);
        });
  m.def("assemble_vector",
        [](py::array_t<PetscScalar, py::array::c_style>& b,
           std::shared_ptr<dolfinx::fem::FunctionSpace> V,
           const py::array_t<std::int32_t, py::array::c_style>& active_cells,
           cuas_wrappers::KernelWrapper& kernel,
           const py::array_t<PetscScalar, py::array::c_style>& coeffs,
           const py::array_t<PetscScalar, py::array::c_style>& constants,
           dolfinx::fem::IntegralType type)
        {
          dolfinx::array2d<PetscScalar> _coeffs(coeffs.shape()[0], coeffs.shape()[1]);
          std::copy_n(coeffs.data(), coeffs.size(), _coeffs.data());
          auto ker = kernel.get();
          dolfinx_cuas::assemble_vector(
              xtl::span(b.mutable_data(), b.shape(0)), V,
              xtl::span<const std::int32_t>(active_cells.data(), active_cells.size()), ker, _coeffs,
              xtl::span(constants.data(), constants.shape(0)), type);
        });
  m.def("pack_coefficients",
        [](std::vector<std::shared_ptr<const dolfinx::fem::Function<PetscScalar>>> coeffs)
        { return dolfinx_cuas_wrappers::as_pyarray2d(dolfinx_cuas::pack_coefficients(coeffs)); });

  py::enum_<dolfinx_cuas::Kernel>(m, "Kernel")
      .value("Mass", dolfinx_cuas::Kernel::Mass)
      .value("MassNonAffine", dolfinx_cuas::Kernel::MassNonAffine)
      .value("Stiffness", dolfinx_cuas::Kernel::Stiffness)
      .value("SymGrad", dolfinx_cuas::Kernel::SymGrad)
      .value("TrEps", dolfinx_cuas::Kernel::TrEps)
      .value("Normal", dolfinx_cuas::Kernel::Normal)
      .value("Rhs", dolfinx_cuas::Kernel::Rhs);
}
