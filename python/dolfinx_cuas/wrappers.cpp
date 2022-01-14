// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINx_CUAS
//
// SPDX-License-Identifier:    MIT

#include <array.h>
#include <caster_petsc.h>
#include <dolfinx/la/petsc.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx_cuas/QuadratureRule.hpp>
#include <dolfinx_cuas/kernels_non_const_coefficient.hpp>
#include <dolfinx_cuas/kernelwrapper.h>
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

PYBIND11_MODULE(cpp, m)
{
  // Create module for C++ wrappers
  m.doc() = "DOLFINX Custom Assemblers Python interface";
#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  // Kernel wrapper class
  py::class_<cuas_wrappers::KernelWrapper<PetscScalar>,
             std::shared_ptr<cuas_wrappers::KernelWrapper<PetscScalar>>>(
      m, "KernelWrapper", "Wrapper for C++ integration kernels");

  // Quadrature rule class
  py::class_<dolfinx_cuas::QuadratureRule, std::shared_ptr<dolfinx_cuas::QuadratureRule>>(
      m, "QuadratureRule", "QuadratureRule object")
      .def(py::init<dolfinx::mesh::CellType, int, int, basix::quadrature::type>(),
           py::arg("cell_type"), py::arg("degree"), py::arg("dim"),
           py::arg("type") = basix::quadrature::type::Default)
      .def("points",
           [](dolfinx_cuas::QuadratureRule& self, int i)
           {
             if (std::size_t(i) >= self.points_ref().size())
               throw std::runtime_error("Entity index out of range");
             return dolfinx_wrappers::xt_as_pyarray(std::move(self.points()[i]));
           })
      .def("weights",
           [](dolfinx_cuas::QuadratureRule& self, int i)
           {
             if (std::size_t(i) >= self.weights_ref().size())
               throw std::runtime_error("Entity index out of range");
             return dolfinx_wrappers::as_pyarray(std::move(self.weights()[i]));
           });

  m.def("generate_surface_kernel",
        [](std::shared_ptr<const dolfinx::fem::FunctionSpace> V, dolfinx_cuas::Kernel type,
           dolfinx_cuas::QuadratureRule& quadrature_rule)
        {
          return cuas_wrappers::KernelWrapper(
              dolfinx_cuas::generate_surface_kernel<PetscScalar>(V, type, quadrature_rule));
        });
  m.def("generate_kernel",
        [](dolfinx_cuas::Kernel type, int p, int bs, dolfinx_cuas::QuadratureRule& q_rule)
        {
          return cuas_wrappers::KernelWrapper(
              dolfinx_cuas::generate_kernel<PetscScalar>(type, p, bs, q_rule));
        });
  m.def("generate_vector_kernel",
        [](std::shared_ptr<const dolfinx::fem::FunctionSpace> V, dolfinx_cuas::Kernel type,
           dolfinx_cuas::QuadratureRule& quadrature_rule)
        {
          return cuas_wrappers::KernelWrapper<PetscScalar>(
              dolfinx_cuas::generate_vector_kernel<PetscScalar>(V, type, quadrature_rule));
        });

  m.def("generate_coeff_kernel",
        [](dolfinx_cuas::Kernel type,
           std::vector<std::shared_ptr<const dolfinx::fem::Function<PetscScalar>>> coeffs, int p,
           dolfinx_cuas::QuadratureRule& q_rule)
        {
          return cuas_wrappers::KernelWrapper<PetscScalar>(
              dolfinx_cuas::generate_coeff_kernel<PetscScalar>(type, coeffs, p, q_rule));
        });
  m.def("generate_surface_vector_kernel",
        [](std::shared_ptr<const dolfinx::fem::FunctionSpace> V, dolfinx_cuas::Kernel type,
           dolfinx_cuas::QuadratureRule& quadrature_rule)
        {
          return cuas_wrappers::KernelWrapper<PetscScalar>(
              dolfinx_cuas::generate_surface_vector_kernel<PetscScalar>(V, type, quadrature_rule));
        });

  m.def("assemble_matrix",
        [](Mat A, std::shared_ptr<dolfinx::fem::FunctionSpace> V,
           const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<PetscScalar>>>& bcs,
           const py::array_t<std::int32_t, py::array::c_style>& active_cells,
           cuas_wrappers::KernelWrapper<PetscScalar>& kernel,
           const py::array_t<PetscScalar, py::array::c_style>& coeffs,
           const py::array_t<PetscScalar, py::array::c_style>& constants,
           dolfinx::fem::IntegralType type)
        {
          auto ker = kernel.get();
          dolfinx_cuas::assemble_matrix(
              dolfinx::la::petsc::Matrix::set_block_fn(A, ADD_VALUES), V, bcs,
              xtl::span<const std::int32_t>(active_cells.data(), active_cells.size()), ker,
              xtl::span<const PetscScalar>(coeffs.data(), coeffs.size()), coeffs.shape(1),
              xtl::span(constants.data(), constants.shape(0)), type);
        });
  m.def("assemble_vector",
        [](py::array_t<PetscScalar, py::array::c_style>& b,
           std::shared_ptr<dolfinx::fem::FunctionSpace> V,
           const py::array_t<std::int32_t, py::array::c_style>& active_cells,
           cuas_wrappers::KernelWrapper<PetscScalar>& kernel,
           const py::array_t<PetscScalar, py::array::c_style>& coeffs,
           const py::array_t<PetscScalar, py::array::c_style>& constants,
           dolfinx::fem::IntegralType type)
        {
          auto ker = kernel.get();
          dolfinx_cuas::assemble_vector<PetscScalar>(
              xtl::span(b.mutable_data(), b.shape(0)), V,
              xtl::span<const std::int32_t>(active_cells.data(), active_cells.size()), ker,
              xtl::span<const PetscScalar>(coeffs.data(), coeffs.size()), coeffs.shape(1),
              xtl::span(constants.data(), constants.shape(0)), type);
        });
  m.def("pack_coefficients",
        [](std::vector<std::shared_ptr<const dolfinx::fem::Function<PetscScalar>>> functions,
           py::array_t<std::int32_t, py::array::c_style>& entities)
        {
          const std::size_t shape_0 = entities.ndim() == 1 ? 1 : entities.shape(0);

          if (entities.ndim() == 1)
          {
            auto cell_span = xtl::span<const std::int32_t>(entities.data(), entities.size());
            auto [coeffs, cstride]
                = dolfinx_cuas::pack_coefficients<PetscScalar>(functions, cell_span);
            int shape0 = cstride == 0 ? 0 : coeffs.size() / cstride;
            return dolfinx_wrappers::as_pyarray(std::move(coeffs), std::array{shape0, cstride});
          }
          else if (entities.ndim() == 2)
          {
            auto ents = entities.unchecked();
            // FIXME: How to avoid copy here
            std::vector<std::pair<std::int32_t, int>> facets;
            facets.reserve(shape_0);
            for (std::size_t i = 0; i < shape_0; i++)
              facets.emplace_back(ents(i, 0), ents(i, 1));

            auto facet_span
                = xtl::span<const std::pair<std::int32_t, int>>(facets.data(), facets.size());
            auto [coeffs, cstride]
                = dolfinx_cuas::pack_coefficients<PetscScalar>(functions, facet_span);
            int shape0 = cstride == 0 ? 0 : coeffs.size() / cstride;
            return dolfinx_wrappers::as_pyarray(std::move(coeffs), std::array{shape0, cstride});
          }
          else if (entities.ndim() == 3)
          {
            auto ents = entities.unchecked();
            // FIXME: How to avoid copy here
            std::vector<std::tuple<std::int32_t, int, std::int32_t, int>> facets;
            facets.reserve(shape_0);
            for (std::size_t i = 0; i < shape_0; i++)
              facets.emplace_back(ents(i, 0, 0), ents(i, 0, 1), ents(i, 1, 0), ents(i, 1, 1));

            auto facet_span = xtl::span<const std::tuple<std::int32_t, int, std::int32_t, int>>(
                facets.data(), facets.size());
            auto [coeffs, cstride]
                = dolfinx_cuas::pack_coefficients<PetscScalar>(functions, facet_span);
            int shape0 = cstride == 0 ? 0 : coeffs.size() / cstride;
            return dolfinx_wrappers::as_pyarray(std::move(coeffs), std::array{shape0, cstride});
          }
          else
          {
            throw std::runtime_error("Unsupported entities");
          }
        });
  m.def(
      "compute_active_entities",
      [](std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
         py::array_t<std::int32_t, py::array::c_style>& entities,
         dolfinx::fem::IntegralType integral)
      {
        auto entity_span = xtl::span<const std::int32_t>(entities.data(), entities.size());
        std::variant<std::vector<std::int32_t>, std::vector<std::pair<std::int32_t, int>>,
                     std::vector<std::tuple<std::int32_t, int, std::int32_t, int>>>
            active_entities = dolfinx_cuas::compute_active_entities(mesh, entity_span, integral);
        py::array_t<std::int32_t> output = std::visit(
            [&](auto&& output)
            {
              using U = std::decay_t<decltype(output)>;
              if constexpr (std::is_same_v<U, std::vector<std::int32_t>>)
              {
                py::array_t<std::int32_t> domains(output.size(), output.data());
                return domains;
              }
              else if constexpr (std::is_same_v<U, std::vector<std::pair<std::int32_t, int>>>)
              {
                std::array<py::ssize_t, 2> shape = {py::ssize_t(output.size()), 2};
                py::array_t<std::int32_t> domains(shape);
                auto d = domains.mutable_unchecked<2>();
                for (py::ssize_t i = 0; i < d.shape(0); ++i)
                {
                  d(i, 0) = output[i].first;
                  d(i, 1) = output[i].second;
                }
                return domains;
              }
              else if constexpr (std::is_same_v<
                                     U,
                                     std::vector<std::tuple<std::int32_t, int, std::int32_t, int>>>)
              {
                std::array<py::ssize_t, 3> shape = {py::ssize_t(output.size()), 2, 2};
                py::array_t<std::int32_t> domains(shape);
                auto d = domains.mutable_unchecked<3>();
                for (py::ssize_t i = 0; i < d.shape(0); ++i)
                {
                  d(i, 0, 0) = std::get<0>(output[i]);
                  d(i, 0, 1) = std::get<1>(output[i]);
                  d(i, 1, 0) = std::get<2>(output[i]);
                  d(i, 1, 1) = std::get<3>(output[i]);
                }
                return domains;
              }
            },
            active_entities);
        return output;
      });

  py::enum_<dolfinx_cuas::Kernel>(m, "Kernel")
      .value("Mass", dolfinx_cuas::Kernel::Mass)
      .value("MassNonAffine", dolfinx_cuas::Kernel::MassNonAffine)
      .value("Stiffness", dolfinx_cuas::Kernel::Stiffness)
      .value("SymGrad", dolfinx_cuas::Kernel::SymGrad)
      .value("TrEps", dolfinx_cuas::Kernel::TrEps)
      .value("Normal", dolfinx_cuas::Kernel::Normal)
      .value("Rhs", dolfinx_cuas::Kernel::Rhs);
}
