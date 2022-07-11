// Copyright (C) 2021 Sarah Roggendorf
//
// This file is part of DOLFINx_CUAS
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "QuadratureRule.hpp"
#include "kernels.hpp"
#include "utils.hpp"
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/FunctionSpace.h>
namespace dolfinx_cuas
{

template <typename T>
kernel_fn<T> generate_vector_kernel(std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
                                    dolfinx_cuas::Kernel type, dolfinx_cuas::QuadratureRule& q_rule)
{
  auto mesh = V->mesh();

  // Get mesh info
  const int gdim = mesh->geometry().dim(); // geometrical dimension
  const int tdim = mesh->topology().dim(); // topological dimension

  const basix::FiniteElement basix_element = mesh_to_basix_element(mesh, tdim);
  const int num_coordinate_dofs = basix_element.dim();

  // Get quadrature rule for cell
  const dolfinx::mesh::CellType ct = mesh->topology().cell_type();
  const xt::xtensor<double, 2>& points = q_rule.points_ref()[0];
  const std::vector<double>& weights = q_rule.weights_ref()[0];

  // Create Finite element for test and trial functions and tabulate shape functions
  std::shared_ptr<const dolfinx::fem::FiniteElement> element = V->element();
  int bs = element->block_size();
  std::uint32_t ndofs_cell = element->space_dimension() / bs;
  xt::xtensor<double, 4> basis({1, weights.size(), ndofs_cell, bs});
  element->tabulate(basis, points, 0);
  xt::xtensor<double, 2> phi = xt::view(basis, 0, xt::all(), xt::all(), 0);

  // Get coordinate element from dolfinx
  std::array<std::size_t, 4> tab_shape = basix_element.tabulate_shape(1, points.shape(0));
  xt::xtensor<double, 4> coordinate_basis(tab_shape);
  basix_element.tabulate(1, points, coordinate_basis);

  const xt::xtensor<double, 2>& dphi0_c
      = xt::view(coordinate_basis, xt::range(1, tdim + 1), 0, xt::all(), 0);

  assert(ndofs_cell == static_cast<std::int32_t>(phi.shape(1)));

  // 1 * v * dx, v TestFunction
  // =====================================================================================
  kernel_fn<T> rhs = [=](T* b, const T* c, const T* w, const double* coordinate_dofs,
                         const int* entity_local_index, const std::uint8_t* quadrature_permutation)
  {
    // Get geometrical data
    xt::xtensor<double, 2> J = xt::zeros<double>({gdim, tdim});
    std::array<std::size_t, 2> shape = {num_coordinate_dofs, 3};
    // FIXME: These array should be views (when compute_jacobian doesn't use xtensor)
    const xt::xtensor<double, 2>& coord
        = xt::adapt(coordinate_dofs, 3 * num_coordinate_dofs, xt::no_ownership(), shape);

    // Compute Jacobian, its inverse and the determinant
    auto c_view = xt::view(coord, xt::all(), xt::range(0, gdim));
    dolfinx::fem::CoordinateElement::compute_jacobian(dphi0_c, c_view, J);
    double detJ = std::fabs(dolfinx::fem::CoordinateElement::compute_jacobian_determinant(J));

    // Main loop
    for (std::size_t q = 0; q < weights.size(); q++)
    {
      double w0 = weights[q] * detJ;
      for (int i = 0; i < ndofs_cell; i++)
        b[i] += w0 * phi.unchecked(q, i);
    }
  };
  switch (type)
  {
  case dolfinx_cuas::Kernel::Rhs:
    return rhs;
  default:
    throw std::runtime_error("Unrecognized kernel");
  }
}

template <typename T>
kernel_fn<T> generate_surface_vector_kernel(std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
                                            dolfinx_cuas::Kernel type,
                                            dolfinx_cuas::QuadratureRule& q_rule)
{

  auto mesh = V->mesh();

  // Get mesh info
  const int gdim = mesh->geometry().dim(); // geometrical dimension
  const int tdim = mesh->topology().dim(); // topological dimension
  const int fdim = tdim - 1;               // topological dimension of facet

  // Create coordinate elements for cell
  const basix::FiniteElement basix_element = mesh_to_basix_element(mesh, tdim);
  const int num_coordinate_dofs = basix_element.dim();

  // Get quadrature rule
  const dolfinx::mesh::CellType ft
      = dolfinx::mesh::cell_entity_type(mesh->topology().cell_type(), fdim, 0);
  // FIXME: For prisms this should be a vector of arrays and vectors
  const std::vector<xt::xtensor<double, 2>>& q_points = q_rule.points_ref();
  const std::vector<std::vector<double>>& q_weights = q_rule.weights_ref();

  const std::uint32_t num_facets = q_weights.size();

  // Structures needed for basis function tabulation
  // phi and grad(phi) and coordinate element derivative at quadrature points
  std::shared_ptr<const dolfinx::fem::FiniteElement> element = V->element();
  int bs = element->block_size();
  std::uint32_t num_local_dofs = element->space_dimension() / bs;
  std::vector<xt::xtensor<double, 2>> phi;
  phi.reserve(num_facets);
  std::vector<xt::xtensor<double, 3>> dphi;
  phi.reserve(num_facets);
  std::vector<xt ::xtensor<double, 3>> dphi_c;
  dphi_c.reserve(num_facets);

  // Tabulate basis functions (for test/trial function) and coordinate element at
  // quadrature points
  for (int i = 0; i < num_facets; ++i)
  {
    const xt::xtensor<double, 2>& q_facet = q_points[i];
    const int num_quadrature_points = q_facet.shape(0);
    xt::xtensor<double, 4> cell_tab({tdim + 1, num_quadrature_points, num_local_dofs, bs});

    // Tabulate at quadrature points on facet
    element->tabulate(cell_tab, q_facet, 1);
    xt::xtensor<double, 2> phi_i = xt::view(cell_tab, 0, xt::all(), xt::all(), 0);
    phi.push_back(phi_i);
    xt::xtensor<double, 3> dphi_i
        = xt::view(cell_tab, xt::range(1, tdim + 1), xt::all(), xt::all(), 0);
    dphi.push_back(dphi_i);

    // Tabulate coordinate element of reference cell
    std::array<std::size_t, 4> tab_shape = basix_element.tabulate_shape(1, q_facet.shape(0));
    xt::xtensor<double, 4> c_tab(tab_shape);
    basix_element.tabulate(1, q_facet, c_tab);
    xt::xtensor<double, 3> dphi_ci
        = xt::view(c_tab, xt::range(1, tdim + 1), xt::all(), xt::all(), 0);
    dphi_c.push_back(dphi_ci);
  }

  // As reference facet and reference cell are affine, we do not need to compute this per
  // quadrature point
  auto [ref_jac, jac_shape] = basix::cell::facet_jacobians(basix_element.cell_type());

  // Define kernels
  // v*ds, v TestFunction
  // =====================================================================================
  kernel_fn<T> rhs_surface
      = [=](T* b, const T* c, const T* w, const double* coordinate_dofs,
            const int* entity_local_index, const std::uint8_t* quadrature_permutation)
  {
    std::size_t facet_index = size_t(*entity_local_index);

    // Reshape coordinate dofs to two dimensional array
    // NOTE: DOLFINx has 3D input coordinate dofs
    std::array<std::size_t, 2> shape = {num_coordinate_dofs, 3};
    const xt::xtensor<double, 2>& coord
        = xt::adapt(coordinate_dofs, num_coordinate_dofs * 3, xt::no_ownership(), shape);

    // Extract the first derivative of the coordinate element (cell) of degrees of freedom on
    // the facet
    const xt::xtensor<double, 2>& dphi0_c
        = xt::view(dphi_c[facet_index], xt::all(), 0,
                   xt::all()); // FIXME: Assumed constant, i.e. only works for simplices

    // NOTE: Affine cell assumption
    // Compute Jacobian and determinant at first quadrature point
    xt::xtensor<double, 2> J = xt::zeros<double>({gdim, tdim});
    auto c_view = xt::view(coord, xt::all(), xt::range(0, gdim));
    dolfinx::fem::CoordinateElement::compute_jacobian(dphi0_c, c_view, J);

    // Compute det(J_C J_f) as it is the mapping to the reference facet
    xt::xtensor<double, 2> J_f = xt::zeros<double>({jac_shape[1], jac_shape[2]});
    for (std::size_t i = 0; i < jac_shape[1]; ++i)
      for (std::size_t j = 0; j < jac_shape[2]; ++j)
        J_f(i, j) = ref_jac[facet_index * jac_shape[1] * jac_shape[2] + i * jac_shape[1] + j];
    xt::xtensor<double, 2> J_tot = xt::zeros<double>({J.shape(0), J_f.shape(1)});
    dolfinx::math::dot(J, J_f, J_tot);
    double detJ = std::fabs(dolfinx::fem::CoordinateElement::compute_jacobian_determinant(J_tot));

    // Get number of dofs per cell
    const std::vector<double>& weights = q_weights[facet_index];
    const xt::xtensor<double, 2>& phi_f = phi[facet_index];

    // Loop over quadrature points
    for (std::size_t q = 0; q < weights.size(); q++)
    {
      // Scale at each quadrature point
      const double w0 = weights[q] * detJ;

      for (int i = 0; i < num_local_dofs; i++)
      {
        // Compute a weighted phi_i(p_q),  i.e. phi_i(p_q) det(J) w_q
        double w1 = w0 * phi_f(q, i);
        // Insert over block size in matrix
        b[i] += w1;
      }
    }
  };
  switch (type)
  {
  case dolfinx_cuas::Kernel::Rhs:
    return rhs_surface;
  default:
    throw std::runtime_error("Unrecognized kernel");
  }
}
} // namespace dolfinx_cuas
