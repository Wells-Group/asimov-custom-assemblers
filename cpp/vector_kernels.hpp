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
  namespace stdex = std::experimental;
  using cmdspan4_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 4>>;
  using mdspan4_t = stdex::mdspan<double, stdex::dextents<std::size_t, 4>>;
  using mdspan2_t = stdex::mdspan<double, stdex::dextents<std::size_t, 2>>;
  using cmdspan2_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
  using mdspan3_t = stdex::mdspan<double, stdex::dextents<std::size_t, 3>>;
  using cmdspan3_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 3>>;

  auto mesh = V->mesh();
  assert(mesh);

  // Get mesh info
  const int gdim = mesh->geometry().dim(); // geometrical dimension
  const int tdim = mesh->topology().dim(); // topological dimension
  const dolfinx::fem::CoordinateElement cmap = mesh->geometry().cmap();
  const int num_coordinate_dofs = cmap.dim();

  // Get quadrature rule for cell
  const dolfinx::mesh::CellType ct = mesh->topology().cell_type();

  const xt::xtensor<double, 2>& points = q_rule.points_ref().front();
  const std::vector<double>& weights = q_rule.weights_ref().front();
  const std::size_t num_quadrature_points = points.shape(0);

  // Create Finite element for test and trial functions and tabulate shape functions
  std::shared_ptr<const dolfinx::fem::FiniteElement> element = V->element();
  int bs = element->block_size();
  std::uint32_t ndofs_cell = element->space_dimension() / bs;

  // Tabulate basis of test/trial functions
  const std::array<std::size_t, 4> e_shape
      = element->basix_element().tabulate_shape(0, num_quadrature_points);
  assert(e_shape.back() == 1);
  std::vector<double> basis(std::reduce(e_shape.begin(), e_shape.end(), 1, std::multiplies()));
  element->tabulate(basis, std::span(points.data(), points.size()), points.shape(), 0);

  // Tabulate basis of coordinate element
  std::array<std::size_t, 4> c_shape = cmap.tabulate_shape(1, points.shape(0));
  std::vector<double> coordinate_basis(
      std::reduce(c_shape.begin(), c_shape.end(), 1, std::multiplies()));
  cmap.tabulate(1, std::span(points.data(), points.size()), points.shape(), coordinate_basis);

  const bool is_affine = cmap.is_affine();
  const std::array<std::size_t, 2> cd_shape = {num_coordinate_dofs, 3};
  // 1 * v * dx, v TestFunction
  // =====================================================================================
  kernel_fn<T> rhs = [=](T* b, const T* c, const T* w, const double* coordinate_dofs,
                         const int* entity_local_index, const std::uint8_t* quadrature_permutation)
  {
    // Reshape and truncate 3D coordinate dofs
    cmdspan2_t coords(coordinate_dofs, cd_shape);
    auto c_view = stdex::submdspan(coords, stdex::full_extent, std::pair{0, gdim});

    assert(is_affine);

    // Compute Jacobian and determinant
    std::vector<double> Jb(gdim * tdim);
    std::vector<double> Kb(tdim * gdim);
    mdspan2_t J(Jb.data(), gdim, tdim);
    mdspan2_t K(Kb.data(), tdim, gdim);
    std::vector<double> detJ_scratch(2 * gdim * tdim);
    cmdspan4_t phi_c_full(coordinate_basis.data(), c_shape);
    auto dphi_c_0 = stdex::submdspan(phi_c_full, std::pair(1, tdim + 1), 0, stdex::full_extent, 0);
    dolfinx::fem::CoordinateElement::compute_jacobian(dphi_c_0, c_view, J);
    const double detJ
        = std::fabs(dolfinx::fem::CoordinateElement::compute_jacobian_determinant(J, detJ_scratch));

    // Get basis function views
    cmdspan4_t full_basis(basis.data(), e_shape);
    cmdspan2_t phi = stdex::submdspan(full_basis, 0, stdex::full_extent, stdex::full_extent, 0);

    // Main loop
    for (std::size_t q = 0; q < weights.size(); q++)
    {
      double w0 = weights[q] * detJ;
      for (int i = 0; i < ndofs_cell; i++)
        b[i] += w0 * phi(q, i);
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

  namespace stdex = std::experimental;
  using cmdspan4_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 4>>;
  using mdspan4_t = stdex::mdspan<double, stdex::dextents<std::size_t, 4>>;
  using mdspan2_t = stdex::mdspan<double, stdex::dextents<std::size_t, 2>>;
  using cmdspan2_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
  using mdspan3_t = stdex::mdspan<double, stdex::dextents<std::size_t, 3>>;
  using cmdspan3_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 3>>;

  auto mesh = V->mesh();
  assert(mesh);
  // Get mesh info
  const int gdim = mesh->geometry().dim(); // geometrical dimension
  const int tdim = mesh->topology().dim(); // topological dimension
  const int fdim = tdim - 1;               // topological dimension of facet

  // Create coordinate elements for cell
  const dolfinx::fem::CoordinateElement cmap = mesh->geometry().cmap();
  const int num_coordinate_dofs = cmap.dim();

  // Create quadrature points on reference facet
  const std::vector<xt::xtensor<double, 2>>& q_points = q_rule.points_ref();
  const std::vector<std::vector<double>>& q_weights = q_rule.weights_ref();

  const std::uint32_t num_facets = q_weights.size();

  // Structures needed for basis function tabulation
  // phi and grad(phi) and coordinate element derivative at quadrature points
  std::shared_ptr<const dolfinx::fem::FiniteElement> element = V->element();
  int bs = element->block_size();
  std::uint32_t num_local_dofs = element->space_dimension() / bs;
  std::vector<std::pair<std::vector<double>, std::array<std::size_t, 4>>> basis_values;
  std::vector<std::pair<std::vector<double>, std::array<std::size_t, 4>>> coordinate_basis_values;

  // Tabulate basis functions (for test/trial function) and coordinate element at
  // quadrature points
  for (int i = 0; i < num_facets; ++i)
  {
    const xt::xtensor<double, 2>& q_facet = q_points[i];
    const int num_quadrature_points = q_facet.shape(0);

    const std::array<std::size_t, 4> e_shape
        = element->basix_element().tabulate_shape(1, num_quadrature_points);
    assert(e_shape.back() == 1);
    basis_values.push_back(
        {std::vector<double>(std::reduce(e_shape.begin(), e_shape.end(), 1, std::multiplies())),
         e_shape});

    // Tabulate at quadrature points on facet
    std::vector<double>& basis = basis_values.back().first;
    element->tabulate(basis, std::span(q_facet.data(), q_facet.size()), q_facet.shape(), 1);

    // Tabulate coordinate element of reference cell
    std::array<std::size_t, 4> tab_shape = cmap.tabulate_shape(1, q_facet.shape(0));
    coordinate_basis_values.push_back(
        {std::vector<double>(std::reduce(tab_shape.begin(), tab_shape.end(), 1, std::multiplies())),
         tab_shape});
    std::vector<double>& cbasis = coordinate_basis_values.back().first;
    cmap.tabulate(1, std::span(q_facet.data(), q_facet.size()), q_facet.shape(), cbasis);
  }

  // As reference facet and reference cell are affine, we do not need to compute this per
  // quadrature point
  basix::cell::type basix_cell
      = dolfinx::mesh::cell_type_to_basix_type(mesh->topology().cell_type());
  std::pair<std::vector<double>, std::array<std::size_t, 3>> jacobians
      = basix::cell::facet_jacobians(basix_cell);
  auto ref_jac = jacobians.first;
  auto jac_shape = jacobians.second;

  // Get facet normals on reference cell
  std::pair<std::vector<double>, std::array<std::size_t, 2>> normals
      = basix::cell::facet_outward_normals(basix_cell);
  auto facet_normals = normals.first;
  auto normal_shape = normals.second;

  const bool is_affine = cmap.is_affine();
  const std::array<std::size_t, 2> cd_shape = {num_coordinate_dofs, 3};
  // Define kernels
  // v*ds, v TestFunction
  // =====================================================================================
  kernel_fn<T> rhs_surface
      = [=](T* b, const T* c, const T* w, const double* coordinate_dofs,
            const int* entity_local_index, const std::uint8_t* quadrature_permutation)
  {
    std::size_t facet_index = size_t(*entity_local_index);

    // Get basis values
    auto [basis, shape] = basis_values[facet_index];
    auto [cbasis, cshape] = coordinate_basis_values[facet_index];

    // Reshape coordinate dofs
    cmdspan2_t coords(coordinate_dofs, cd_shape);
    auto c_view = stdex::submdspan(coords, stdex::full_extent, std::pair{0, gdim});

    //  FIXME: Assumed constant, i.e. only works for simplices
    assert(is_affine);

    // Compute Jacobian and determinant on facet
    std::vector<double> Jb(gdim * tdim);
    std::vector<double> Kb(tdim * gdim);
    mdspan2_t J(Jb.data(), gdim, tdim);
    mdspan2_t K(Kb.data(), tdim, gdim);
    std::vector<double> detJ_scratch(2 * gdim * tdim);
    cmdspan4_t phi_c_full(cbasis.data(), cshape);
    auto dphi_c_0 = stdex::submdspan(phi_c_full, std::pair(1, tdim + 1), 0, stdex::full_extent, 0);
    dolfinx::fem::CoordinateElement::compute_jacobian(dphi_c_0, c_view, J);
    dolfinx::fem::CoordinateElement::compute_jacobian_inverse(J, K);

    // Extract reference Jacobian at facet
    cmdspan3_t reference_jacobians(ref_jac.data(), jac_shape);
    auto J_f = stdex::submdspan(reference_jacobians, facet_index, stdex::full_extent,
                                stdex::full_extent);
    std::vector<double> J_totb(J.extent(0) * J_f.extent(1));
    mdspan2_t J_tot(J_totb.data(), J.extent(0), J_f.extent(1));
    dolfinx::math::dot(J, J_f, J_tot);

    const double detJ = std::fabs(
        dolfinx::fem::CoordinateElement::compute_jacobian_determinant(J_tot, detJ_scratch));

    // Get number of dofs per cell
    const std::vector<double>& weights = q_weights[facet_index];

    // Extract basis values of test/trial function
    cmdspan4_t phi_full(basis.data(), shape);
    auto phi_f = stdex::submdspan(phi_full, 0, stdex::full_extent, stdex::full_extent, 0);

    // Loop over quadrature points
    for (std::size_t q = 0; q < weights.size(); q++)
    {
      // Scale at each quadrature point
      const double w0 = weights[q] * detJ;

      for (int i = 0; i < num_local_dofs; i++)
      {
        // Compute a weighted phi_i(p_q),  i.e. phi_i(p_q) det(J) w_q and insert
        b[i] += w0 * phi_f(q, i);
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
