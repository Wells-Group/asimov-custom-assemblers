// Copyright (C) 2021 Sarah Roggendorf
//
// This file is part of DOLFINx_CUAS
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "kernels.hpp"
#include "utils.hpp"
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/FunctionSpace.h>

namespace dolfinx_cuas
{

kernel_fn generate_vector_kernel(dolfinx_cuas::Kernel type, int P)
{
  // Problem specific parameters
  std::string family = "Lagrange";
  std::string cell = "tetrahedron";
  constexpr std::int32_t gdim = 3;
  constexpr std::int32_t tdim = 3;
  constexpr std::int32_t d = 4;
  std::int32_t ndofs_cell = (P + 1) * (P + 2) * (P + 3) / 6;

  // NOTE: These assumptions are only fine for simplices
  int quad_degree = (P + 1);

  auto [points, weight]
      = basix::quadrature::make_quadrature("default", basix::cell::str_to_type(cell), quad_degree);
  std::vector<double> weights(weight);

  // Create Finite element for test and trial functions and tabulate shape functions
  basix::FiniteElement element = basix::create_element(family, cell, P);
  xt::xtensor<double, 4> basis = element.tabulate(1, points);
  xt::xtensor<double, 2> phi = xt::view(basis, 0, xt::all(), xt::all(), 0);
  xt::xtensor<double, 3> dphi = xt::view(basis, xt::range(1, tdim + 1), xt::all(), xt::all(), 0);

  // Get coordinate element from dolfinx
  basix::FiniteElement coordinate_element = basix::create_element("Lagrange", cell, 1);
  xt::xtensor<double, 4> coordinate_basis = coordinate_element.tabulate(1, points);

  xt::xtensor<double, 2> dphi0_c
      = xt::view(coordinate_basis, xt::range(1, tdim + 1), 0, xt::all(), 0);

  assert(ndofs_cell == static_cast<std::int32_t>(phi.shape(1)));

  // v*dx, v TestFunction
  // =====================================================================================
  kernel_fn rhs = [=](double* b, const double* c, const double* w, const double* coordinate_dofs,
                      const int* entity_local_index, const std::uint8_t* quadrature_permutation)
  {
    // Get geometrical data
    xt::xtensor<double, 2> J = xt::zeros<double>({gdim, tdim});
    std::array<std::size_t, 2> shape = {d, gdim};
    // FIXME: These array should be views (when compute_jacobian doesn't use xtensor)
    xt::xtensor<double, 2> coord = xt::adapt(coordinate_dofs, gdim * d, xt::no_ownership(), shape);

    // Compute Jacobian, its inverse and the determinant
    dolfinx_cuas::math::compute_jacobian(dphi0_c, coord, J);
    double detJ = std::fabs(dolfinx_cuas::math::compute_determinant(J));

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

kernel_fn generate_surface_vector_kernel(std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
                                         dolfinx_cuas::Kernel type, int P)
{

  auto mesh = V->mesh();
  int quadrature_degree = P + 1;

  // Get mesh info
  const int gdim = mesh->geometry().dim(); // geometrical dimension
  const int tdim = mesh->topology().dim(); // topological dimension
  const int fdim = tdim - 1;               // topological dimension of facet

  // Create coordinate elements (for facet and cell)
  const basix::FiniteElement surface_element = mesh_to_basix_element(mesh, fdim);
  const basix::FiniteElement basix_element = mesh_to_basix_element(mesh, tdim);
  const int num_coordinate_dofs = basix_element.dim();

  // Create quadrature points on reference facet
  const basix::cell::type basix_facet = surface_element.cell_type();
  auto quadrature_data
      = basix::quadrature::make_quadrature("default", basix_facet, quadrature_degree);
  auto qp_ref_facet = quadrature_data.first;
  auto q_weights = quadrature_data.second;

  // Tabulate coordinate element of reference facet (used to compute Jacobian on
  // facet) and push forward quadrature points
  auto f_tab = surface_element.tabulate(0, qp_ref_facet);
  xt::xtensor<double, 2> phi_f = xt::view(f_tab, 0, xt::all(), xt::all(), 0);

  // Structures required for pushing forward quadrature points
  auto facets
      = basix::cell::topology(basix_element.cell_type())[tdim - 1]; // Topology of basix facets
  const xt::xtensor<double, 2> x
      = basix::cell::geometry(basix_element.cell_type()); // Geometry of basix cell
  const std::uint32_t num_facets = facets.size();
  const std::uint32_t num_quadrature_pts = qp_ref_facet.shape(0);

  // Structures needed for basis function tabulation
  // phi and grad(phi) at quadrature points
  std::shared_ptr<const dolfinx::fem::FiniteElement> element = V->element();
  int bs = element->block_size();
  std::uint32_t num_local_dofs = element->space_dimension() / bs;
  xt::xtensor<double, 3> phi({num_facets, num_quadrature_pts, num_local_dofs});
  xt::xtensor<double, 4> cell_tab({tdim + 1, num_quadrature_pts, num_local_dofs, bs});

  // Structure needed for jacobian of cell basis function
  xt::xtensor<double, 4> dphi_c({num_facets, tdim, num_quadrature_pts, basix_element.dim()});

  for (int i = 0; i < num_facets; ++i)
  {
    // Push quadrature points forward
    auto facet = facets[i];
    auto coords = xt::view(x, xt::keep(facet), xt::all());
    auto q_facet = xt::linalg::dot(phi_f, coords);

    // Tabulate at quadrature points on facet
    auto phi_i = xt::view(phi, i, xt::all(), xt::all());
    // replace with element->tabulate(cell_tab, q_facet, 1); for first order derivatives
    element->tabulate(cell_tab, q_facet, 0);
    phi_i = xt::view(cell_tab, 0, xt::all(), xt::all(), 0);

    // Tabulate coordinate element of reference cell
    auto c_tab = basix_element.tabulate(1, q_facet);
    auto dphi_ci = xt::view(dphi_c, i, xt::all(), xt::all(), xt::all());
    dphi_ci = xt::view(c_tab, xt::range(1, tdim + 1), xt::all(), xt::all(), 0);
  }

  // As reference facet and reference cell are affine, we do not need to compute this per
  // quadrature point
  auto ref_jacobians = basix::cell::facet_jacobians(basix_element.cell_type());

  // Get facet normals on reference cell
  auto facet_normals = basix::cell::facet_outward_normals(basix_element.cell_type());

  // Define kernels
  // v*ds, v TestFunction
  // =====================================================================================
  kernel_fn rhs_surface
      = [dphi_c, phi, gdim, tdim, fdim, q_weights, num_coordinate_dofs,
         ref_jacobians](double* b, const double* c, const double* w, const double* coordinate_dofs,
                        const int* entity_local_index, const std::uint8_t* quadrature_permutation)
  {
    std::size_t facet_index = size_t(*entity_local_index);

    // Reshape coordinate dofs to two dimensional array
    // NOTE: DOLFINx has 3D input coordinate dofs
    std::array<std::size_t, 2> shape = {num_coordinate_dofs, 3};

    // FIXME: These array should be views (when compute_jacobian doesn't use xtensor)
    xt::xtensor<double, 2> coord
        = xt::adapt(coordinate_dofs, num_coordinate_dofs * 3, xt::no_ownership(), shape);

    // Extract the first derivative of the coordinate element (cell) of degrees of freedom on
    // the facet
    xt::xtensor<double, 2> dphi0_c
        = xt::view(dphi_c, facet_index, xt::all(), 0,
                   xt::all()); // FIXME: Assumed constant, i.e. only works for simplices

    // Compute Jacobian and determinant at each quadrature point
    xt::xtensor<double, 2> J = xt::zeros<double>({gdim, tdim});
    dolfinx_cuas::math::compute_jacobian(dphi0_c, coord, J);

    // Compute det(J_C J_f) as it is the mapping to the reference facet
    xt::xtensor<double, 2> J_f = xt::view(ref_jacobians, facet_index, xt::all(), xt::all());
    xt::xtensor<double, 2> J_tot = xt::linalg::dot(J, J_f);
    double detJ = std::fabs(dolfinx_cuas::math::compute_determinant(J_tot));

    // Get number of dofs per cell
    // FIXME: Should be templated
    std::int32_t ndofs_cell = phi.shape(2);

    // Loop over quadrature points
    for (std::size_t q = 0; q < phi.shape(1); q++)
    {
      // Scale at each quadrature point
      const double w0 = q_weights[q] * detJ;

      for (int i = 0; i < ndofs_cell; i++)
      {
        // Compute a weighted phi_i(p_q),  i.e. phi_i(p_q) det(J) w_q
        double w1 = w0 * phi(facet_index, q, i);
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
