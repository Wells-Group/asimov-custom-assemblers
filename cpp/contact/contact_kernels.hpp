// Copyright (C) 2021 Sarah Roggendorf
//
// This file is part of DOLFINx_CUAS
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "../utils.h"
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/FunctionSpace.h>

using kernel_fn = std::function<void(double*, const double*, const double*, const double*,
                                     const int*, const std::uint8_t*)>;

namespace dolfinx_cuas
{
namespace contact
{
enum Kernel
{
  NitscheRigidSurface
};

kernel_fn
generate_rhs_kernel(std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
                    dolfinx_cuas::contact::Kernel type, int quadrature_degree,
                    std::vector<std::shared_ptr<const dolfinx::fem::Function<PetscScalar>>> coeffs)
{

  auto mesh = V->mesh();

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
  std::uint32_t ndofs_cell = element->space_dimension() / bs;
  xt::xtensor<double, 3> phi({num_facets, num_quadrature_pts, ndofs_cell});
  xt::xtensor<double, 4> dphi({num_facets, tdim, num_quadrature_pts, ndofs_cell});
  xt::xtensor<double, 4> cell_tab({tdim + 1, num_quadrature_pts, ndofs_cell, bs});

  // Structure needed for Jacobian of cell basis function
  xt::xtensor<double, 4> dphi_c({num_facets, tdim, num_quadrature_pts, basix_element.dim()});

  // Structures for coefficient data
  int num_coeffs = coeffs.size();
  std::vector<int> offsets(num_coeffs + 1);
  offsets[0] = 0;
  for (int i = 1; i < num_coeffs + 1; i++)
  {
    std::shared_ptr<const dolfinx::fem::FiniteElement> coeff_element
        = coeffs[i - 1]->function_space()->element();
    offsets[i] = offsets[i - 1] + coeff_element->space_dimension() / coeff_element->block_size();
  }
  xt::xtensor<double, 3> phi_coeffs({num_facets, q_weights.size(), offsets[num_coeffs]});
  xt::xtensor<double, 4> dphi_coeffs({num_facets, tdim, q_weights.size(), offsets[num_coeffs]});
  for (int i = 0; i < num_facets; ++i)
  {
    // Push quadrature points forward
    auto facet = facets[i];
    auto coords = xt::view(x, xt::keep(facet), xt::all());
    auto q_facet = xt::linalg::dot(phi_f, coords);

    // Tabulate at quadrature points on facet
    auto phi_i = xt::view(phi, i, xt::all(), xt::all());
    element->tabulate(cell_tab, q_facet, 1);
    phi_i = xt::view(cell_tab, 0, xt::all(), xt::all(), 0);
    auto dphi_i = xt::view(dphi, i, xt::all(), xt::all(), xt::all());
    dphi_i = xt::view(cell_tab, xt::range(1, tdim + 1), xt::all(), xt::all(), 0);

    // Tabulate coordinate element of reference cell
    auto c_tab = basix_element.tabulate(1, q_facet);
    auto dphi_ci = xt::view(dphi_c, i, xt::all(), xt::all(), xt::all());
    dphi_ci = xt::view(c_tab, xt::range(1, tdim + 1), xt::all(), xt::all(), 0);
    // Create Finite elements for coefficient functions and tabulate shape functions
    for (int j = 0; j < num_coeffs; j++)
    {
      std::shared_ptr<const dolfinx::fem::FiniteElement> coeff_element
          = coeffs[j]->function_space()->element();
      xt::xtensor<double, 4> coeff_basis(
          {tdim + 1, q_weights.size(),
           coeff_element->space_dimension() / coeff_element->block_size(), 1});
      coeff_element->tabulate(coeff_basis, q_facet, 1);
      auto phi_ij = xt::view(phi_coeffs, i, xt::all(), xt::range(offsets[j], offsets[j + 1]));
      phi_ij = xt::view(coeff_basis, 0, xt::all(), xt::all(), 0);
      auto dphi_ij
          = xt::view(dphi_coeffs, i, xt::all(), xt::all(), xt::range(offsets[j], offsets[j + 1]));
      dphi_ij = xt::view(coeff_basis, xt::range(1, tdim + 1), xt::all(), xt::all(), 0);
    }
  }

  // As reference facet and reference cell are affine, we do not need to compute this per
  // quadrature point
  auto ref_jacobians = basix::cell::facet_jacobians(basix_element.cell_type());

  // Define kernels
  // v*ds, v TestFunction
  // =====================================================================================
  kernel_fn nitsche_rigid_rhs
      = [dphi_c, phi, dphi, phi_coeffs, dphi_coeffs, offsets, num_coeffs, gdim, tdim, fdim,
         q_weights, num_coordinate_dofs, ref_jacobians,
         bs](double* b, const double* c, const double* w, const double* coordinate_dofs,
             const int* entity_local_index, const std::uint8_t* quadrature_permutation)
  {
    // assumption that the vector function space has block size tdim
    assert(bs == gdim);
    // assumption that u lives in the same space as v
    assert(phi.shape(2) == offsets[1] - offset[0]);
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

    // NOTE: Affine cell assumption
    // Compute Jacobian and determinant at first quadrature point
    xt::xtensor<double, 2> J = xt::zeros<double>({gdim, tdim});
    xt::xtensor<double, 2> K = xt::zeros<double>({tdim, gdim});
    dolfinx_cuas::math::compute_jacobian(dphi0_c, coord, J);
    dolfinx_cuas::math::compute_inv(J, K);

    // Compute det(J_C J_f) as it is the mapping to the reference facet
    xt::xtensor<double, 2> J_f = xt::view(ref_jacobians, facet_index, xt::all(), xt::all());
    xt::xtensor<double, 2> J_tot = xt::linalg::dot(J, J_f);
    double detJ = std::fabs(dolfinx_cuas::math::compute_determinant(J_tot));

    // Get number of dofs per cell
    // FIXME: Should be templated
    std::int32_t ndofs_cell = phi.shape(2);
    // Temporary variable for grad(phi) on physical cell
    xt::xtensor<double, 2> dphi_phys({bs, ndofs_cell});

    // Loop over quadrature points
    for (std::size_t q = 0; q < phi.shape(1); q++)
    {

      xt::xtensor<double, 2> tr = xt::zeros<double>({offsets[1] - offsets[0], gdim});
      // precompute tr(eps(phi_j e_l))
      for (int j = 0; j < offsets[1] - offsets[0]; j++)
      {
        for (int l = 0; l < bs; l++)
        {
          for (int k = 0; k < tdim; k++)
          {
            tr(j, l) += K(k, l) * dphi(facet_index, k, q, j);
          }
        }
      }
      // compute tr(eps(u)) at q
      double tr_u = 0;
      for (int i = 0; i < offsets[1] - offsets[0]; i++)
        for (int j = 0; j < bs; j++)
          tr_u += c[(i + offsets[0]) * bs + j] * tr(i, j);
      // Multiply tr_u by weight
      tr_u *= q_weights[q] * detJ;
      for (int j = 0; j < ndofs_cell; j++)
      {
        // Insert over block size in matrix
        for (int l = 0; l < bs; l++)
          b[j * bs + l] += tr(j, l) * tr_u;
      }
    }
  };
  switch (type)
  {
  case dolfinx_cuas::contact::Kernel::NitscheRigidSurface:
    return nitsche_rigid_rhs;
  default:
    throw std::runtime_error("Unrecognized kernel");
  }
}

} // namespace contact
} // namespace dolfinx_cuas