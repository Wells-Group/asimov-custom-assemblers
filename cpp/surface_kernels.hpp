
// Copyright (C) 2021 Jørgen S. Dokken
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

kernel_fn generate_surface_kernel(std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
                                  dolfinx_cuas::Kernel type, int quadrature_degree)
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
  auto [qp_ref_facet, qw_ref_facet]
      = basix::quadrature::make_quadrature("default", basix_facet, quadrature_degree);

  // Tabulate coordinate elemetn of reference facet (used to compute Jacobian on facet)
  // and push forward quadrature points
  auto f_tab = surface_element.tabulate(1, qp_ref_facet);
  xt::xtensor<double, 2> phi_f = xt::view(f_tab, 0, xt::all(), xt::all(), 0);
  xt::xtensor<double, 3> dphi_f
      = xt::round(xt::view(f_tab, xt::range(1, tdim + 1), xt::all(), xt::all(), 0));

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
  xt::xtensor<double, 4> dphi({num_facets, tdim, num_quadrature_pts, num_local_dofs});
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
    auto dphi_i = xt::view(dphi, i, xt::all(), xt::all(), xt::all());
    element->tabulate(cell_tab, q_facet, 1);
    phi_i = xt::view(cell_tab, 0, xt::all(), xt::all(), 0);
    dphi_i = xt::view(cell_tab, xt::range(1, tdim + 1), xt::all(), xt::all(), 0);

    // Tabulate coordinate element of reference cell
    auto c_tab = basix_element.tabulate(1, q_facet);
    auto dphi_ci = xt::view(dphi_c, i, xt::all(), xt::all(), xt::all());
    dphi_ci = xt::round(xt::view(c_tab, xt::range(1, tdim + 1), xt::all(), xt::all(), 0));
  }

  // FIXME: When are reference jacobians needed?
  // auto ref_jacobians = basix::cell::facet_jacobians(basix_element.cell_type());
  // Define kernels
  auto q_weights = qw_ref_facet;
  kernel_fn mass = [facets, dphi_f, phi, gdim, tdim, fdim, bs, q_weights, num_coordinate_dofs](
                       double* A, const double* c, const double* w, const double* coordinate_dofs,
                       const int* entity_local_index, const std::uint8_t* quadrature_permutation)
  {
    // Reshape coordinate dofs to two dimensional array
    // NOTE: DOLFINx has 3D input coordinate dofs
    std::array<std::size_t, 2> shape = {num_coordinate_dofs, 3};
    xt::xtensor<double, 2> coord
        = xt::adapt(coordinate_dofs, num_coordinate_dofs * 3, xt::no_ownership(), shape);

    // Compute Jacobian and determinant at each quadrature point
    xt::xtensor<double, 2> J = xt::zeros<double>({gdim, fdim});

    // FIXME: Assumed constant, i.e. only works for simplices
    xt::xtensor<double, 2> dphi0_f = xt::view(dphi_f, xt::all(), 0, xt::all());
    dolfinx_cuas::math::compute_jacobian(dphi0_f,
                                         xt::view(coord, xt::keep(facets[*entity_local_index])), J);

    double detJ = std::fabs(dolfinx_cuas::math::compute_determinant(J));

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
        double w1 = w0 * phi(*entity_local_index, q, i);
        for (int j = 0; j < ndofs_cell; j++)
        {
          // Compute phi_j(p_q) phi_i(p_q) det(J) w_q (block invariant)
          const double integrand = w1 * phi(*entity_local_index, q, j);

          // Insert over block size in matrix
          for (int k = 0; k < bs; k++)
            A[(k + i * bs) * (ndofs_cell * bs) + k + j * bs] += integrand;
        }
      }
    }
  };

  // FIXME: Template over gdim and tdim?
  kernel_fn stiffness
      = [facets, dphi_f, dphi, gdim, tdim, fdim, bs, dphi_c, q_weights, num_coordinate_dofs](
            double* A, const double* c, const double* w, const double* coordinate_dofs,
            const int* entity_local_index, const std::uint8_t* quadrature_permutation)
  {
    // Reshape coordinate dofs to two dimensional array
    // NOTE: DOLFINx has 3D input coordinate dofs
    std::array<std::size_t, 2> shape = {num_coordinate_dofs, 3};
    xt::xtensor<double, 2> coord
        = xt::adapt(coordinate_dofs, num_coordinate_dofs * 3, xt::no_ownership(), shape);

    // Compute Jacobian, inverse Jacobian and determinant
    xt::xtensor<double, 2> J_facet = xt::zeros<double>({gdim, fdim});
    xt::xtensor<double, 2> J = xt::zeros<double>({gdim, tdim});
    xt::xtensor<double, 2> K = xt::zeros<double>({tdim, gdim});

    // NOTE: Currently assumed to be constant, thus would only work on affine geometries
    xt::xtensor<double, 2> dphi0_f = xt::view(dphi_f, xt::all(), 0, xt::all());
    dolfinx_cuas::math::compute_jacobian(
        dphi0_f, xt::view(coord, xt::keep(facets[*entity_local_index])), J_facet);
    auto dphi0_c = xt::view(dphi_c, size_t(*entity_local_index), xt::all(), 0, xt::all());

    dolfinx_cuas::math::compute_jacobian(dphi0_c, coord, J);
    dolfinx_cuas::math::compute_inv(J, K);

    double detJ = std::fabs(dolfinx_cuas::math::compute_determinant(J_facet));

    // Get number of dofs per cell.
    // FIXME: This should be templated
    std::int32_t ndofs_cell = dphi.shape(3);

    // Temporary variable for grad(phi) on physical cell
    xt::xtensor<double, 2> dphi_phys({gdim, ndofs_cell});

    // Loop over quadrature points.
    for (std::size_t q = 0; q < dphi.shape(2); q++)
    {
      // Scale for integral. NOTE: for non-simplices detJ is detJ[q]
      const double w0 = q_weights[q] * detJ;

      // Precompute J^-T * dphi
      std::fill(dphi_phys.begin(), dphi_phys.end(), 0);
      for (int i = 0; i < ndofs_cell; i++)
        for (int j = 0; j < gdim; j++)
          for (int k = 0; k < tdim; k++)
            dphi_phys(j, i) += K(k, j) * dphi(*entity_local_index, k, q, i);

      for (int i = 0; i < ndofs_cell; i++)
        for (int j = 0; j < ndofs_cell; j++)
        {
          // Compute dphi_i/dx_k dphi_j/dx_k (block invariant)
          double block_invariant_contr = 0;
          for (int k = 0; k < gdim; k++)
            block_invariant_contr += dphi_phys(k, i) * dphi_phys(k, j);
          block_invariant_contr *= w0;
          // Insert into local matrix (repeated over block size)
          for (int k = 0; k < bs; k++)
            A[(k + i * bs) * (ndofs_cell * bs) + k + j * bs] += block_invariant_contr;
        }
    }
  };

  kernel_fn sym_grad
      = [facets, dphi_f, dphi, gdim, tdim, fdim, bs, dphi_c, q_weights, num_coordinate_dofs](
            double* A, const double* c, const double* w, const double* coordinate_dofs,
            const int* entity_local_index, const std::uint8_t* quadrature_permutation)
  {
    assert(bs == tdim);

    // Reshape coordinate dofs to two dimensional array
    // NOTE: DOLFINx assumes 3D coordinate dofs input
    std::array<std::size_t, 2> shape = {num_coordinate_dofs, 3};
    xt::xtensor<double, 2> coord
        = xt::adapt(coordinate_dofs, num_coordinate_dofs * 3, xt::no_ownership(), shape);

    // Compute Jacobian, inverse Jacobian and determinant
    xt::xtensor<double, 2> J_facet = xt::zeros<double>({gdim, fdim});
    xt::xtensor<double, 2> J = xt::zeros<double>({gdim, tdim});
    xt::xtensor<double, 2> K = xt::zeros<double>({tdim, gdim});
    // NOTE: Currently assumed to be constant, thus would only work on affine geometries
    auto dphi0_f = xt::view(dphi_f, xt::all(), 0, xt::all());
    dolfinx_cuas::math::compute_jacobian(
        dphi0_f, xt::view(coord, xt::keep(facets[*entity_local_index])), J_facet);
    auto dphi0_c = xt::view(dphi_c, size_t(*entity_local_index), xt::all(), 0, xt::all());
    dolfinx_cuas::math::compute_jacobian(dphi0_c, coord, J);
    dolfinx_cuas::math::compute_inv(J, K);
    double detJ = std::fabs(dolfinx_cuas::math::compute_determinant(J_facet));

    // Get number of dofs per cell
    // FIXME: Should be templated
    std::int32_t ndofs_cell = dphi.shape(3);

    // Temporary variable for grad(phi) on physical cell
    xt::xtensor<double, 2> dphi_phys({gdim, ndofs_cell});

    // Loop over quadrature points
    for (std::size_t q = 0; q < dphi.shape(2); q++)
    {
      // Create for integral. NOTE: for non-simplices detJ is detJ[q]
      const double w0 = q_weights[q] * detJ;

      // Precompute J^-T * dphi
      std::fill(dphi_phys.begin(), dphi_phys.end(), 0);
      for (int i = 0; i < ndofs_cell; i++)
        for (int j = 0; j < gdim; j++)
          for (int k = 0; k < tdim; k++)
            dphi_phys(j, i) += K(k, j) * dphi(*entity_local_index, k, q, i);

      // This corresponds to the term sym(grad(u)):sym(grad(v)) (see
      // https://www.overleaf.com/read/wnvkgjfnhkrx for details)
      for (int i = 0; i < ndofs_cell; i++)
      {
        for (int j = 0; j < ndofs_cell; j++)
        {
          // Compute sum_t dphi^j/dx_t dphi^i/dx_t
          // Component is invarient of block size
          double block_invariant_cont = 0;
          for (int t = 0; t < gdim; t++)
            block_invariant_cont += dphi_phys(t, i) * dphi_phys(t, j);
          block_invariant_cont *= 0.5 * w0;

          for (int k = 0; k < bs; ++k)
          {
            const size_t row = (k + i * bs) * (ndofs_cell * bs);
            A[row + j * bs + k] += block_invariant_cont;

            // Add dphi^j/dx_k dphi^i/dx_l
            for (int l = 0; l < bs; ++l)
              A[row + j * bs + l] += 0.5 * w0 * dphi_phys(l, i) * dphi_phys(k, j);
          }
        }
      }
    }
  };
  switch (type)
  {
  case dolfinx_cuas::Kernel::Mass:
    return mass;
  case dolfinx_cuas::Kernel::Stiffness:
    return stiffness;
  case dolfinx_cuas::Kernel::SymGrad:
    return sym_grad;
  default:
    throw std::runtime_error("Unrecognized kernel");
  }
}

} // namespace dolfinx_cuas