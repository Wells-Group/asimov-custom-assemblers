
// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINx_CUAS
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "QuadratureRule.hpp"
#include "kernels.hpp"
#include "utils.hpp"
#include <dolfinx/common/math.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <xtensor/xadapt.hpp>

namespace dolfinx_cuas
{

kernel_fn generate_surface_kernel(std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
                                  dolfinx_cuas::Kernel type,
                                  dolfinx_cuas::QuadratureRule& quadrature_rule)
{

  auto mesh = V->mesh();

  // Get mesh info
  const int gdim = mesh->geometry().dim(); // geometrical dimension
  const int tdim = mesh->topology().dim(); // topological dimension
  const int fdim = tdim - 1;               // topological dimension of facet

  // Create coordinate elements for cell
  const basix::FiniteElement basix_element = mesh_to_basix_element(mesh, tdim);
  const int num_coordinate_dofs = basix_element.dim();

  // Create quadrature points on reference facet
  const std::vector<xt::xarray<double>>& q_points = quadrature_rule.points_ref();
  const std::vector<std::vector<double>>& q_weights = quadrature_rule.weights_ref();

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
    const xt::xarray<double>& q_facet = q_points[i];
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
    auto c_tab = basix_element.tabulate(1, q_facet);
    xt::xtensor<double, 3> dphi_ci
        = xt::view(c_tab, xt::range(1, tdim + 1), xt::all(), xt::all(), 0);
    dphi_c.push_back(dphi_ci);
  }

  // As reference facet and reference cell are affine, we do not need to compute this per
  // quadrature point
  auto ref_jacobians = basix::cell::facet_jacobians(basix_element.cell_type());

  // Get facet normals on reference cell
  auto facet_normals = basix::cell::facet_outward_normals(basix_element.cell_type());

  // Define kernels
  kernel_fn mass = [=](double* A, const double* c, const double* w, const double* coordinate_dofs,
                       const int* entity_local_index, const std::uint8_t* quadrature_permutation)
  {
    std::size_t facet_index = size_t(*entity_local_index);

    // Reshape coordinate dofs to two dimensional array
    // NOTE: DOLFINx has 3D input coordinate dofs
    std::array<std::size_t, 2> shape = {num_coordinate_dofs, 3};
    const xt::xtensor<double, 2> coord
        = xt::adapt(coordinate_dofs, num_coordinate_dofs * 3, xt::no_ownership(), shape);
    auto c_view = xt::view(coord, xt::all(), xt::range(0, gdim));
    const xt::xtensor<double, 3>& dphi_fc = dphi_c[facet_index];
    const xt::xtensor<double, 2>& dphi0_c
        = xt::view(dphi_fc, xt::all(), 0,
                   xt::all()); // FIXME: Assumed constant, i.e. only works for simplices
    // Compute Jacobian and determinant at each quadrature point
    xt::xtensor<double, 2> J = xt::zeros<double>({gdim, tdim});
    dolfinx_cuas::math::compute_jacobian(dphi0_c, c_view, J);
    const xt::xtensor<double, 2>& J_f = xt::view(ref_jacobians, facet_index, xt::all(), xt::all());

    xt::xtensor<double, 2> J_tot = xt::zeros<double>({J.shape(0), J_f.shape(1)});
    dolfinx::math::dot(J, J_f, J_tot);

    const double detJ = std::fabs(dolfinx_cuas::math::compute_determinant(J_tot));
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
        for (int j = 0; j < num_local_dofs; j++)
        {
          // Compute phi_j(p_q) phi_i(p_q) det(J) w_q (block invariant)
          const double integrand = w1 * phi_f(q, j);

          // Insert over block size in matrix
          for (int k = 0; k < bs; k++)
            A[(k + i * bs) * (num_local_dofs * bs) + k + j * bs] += integrand;
        }
      }
    }
  };

  kernel_fn mass_nonaffine
      = [=](double* A, const double* c, const double* w, const double* coordinate_dofs,
            const int* entity_local_index, const std::uint8_t* quadrature_permutation)
  {
    std::size_t facet_index = size_t(*entity_local_index);

    // Reshape coordinate dofs to two dimensional array
    // NOTE: DOLFINx has 3D input coordinate dofs
    std::array<std::size_t, 2> shape = {num_coordinate_dofs, 3};
    const xt::xtensor<double, 2>& coord
        = xt::adapt(coordinate_dofs, num_coordinate_dofs * 3, xt::no_ownership(), shape);
    auto c_view = xt::view(coord, xt::all(), xt::range(0, gdim));

    // Compute Jacobian and determinant at each quadrature point
    xt::xtensor<double, 2> J = xt::zeros<double>({gdim, tdim});
    const xt::xtensor<double, 2>& J_f = xt::view(ref_jacobians, facet_index, xt::all(), xt::all());

    const std::vector<double>& weights = q_weights[facet_index];
    const xt::xtensor<double, 2>& phi_f = phi[facet_index];
    const xt::xtensor<double, 3>& dphi_fc = dphi_c[facet_index];
    xt::xtensor<double, 2> J_tot = xt::zeros<double>({J.shape(0), J_f.shape(1)});

    // Loop over quadrature points
    for (std::size_t q = 0; q < weights.size(); q++)
    {

      // Extract the first derivative of the coordinate element (cell) of degrees of freedom
      // on the facet
      const xt::xtensor<double, 2>& dphi_c_q = xt::view(dphi_fc, xt::all(), q, xt::all());
      J.fill(0);
      dolfinx_cuas::math::compute_jacobian(dphi_c_q, c_view, J);

      // Compute det(J_C J_f) as it is the mapping to the reference facet
      J_tot.fill(0);

      dolfinx::math::dot(J, J_f, J_tot);
      double detJ = std::fabs(dolfinx_cuas::math::compute_determinant(J_tot));

      // Scale at each quadrature point
      const double w0 = weights[q] * detJ;

      for (int i = 0; i < num_local_dofs; i++)
      {
        // Compute a weighted phi_i(p_q),  i.e. phi_i(p_q) det(J) w_q
        double w1 = w0 * phi_f(q, i);
        for (int j = 0; j < num_local_dofs; j++)
        {
          // Compute phi_j(p_q) phi_i(p_q) det(J) w_q (block invariant)
          const double integrand = w1 * phi_f(q, j);

          // Insert over block size in matrix
          for (int k = 0; k < bs; k++)
            A[(k + i * bs) * (num_local_dofs * bs) + k + j * bs] += integrand;
        }
      }
    }
  };
  // FIXME: Template over gdim and tdim?
  kernel_fn stiffness
      = [=](double* A, const double* c, const double* w, const double* coordinate_dofs,
            const int* entity_local_index, const std::uint8_t* quadrature_permutation)
  {
    std::size_t facet_index = size_t(*entity_local_index);
    // Reshape coordinate dofs to two dimensional array
    // NOTE: DOLFINx has 3D input coordinate dofs
    std::array<std::size_t, 2> shape = {num_coordinate_dofs, 3};
    auto coord = xt::adapt(coordinate_dofs, num_coordinate_dofs * 3, xt::no_ownership(), shape);
    auto c_view = xt::view(coord, xt::all(), xt::range(0, gdim));

    // Extract the first derivative of the coordinate element (cell) of degrees of freedom on
    // the facet
    const xt::xtensor<double, 3>& dphi_fc = dphi_c[facet_index];
    auto dphi0_c = xt::view(dphi_fc, xt::all(), 0,
                            xt::all()); // FIXME: Assumed constant, i.e. only works for simplices

    // Compute Jacobian and inverse of cell mapping at each quadrature point
    xt::xtensor<double, 2> J = xt::zeros<double>({gdim, tdim});
    xt::xtensor<double, 2> K = xt::zeros<double>({tdim, gdim});
    dolfinx_cuas::math::compute_jacobian(dphi0_c, c_view, J);
    dolfinx_cuas::math::compute_inv(J, K);

    // Compute det(J_C J_f) as it is the mapping to the reference facet
    auto J_f = xt::view(ref_jacobians, facet_index, xt::all(), xt::all());
    xt::xtensor<double, 2> J_tot = xt::zeros<double>({J.shape(0), J_f.shape(1)});
    dolfinx::math::dot(J, J_f, J_tot);
    double detJ = std::fabs(dolfinx_cuas::math::compute_determinant(J_tot));

    // Temporary variable for grad(phi) on physical cell
    xt::xtensor<double, 2> dphi_phys({gdim, num_local_dofs});

    const std::vector<double>& weights = q_weights[facet_index];
    const xt::xtensor<double, 2>& phi_f = phi[facet_index];
    const xt::xtensor<double, 3>& dphi_f = dphi[facet_index];

    // Loop over quadrature points.
    for (std::size_t q = 0; q < weights.size(); q++)
    {
      // Scale for integral. NOTE: for non-simplices detJ is detJ[q]
      const double w0 = weights[q] * detJ;

      // Precompute J^-T * dphi
      std::fill(dphi_phys.begin(), dphi_phys.end(), 0);
      for (int i = 0; i < num_local_dofs; i++)
        for (int k = 0; k < tdim; k++)
          for (int j = 0; j < gdim; j++)
            dphi_phys(j, i) += K(k, j) * dphi_f(k, q, i);

      for (int i = 0; i < num_local_dofs; i++)
        for (int j = 0; j < num_local_dofs; j++)
        {
          // Compute dphi_i/dx_k dphi_j/dx_k (block invariant)
          double block_invariant_contr = 0;
          for (int k = 0; k < gdim; k++)
            block_invariant_contr += dphi_phys(k, i) * dphi_phys(k, j);
          block_invariant_contr *= w0;
          // Insert into local matrix (repeated over block size)
          for (int k = 0; k < bs; k++)
            A[(k + i * bs) * (num_local_dofs * bs) + k + j * bs] += block_invariant_contr;
        }
    }
  };

  kernel_fn sym_grad
      = [=](double* A, const double* c, const double* w, const double* coordinate_dofs,
            const int* entity_local_index, const std::uint8_t* quadrature_permutation)
  {
    assert(bs == tdim);
    std::size_t facet_index = size_t(*entity_local_index);
    // Reshape coordinate dofs to two dimensional array
    // NOTE: DOLFINx assumes 3D coordinate dofs input
    std::array<std::size_t, 2> shape = {num_coordinate_dofs, 3};
    auto coord = xt::adapt(coordinate_dofs, num_coordinate_dofs * 3, xt::no_ownership(), shape);
    auto c_view = xt::view(coord, xt::all(), xt::range(0, gdim));

    // Extract the first derivative of the coordinate element(cell) of degrees of freedom on
    // the facet
    const xt::xtensor<double, 3>& dphi_cf = dphi_c[facet_index];
    auto dphi0_c
        = xt::view(dphi_cf, xt::all(), 0,
                   xt::all()); // dphi_cf FIXME: Assumed constant, i.e. only works for simplices

    // Compute Jacobian and inverse of cell mapping at each quadrature point
    xt::xtensor<double, 2> J = xt::zeros<double>({gdim, tdim});
    xt::xtensor<double, 2> K = xt::zeros<double>({tdim, gdim});
    dolfinx_cuas::math::compute_jacobian(dphi0_c, c_view, J);
    dolfinx_cuas::math::compute_inv(J, K);

    // Compute det(J_C J_f) as it is the mapping to the reference facet
    xt::xtensor<double, 2> J_f = xt::view(ref_jacobians, facet_index, xt::all(), xt::all());
    xt::xtensor<double, 2> J_tot = xt::zeros<double>({J.shape(0), J_f.shape(1)});
    dolfinx::math::dot(J, J_f, J_tot);
    double detJ = std::fabs(dolfinx_cuas::math::compute_determinant(J_tot));

    // Temporary variable for grad(phi) on physical cell
    xt::xtensor<double, 2> dphi_phys({gdim, num_local_dofs});

    // Get tables for facet
    const std::vector<double>& weights = q_weights[facet_index];
    const xt::xtensor<double, 2>& phi_f = phi[facet_index];
    const xt::xtensor<double, 3>& dphi_f = dphi[facet_index];

    // Loop over quadrature points
    for (std::size_t q = 0; q < weights.size(); q++)
    {
      // Create for integral. NOTE: for non-simplices detJ is detJ[q]
      const double w0 = weights[q] * detJ;

      // Precompute J^-T * dphi
      std::fill(dphi_phys.begin(), dphi_phys.end(), 0);
      for (int i = 0; i < num_local_dofs; i++)
        for (int j = 0; j < gdim; j++)
          for (int k = 0; k < tdim; k++)
            dphi_phys(j, i) += K(k, j) * dphi_f(k, q, i);

      // This corresponds to the term sym(grad(u)):sym(grad(v)) (see
      // https://www.overleaf.com/read/wnvkgjfnhkrx for details)
      for (int i = 0; i < num_local_dofs; i++)
      {
        for (int j = 0; j < num_local_dofs; j++)
        {
          // Compute sum_t dphi^j/dx_t dphi^i/dx_t
          // Component is invarient of block size
          double block_invariant_cont = 0;
          for (int s = 0; s < gdim; s++)
            block_invariant_cont += dphi_phys(s, i) * dphi_phys(s, j);
          block_invariant_cont *= 0.5 * w0;

          for (int k = 0; k < bs; ++k)
          {
            const std::size_t row = (k + i * bs) * (num_local_dofs * bs);
            A[row + j * bs + k] += block_invariant_cont;

            // Add dphi^j/dx_k dphi^i/dx_l
            for (int l = 0; l < bs; ++l)
              A[row + j * bs + l] += 0.5 * w0 * dphi_phys(l, i) * dphi_phys(k, j);
          }
        }
      }
    }
  };
  kernel_fn normal = [=](double* A, const double* c, const double* w, const double* coordinate_dofs,
                         const int* entity_local_index, const std::uint8_t* quadrature_permutation)
  {
    assert(bs == tdim);
    std::size_t facet_index = size_t(*entity_local_index);
    // Reshape coordinate dofs to two dimensional array
    // NOTE: DOLFINx assumes 3D coordinate dofs input
    std::array<std::size_t, 2> shape = {num_coordinate_dofs, 3};
    xt::xtensor<double, 2> coord
        = xt::adapt(coordinate_dofs, num_coordinate_dofs * 3, xt::no_ownership(), shape);
    auto c_view = xt::view(coord, xt::all(), xt::range(0, gdim));

    // Extract the first derivative of the coordinate element(cell) of degrees of freedom on
    // the facet
    const xt::xtensor<double, 3> dphi_cf = dphi_c[facet_index];
    auto dphi0_c = xt::view(dphi_cf, xt::all(), 0,
                            xt::all()); // FIXME: Assumed constant, i.e. only works for simplices

    // Compute Jacobian and inverse of cell mapping at each quadrature point
    xt::xtensor<double, 2> J = xt::zeros<double>({gdim, tdim});
    xt::xtensor<double, 2> K = xt::zeros<double>({tdim, gdim});
    dolfinx_cuas::math::compute_jacobian(dphi0_c, c_view, J);
    dolfinx_cuas::math::compute_inv(J, K);

    // Compute normal of physical facet using a normalized covariant Piola transform
    // n_phys = J^{-T} n_ref / ||J^{-T} n_ref||
    // See for instance DOI: 10.1137/08073901X
    xt::xarray<double> n_phys = xt::zeros<double>({gdim});
    auto facet_normal = xt::row(facet_normals, facet_index);
    for (std::size_t i = 0; i < gdim; i++)
      for (std::size_t j = 0; j < tdim; j++)
        n_phys[i] += K(j, i) * facet_normal[j];
    double n_norm = 0;
    for (std::size_t i = 0; i < gdim; i++)
      n_norm += n_phys[i] * n_phys[i];
    n_phys /= std::sqrt(n_norm);

    // Compute det(J_C J_f) as it is the mapping to the reference facet
    xt::xtensor<double, 2> J_f = xt::view(ref_jacobians, facet_index, xt::all(), xt::all());
    xt::xtensor<double, 2> J_tot = xt::zeros<double>({J.shape(0), J_f.shape(1)});
    dolfinx::math::dot(J, J_f, J_tot);
    double detJ = std::fabs(dolfinx_cuas::math::compute_determinant(J_tot));

    // Get tables for facet
    const std::vector<double>& weights = q_weights[facet_index];
    const xt::xtensor<double, 2>& phi_f = phi[facet_index];
    const xt::xtensor<double, 3>& dphi_f = dphi[facet_index];

    // Temporary variable for grad(phi) on physical cell
    xt::xtensor<double, 2> dphi_phys({gdim, num_local_dofs});

    // Loop over quadrature points
    for (std::size_t q = 0; q < weights.size(); q++)
    {
      // Create for integral. NOTE: for non-simplices detJ is detJ[q]
      const double w0 = weights[q] * detJ;

      // Precompute J^-T * dphi
      std::fill(dphi_phys.begin(), dphi_phys.end(), 0);
      for (int i = 0; i < num_local_dofs; i++)
        for (int j = 0; j < gdim; j++)
          for (int k = 0; k < tdim; k++)
            dphi_phys(j, i) += K(k, j) * dphi_f(k, q, i);
      for (int j = 0; j < num_local_dofs; j++)
      {
        for (int i = 0; i < num_local_dofs; i++)
        {
          // Compute sum_s dphi_phys^i/dx_s n_s \phi^j
          // Component is invarient of block size
          double block_invariant_cont = 0;
          for (int s = 0; s < gdim; s++)
            block_invariant_cont += dphi_phys(s, i) * n_phys(s) * phi_f(q, j);
          block_invariant_cont *= w0;

          for (int l = 0; l < bs; ++l)
          {
            const std::size_t row = (l + j * bs) * (num_local_dofs * bs);
            A[row + i * bs + l] += block_invariant_cont;

            // Add dphi^j/dx_k dphi^i/dx_l
            for (int b = 0; b < bs; ++b)
              A[row + i * bs + b] += w0 * dphi_phys(l, i) * n_phys(b) * phi_f(q, j);
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
  case dolfinx_cuas::Kernel::MassNonAffine:
    return mass_nonaffine;
  case dolfinx_cuas::Kernel::Normal:
    return normal;
  default:
    throw std::runtime_error("Unrecognized kernel");
  }
}

} // namespace dolfinx_cuas
