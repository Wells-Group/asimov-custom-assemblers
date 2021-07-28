// Copyright (C) 2021 Jørgen S. Dokken, Igor A. Baratta, Sarah Roggendorf
//
// This file is part of DOLFINx_CUAS
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "math.hpp"
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <string>
#include <xtensor-blas/xlinalg.hpp>

using kernel_fn = std::function<void(double*, const double*, const double*, const double*,
                                     const int*, const std::uint8_t*)>;

namespace dolfinx_cuas
{
enum Kernel
{
  Mass,
  MassTensor,
  MassNonAffine,
  Stiffness,
  SymGrad,
  TrEps,
  Normal
};
}

namespace
{
/// Create integration kernel for Pth order Lagrange elements
/// @param[in] type The kernel type (Mass or Stiffness)
/// @return The integration kernel
template <int P, int bs>
kernel_fn generate_tet_kernel(dolfinx_cuas::Kernel type)
{
  // Problem specific parameters
  std::string family = "Lagrange";
  std::string cell = "tetrahedron";
  constexpr std::int32_t gdim = 3;
  constexpr std::int32_t tdim = 3;
  constexpr std::int32_t d = 4;
  constexpr std::int32_t ndofs_cell = (P + 1) * (P + 2) * (P + 3) / 6;

  // NOTE: These assumptions are only fine for simplices
  int quad_degree = 0;
  if (type == dolfinx_cuas::Kernel::Stiffness)
    quad_degree = (P - 1) + (P - 1);
  else if (type == dolfinx_cuas::Kernel::Mass or type == dolfinx_cuas::Kernel::MassTensor)
    quad_degree = 2 * P;
  else if (type == dolfinx_cuas::Kernel::TrEps)
    quad_degree = (P - 1) + (P - 1);
  else if (type == dolfinx_cuas::Kernel::SymGrad)
    quad_degree = (P - 1) + (P - 1);

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

  // Stiffness Matrix using quadrature formulation
  // =====================================================================================

  xt::xtensor<double, 3> _dphi({dphi.shape(1), dphi.shape(2), dphi.shape(0)});
  for (std::int32_t k = 0; k < tdim; k++)
    for (std::size_t q = 0; q < weights.size(); q++)
      for (std::int32_t i = 0; i < ndofs_cell; i++)
        _dphi(q, i, k) = dphi(k, q, i);

  std::array<std::size_t, 2> shape = {d, gdim};
  std::array<std::size_t, 2> shape_d = {ndofs_cell, gdim};
  kernel_fn stiffness
      = [=](double* A, const double* c, const double* w, const double* coordinate_dofs,
            const int* entity_local_index, const std::uint8_t* quadrature_permutation)
  {
    // Get geometrical data
    xt::xtensor<double, 2> J = xt::zeros<double>({gdim, tdim});
    xt::xtensor<double, 2> K = xt::zeros<double>({tdim, gdim});
    xt::xtensor<double, 2> coord = xt::adapt(coordinate_dofs, gdim * d, xt::no_ownership(), shape);

    // Compute Jacobian, its inverse and the determinant
    dolfinx_cuas::math::compute_jacobian(dphi0_c, coord, J);
    dolfinx_cuas::math::compute_inv(J, K);
    const double detJ = std::fabs(dolfinx_cuas::math::compute_determinant(J));

    xt::xtensor<double, 2> dphi_kernel(shape_d);
    for (std::size_t q = 0; q < weights.size(); q++)
    {
      const double w0 = weights[q] * detJ;
      // precompute J^-T * dphi in temporary array d
      std::fill(dphi_kernel.begin(), dphi_kernel.end(), 0);
      for (int i = 0; i < ndofs_cell; i++)
        for (int j = 0; j < gdim; j++)
          for (int k = 0; k < tdim; k++)
            dphi_kernel(i, j) += K(k, j) * _dphi(q, i, k);

      // Special handling of scalar space
      if constexpr (bs == 1)
      {
        // Assemble into local matrix
        for (int i = 0; i < ndofs_cell; i++)
          for (int j = 0; j < ndofs_cell; j++)
            for (int k = 0; k < gdim; k++)
              A[i * ndofs_cell + j] += w0 * dphi_kernel(i, k) * dphi_kernel(j, k);
      }
      else
      {
        // Assemble into local matrix
        for (int i = 0; i < ndofs_cell; i++)
        {
          for (int j = 0; j < ndofs_cell; j++)
          {
            // Compute block invariant contribution
            double block_invariant = 0;
            for (int k = 0; k < gdim; k++)
              block_invariant += dphi_kernel(i, k) * dphi_kernel(j, k);
            block_invariant *= w0;

            // Insert into matrix
            for (int k = 0; k < bs; k++)
              A[(k + i * bs) * (ndofs_cell * bs) + j * bs + k] += block_invariant;
          }
        }
      }
    }
  };

  // Mass Matrix using quadrature formulation
  // =====================================================================================
  kernel_fn mass = [=](double* A, const double* c, const double* w, const double* coordinate_dofs,
                       const int* entity_local_index, const std::uint8_t* quadrature_permutation)
  {
    // Get geometrical data
    xt::xtensor<double, 2> J = xt::zeros<double>({gdim, tdim});
    std::array<std::size_t, 2> shape = {d, gdim};
    xt::xtensor<double, 2> coord = xt::adapt(coordinate_dofs, gdim * d, xt::no_ownership(), shape);

    // Compute Jacobian, its inverse and the determinant
    dolfinx_cuas::math::compute_jacobian(dphi0_c, coord, J);
    double detJ = std::fabs(dolfinx_cuas::math::compute_determinant(J));

    // Main loop
    for (std::size_t q = 0; q < weights.size(); q++)
    {
      double w0 = weights[q] * detJ;
      for (int i = 0; i < ndofs_cell; i++)
      {
        double w1 = w0 * phi.unchecked(q, i);
        for (int j = 0; j < ndofs_cell; j++)
          A[i * ndofs_cell + j] += w1 * phi.unchecked(q, j);
      }
    }
  };

  // Mass Matrix using tensor contraction formulation
  // =====================================================================================

  // Pre-compute local matrix for reference element
  xt::xtensor<double, 2> A0 = xt::zeros<double>({ndofs_cell, ndofs_cell});
  for (std::size_t q = 0; q < weights.size(); q++)
    for (int i = 0; i < ndofs_cell; i++)
      for (int j = 0; j < ndofs_cell; j++)
        A0(i, j) += weights[q] * phi(q, i) * phi(q, j);

  kernel_fn masstensor
      = [=](double* A, const double* c, const double* w, const double* coordinate_dofs,
            const int* entity_local_index, const std::uint8_t* quadrature_permutation)
  {
    // Get geometrical data
    xt::xtensor<double, 2> J = xt::zeros<double>({gdim, tdim});
    std::array<std::size_t, 2> shape = {d, gdim};
    xt::xtensor<double, 2> coord = xt::adapt(coordinate_dofs, gdim * d, xt::no_ownership(), shape);

    // Compute Jacobian, its inverse and the determinant
    dolfinx_cuas::math::compute_jacobian(dphi0_c, coord, J);
    double detJ = std::fabs(dolfinx_cuas::math::compute_determinant(J));

    for (int i = 0; i < ndofs_cell; i++)
      for (int j = 0; j < ndofs_cell; j++)
        A[i * ndofs_cell + j] += detJ * A0.unchecked(i, j);
  };

  // Tr(eps(u))I:eps(v) dx
  //========================================================================================
  kernel_fn tr_eps = [=](double* A, const double* c, const double* w, const double* coordinate_dofs,
                         const int* entity_local_index, const std::uint8_t* quadrature_permutation)
  {
    assert(bs == 3);
    // Get geometrical data
    xt::xtensor<double, 2> J = xt::zeros<double>({gdim, tdim});
    xt::xtensor<double, 2> K = xt::zeros<double>({tdim, gdim});
    std::array<std::size_t, 2> shape = {d, gdim};
    xt::xtensor<double, 2> coord = xt::adapt(coordinate_dofs, gdim * d, xt::no_ownership(), shape);

    // Compute Jacobian, its inverse and the determinant
    dolfinx_cuas::math::compute_jacobian(dphi0_c, coord, J);
    dolfinx_cuas::math::compute_inv(J, K);
    double detJ = std::fabs(dolfinx_cuas::math::compute_determinant(J));

    // Temporary variable for grad(phi) on physical cell
    xt::xtensor<double, 2> dphi_phys({bs, ndofs_cell});

    // Main loop
    for (std::size_t q = 0; q < weights.size(); q++)
    {
      double w0 = weights[q] * detJ;

      // Precompute J^-T * dphi
      std::fill(dphi_phys.begin(), dphi_phys.end(), 0);
      for (int i = 0; i < ndofs_cell; i++)
        for (int j = 0; j < bs; j++)
          for (int k = 0; k < tdim; k++)
            dphi_phys(j, i) += K(k, j) * _dphi(q, i, k);

      // Add contributions to local matrix
      for (int i = 0; i < ndofs_cell; i++)
      {
        for (int j = 0; j < ndofs_cell; j++)
        {
          for (int k = 0; k < bs; ++k)
          {
            const size_t row = (k + i * bs) * (ndofs_cell * bs);

            for (int l = 0; l < bs; ++l)
            {
              // Add term from tr(eps(u))I: eps(v)
              A[row + j * bs + l] += dphi_phys(k, i) * dphi_phys(l, j) * w0;
            }
          }
        }
      }
    }
  };

  // sym(grad(eps(u))):eps(v) dx
  //========================================================================================
  kernel_fn sym_grad_eps
      = [=](double* A, const double* c, const double* w, const double* coordinate_dofs,
            const int* entity_local_index, const std::uint8_t* quadrature_permutation)
  {
    assert(bs == 3);
    // Get geometrical data
    xt::xtensor<double, 2> J = xt::zeros<double>({gdim, tdim});
    xt::xtensor<double, 2> K = xt::zeros<double>({tdim, gdim});
    std::array<std::size_t, 2> shape = {d, gdim};
    xt::xtensor<double, 2> coord = xt::adapt(coordinate_dofs, gdim * d, xt::no_ownership(), shape);

    // Compute Jacobian, its inverse and the determinant
    dolfinx_cuas::math::compute_jacobian(dphi0_c, coord, J);
    dolfinx_cuas::math::compute_inv(J, K);
    double detJ = std::fabs(dolfinx_cuas::math::compute_determinant(J));

    // Temporary variable for grad(phi) on physical cell
    xt::xtensor<double, 2> dphi_phys({ndofs_cell, bs});

    // Main loop
    for (std::size_t q = 0; q < weights.size(); q++)
    {
      const double w0 = weights[q] * detJ;
      // Precompute J^-T * dphi
      std::fill(dphi_phys.begin(), dphi_phys.end(), 0);
      for (int i = 0; i < ndofs_cell; i++)
      {
        for (int j = 0; j < bs; j++)
        {
          double acc = 0;
          for (int k = 0; k < tdim; k++)
            acc += K(k, j) * _dphi(q, i, k);
          dphi_phys(i, j) += acc;
        }
      }

      // Add contributions to local matrix
      for (int i = 0; i < ndofs_cell; i++)
      {
        for (int j = 0; j < ndofs_cell; j++)
        {
          // Compute block invariant term from sigma(u):eps(v)
          double block_invariant = 0;
          for (int s = 0; s < bs; s++)
            block_invariant += dphi_phys(i, s) * dphi_phys(j, s);
          block_invariant *= w0;

          for (int k = 0; k < bs; ++k)
          {
            const size_t row = (k + i * bs) * (ndofs_cell * bs);

            // Add block invariant term from sigma(u):eps(v)
            A[row + j * bs + k] += block_invariant;

            for (int l = 0; l < bs; ++l)
              A[row + j * bs + l] += dphi_phys(i, l) * dphi_phys(j, k) * w0;
          }
        }
      }
    }
  };

  switch (type)
  {
  case dolfinx_cuas::Kernel::Mass:
    return mass;
  case dolfinx_cuas::Kernel::MassTensor:
    return masstensor;
  case dolfinx_cuas::Kernel::Stiffness:
    return stiffness;
  case dolfinx_cuas::Kernel::TrEps:
    return tr_eps;
  case dolfinx_cuas::Kernel::SymGrad:
    return sym_grad_eps;
  default:
    throw std::runtime_error("unrecognized kernel");
  }
}
} // namespace
namespace dolfinx_cuas
{

/// Create integration kernel for Pth order Lagrange elements
/// @param[in] type The kernel type (Mass or Stiffness)
/// @param[in] P Degree of the element
/// @param[in] bs The block size
/// @return The integration kernel
kernel_fn generate_kernel(dolfinx_cuas::Kernel type, int P, int bs)
{
  switch (P)
  {
  case 1:
    switch (bs)
    {
    case 1:
      return generate_tet_kernel<1, 1>(type);
    case 2:
      return generate_tet_kernel<1, 2>(type);
    case 3:
      return generate_tet_kernel<1, 3>(type);
    default:
      throw std::runtime_error("Can only have block size from 1 to 3.");
    }
  case 2:
    switch (bs)
    {
    case 1:
      return generate_tet_kernel<2, 1>(type);
    case 2:
      return generate_tet_kernel<2, 2>(type);
    case 3:
      return generate_tet_kernel<2, 3>(type);
    default:
      throw std::runtime_error("Can only have block size from 1 to 3.");
    }
  case 3:
    switch (bs)
    {
    case 1:
      return generate_tet_kernel<3, 1>(type);
    case 2:
      return generate_tet_kernel<3, 2>(type);
    case 3:
      return generate_tet_kernel<3, 3>(type);
    default:
      throw std::runtime_error("Can only have block size from 1 to 3.");
    }
  case 4:
    switch (bs)
    {
    case 1:
      return generate_tet_kernel<4, 1>(type);
    case 2:
      return generate_tet_kernel<4, 2>(type);
    case 3:
      return generate_tet_kernel<4, 3>(type);
    default:
      throw std::runtime_error("Can only have block size from 1 to 3.");
    }
  case 5:
    switch (bs)
    {
    case 1:
      return generate_tet_kernel<5, 1>(type);
    case 2:
      return generate_tet_kernel<5, 2>(type);
    case 3:
      return generate_tet_kernel<5, 3>(type);
    default:
      throw std::runtime_error("Can only have block size from 1 to 3.");
    }
  default:
    throw std::runtime_error("Custom kernel only supported up to 5th order");
  }
}
} // namespace dolfinx_cuas
