// Copyright (C) 2021-2022 Jørgen S. Dokken, Igor A. Baratta, Sarah Roggendorf
//
// This file is part of DOLFINx_CUAS
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "QuadratureRule.hpp"
#include <basix/element-families.h>
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <string>

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
  Normal,
  Rhs
};
template <typename T>
using kernel_fn
    = std::function<void(T*, const T*, const T*, const double*, const int*, const std::uint8_t*)>;

} // namespace dolfinx_cuas

namespace
{
/// Create integration kernel for Pth order Lagrange elements
/// @param[in] type The kernel type (Mass or Stiffness)
/// @param[in] quadrature_rule The quadrature rule
/// @return The integration kernel
template <int P, int bs, typename T>
dolfinx_cuas::kernel_fn<T> generate_tet_kernel(dolfinx_cuas::Kernel type,
                                               dolfinx_cuas::QuadratureRule& quadrature_rule)
{
  namespace stdex = std::experimental;
  using cmdspan4_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 4>>;
  using mdspan4_t = stdex::mdspan<double, stdex::dextents<std::size_t, 4>>;
  using mdspan2_t = stdex::mdspan<double, stdex::dextents<std::size_t, 2>>;
  using cmdspan2_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
  using mdspan3_t = stdex::mdspan<double, stdex::dextents<std::size_t, 3>>;
  using cmdspan3_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 3>>;

  // Problem specific parameters
  basix::element::family family = basix::element::family::P;
  basix::cell::type cell = basix::cell::type::tetrahedron;
  constexpr std::int32_t gdim = 3;
  constexpr std::int32_t tdim = 3;
  constexpr std::int32_t d = 4;
  constexpr std::int32_t ndofs_cell = (P + 1) * (P + 2) * (P + 3) / 6;

  const std::vector<double>& weights = quadrature_rule.weights_ref()[0];
  const xt::xarray<double>& points = quadrature_rule.points_ref()[0];

  // Create Finite element for test and trial functions and tabulate shape functions
  basix::FiniteElement element
      = basix::create_element(family, cell, P, basix::element::lagrange_variant::gll_warped);
  std::array<std::size_t, 4> basis_shape = element.tabulate_shape(1, points.shape(0));
  std::vector<double> basisb(
      std::reduce(basis_shape.begin(), basis_shape.end(), 1, std::multiplies{}));
  std::array<std::size_t, 2> pts_shape = {points.shape(0), points.shape(1)};
  element.tabulate(1, cmdspan2_t(points.data(), pts_shape), mdspan4_t(basisb.data(), basis_shape));

  // Get coordinate element from dolfinx
  basix::FiniteElement coordinate_element = basix::create_element(
      basix::element::family::P, cell, 1, basix::element::lagrange_variant::gll_warped);
  std::array<std::size_t, 4> tab_shape = coordinate_element.tabulate_shape(1, points.shape(0));
  xt::xtensor<double, 4> coordinate_basis(tab_shape);
  coordinate_element.tabulate(1, basix::impl::cmdspan2_t(points.data(), pts_shape),
                              basix::impl::mdspan4_t(coordinate_basis.data(), tab_shape));

  assert(ndofs_cell == static_cast<std::int32_t>(phi.shape(1)));

  // Stiffness Matrix using quadrature formulation
  // =====================================================================================
  cmdspan4_t basis(basisb.data(), basis_shape);
  std::vector<double> dphib(tdim * basis_shape[1] * basis_shape[2]);
  mdspan3_t dphi(dphib.data(), tdim, basis_shape[1], basis_shape[2]);
  for (std::int32_t k = 0; k < tdim; k++)
    for (std::size_t q = 0; q < weights.size(); q++)
      for (std::int32_t i = 0; i < ndofs_cell; i++)
        dphi(q, i, k) = basis(k + 1, q, i, 0);

  std::array<std::size_t, 2> shape = {d, gdim};
  std::array<std::size_t, 2> shape_d = {ndofs_cell, gdim};
  dolfinx_cuas::kernel_fn<T> stiffness
      = [=](T* A, const T* c, const T* w, const double* coordinate_dofs,
            const int* entity_local_index, const std::uint8_t* quadrature_permutation)
  {
    // Create buffers for jacobian (inverse and determinant) computations
    std::vector<double> Jb(gdim * tdim);
    std::vector<double> Kb(tdim * gdim);
    mdspan2_t J(Jb.data(), gdim, tdim);
    mdspan2_t K(Kb.data(), tdim, gdim);
    cmdspan2_t coords(coordinate_dofs, shape);
    std::vector<double> detJ_scratch(2 * gdim * tdim);

    // Compute Jacobian, its inverse and the determinant
    cmdspan4_t dphi_c_full(coordinate_basis.data(), tab_shape);

    auto dphi_c_0 = stdex::submdspan(dphi_c_full, std::pair(1, tdim + 1), 0, stdex::full_extent, 0);
    dolfinx::fem::CoordinateElement::compute_jacobian(dphi_c_0, coords, J);
    dolfinx::fem::CoordinateElement::compute_jacobian_inverse(J, K);
    const double detJ
        = std::fabs(dolfinx::fem::CoordinateElement::compute_jacobian_determinant(J, detJ_scratch));

    // Get derivatives of element basis
    cmdspan3_t dphi(dphib.data(), tdim, basis_shape[1], basis_shape[2]);

    // Create temporary storage for K*dphi_ref
    std::vector<double> dphi_physb(
        std::reduce(shape_d.begin(), shape_d.end(), 1, std::multiplies{}));
    mdspan2_t dphi_phys(dphi_physb.data(), shape_d);

    for (std::size_t q = 0; q < weights.size(); q++)
    {
      const double w0 = weights[q] * detJ;
      // precompute J^-T * dphi in temporary array d
      std::fill(dphi_physb.begin(), dphi_physb.end(), 0);
      for (int i = 0; i < ndofs_cell; i++)
        for (int j = 0; j < gdim; j++)
          for (int k = 0; k < tdim; k++)
            dphi_phys(i, j) += K(k, j) * dphi(q, i, k);

      // Special handling of scalar space
      if constexpr (bs == 1)
      {
        // Assemble into local matrix
        for (int i = 0; i < ndofs_cell; i++)
          for (int j = 0; j < ndofs_cell; j++)
            for (int k = 0; k < gdim; k++)
              A[i * ndofs_cell + j] += w0 * dphi_phys(i, k) * dphi_phys(j, k);
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
              block_invariant += dphi_phys(i, k) * dphi_phys(j, k);
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
  dolfinx_cuas::kernel_fn<T> mass
      = [=](T* A, const T* c, const T* w, const double* coordinate_dofs,
            const int* entity_local_index, const std::uint8_t* quadrature_permutation)
  {
    // Create buffers for jacobian (inverse and determinant) computations
    std::vector<double> Jb(gdim * tdim);
    std::vector<double> Kb(tdim * gdim);
    mdspan2_t J(Jb.data(), gdim, tdim);
    mdspan2_t K(Kb.data(), tdim, gdim);
    cmdspan2_t coords(coordinate_dofs, shape);
    std::vector<double> detJ_scratch(2 * gdim * tdim);

    // Compute Jacobian, its inverse and the determinant
    cmdspan4_t dphi_c_full(coordinate_basis.data(), tab_shape);

    auto dphi_c_0 = stdex::submdspan(dphi_c_full, std::pair(1, tdim + 1), 0, stdex::full_extent, 0);
    dolfinx::fem::CoordinateElement::compute_jacobian(dphi_c_0, coords, J);
    dolfinx::fem::CoordinateElement::compute_jacobian_inverse(J, K);
    const double detJ
        = std::fabs(dolfinx::fem::CoordinateElement::compute_jacobian_determinant(J, detJ_scratch));

    // Get basis function views
    cmdspan4_t full_basis(basisb.data(), basis_shape);
    cmdspan2_t phi = stdex::submdspan(full_basis, 0, stdex::full_extent, stdex::full_extent, 0);

    // Main loop
    for (std::size_t q = 0; q < weights.size(); q++)
    {
      double w0 = weights[q] * detJ;
      for (int i = 0; i < ndofs_cell; i++)
      {
        double w1 = w0 * phi(q, i);
        for (int j = 0; j < ndofs_cell; j++)
        {
          // Special handling of scalar space
          if constexpr (bs == 1)
            A[i * ndofs_cell + j] += w1 * phi(q, j);
          else
            for (int b = 0; b < bs; b++)
              A[(i * bs + b) * (ndofs_cell * bs) + bs * j + b] += w1 * phi(q, j);
        }
      }
    }
  };

  // Mass Matrix using tensor contraction formulation
  // =====================================================================================

  // Pre-compute local matrix for reference element
  cmdspan4_t full_basis(basisb.data(), basis_shape);
  cmdspan2_t phi = stdex::submdspan(full_basis, 0, stdex::full_extent, stdex::full_extent, 0);
  std::vector<double> A0b(ndofs_cell * ndofs_cell);
  mdspan2_t A0(A0b.data(), ndofs_cell, ndofs_cell);
  for (std::size_t q = 0; q < weights.size(); q++)
    for (int i = 0; i < ndofs_cell; i++)
      for (int j = 0; j < ndofs_cell; j++)
        A0(i, j) += weights[q] * phi(q, i) * phi(q, j);

  dolfinx_cuas::kernel_fn<T> masstensor
      = [=](T* A, const T* c, const T* w, const double* coordinate_dofs,
            const int* entity_local_index, const std::uint8_t* quadrature_permutation)
  {
    // Create buffers for jacobian (inverse and determinant) computations
    std::vector<double> Jb(gdim * tdim);
    std::vector<double> Kb(tdim * gdim);
    mdspan2_t J(Jb.data(), gdim, tdim);
    mdspan2_t K(Kb.data(), tdim, gdim);
    cmdspan2_t coords(coordinate_dofs, shape);
    std::vector<double> detJ_scratch(2 * gdim * tdim);

    // Compute Jacobian, its inverse and the determinant
    cmdspan4_t dphi_c_full(coordinate_basis.data(), tab_shape);

    auto dphi_c_0 = stdex::submdspan(dphi_c_full, std::pair(1, tdim + 1), 0, stdex::full_extent, 0);
    dolfinx::fem::CoordinateElement::compute_jacobian(dphi_c_0, coords, J);
    dolfinx::fem::CoordinateElement::compute_jacobian_inverse(J, K);
    const double detJ
        = std::fabs(dolfinx::fem::CoordinateElement::compute_jacobian_determinant(J, detJ_scratch));

    cmdspan2_t A0(A0b.data(), (std::size_t)ndofs_cell, (std::size_t)ndofs_cell);
    for (int i = 0; i < ndofs_cell; i++)
      for (int j = 0; j < ndofs_cell; j++)
        A[i * ndofs_cell + j] += detJ * A0(i, j);
  };

  // Tr(eps(u))I:eps(v) dx
  //========================================================================================
  dolfinx_cuas::kernel_fn<T> tr_eps
      = [=](T* A, const T* c, const T* w, const double* coordinate_dofs,
            const int* entity_local_index, const std::uint8_t* quadrature_permutation)
  {
    assert(bs == 3);

    // Create buffers for jacobian (inverse and determinant) computations
    std::vector<double> Jb(gdim * tdim);
    std::vector<double> Kb(tdim * gdim);
    mdspan2_t J(Jb.data(), gdim, tdim);
    mdspan2_t K(Kb.data(), tdim, gdim);
    cmdspan2_t coords(coordinate_dofs, shape);
    std::vector<double> detJ_scratch(2 * gdim * tdim);

    // Compute Jacobian, its inverse and the determinant
    cmdspan4_t dphi_c_full(coordinate_basis.data(), tab_shape);

    auto dphi_c_0 = stdex::submdspan(dphi_c_full, std::pair(1, tdim + 1), 0, stdex::full_extent, 0);
    dolfinx::fem::CoordinateElement::compute_jacobian(dphi_c_0, coords, J);
    dolfinx::fem::CoordinateElement::compute_jacobian_inverse(J, K);
    const double detJ
        = std::fabs(dolfinx::fem::CoordinateElement::compute_jacobian_determinant(J, detJ_scratch));

    // Get derivatives of element basis
    cmdspan3_t dphi(dphib.data(), tdim, basis_shape[1], basis_shape[2]);

    // Temporary variable for grad(phi) on physical cell
    std::vector<double> dphi_physb(
        std::reduce(shape_d.begin(), shape_d.end(), 1, std::multiplies{}));
    mdspan2_t dphi_phys(dphi_physb.data(), shape_d[1], shape_d[0]);

    // Main loop
    for (std::size_t q = 0; q < weights.size(); q++)
    {
      double w0 = weights[q] * detJ;

      // Precompute J^-T * dphi
      std::fill(dphi_physb.begin(), dphi_physb.end(), 0);
      for (int i = 0; i < ndofs_cell; i++)
        for (int j = 0; j < bs; j++)
          for (int k = 0; k < tdim; k++)
            dphi_phys(j, i) += K(k, j) * dphi(q, i, k);

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
  dolfinx_cuas::kernel_fn<T> sym_grad_eps
      = [=](T* A, const T* c, const T* w, const double* coordinate_dofs,
            const int* entity_local_index, const std::uint8_t* quadrature_permutation)
  {
    assert(bs == 3);
    // Create buffers for jacobian (inverse and determinant) computations
    std::vector<double> Jb(gdim * tdim);
    std::vector<double> Kb(tdim * gdim);
    mdspan2_t J(Jb.data(), gdim, tdim);
    mdspan2_t K(Kb.data(), tdim, gdim);
    cmdspan2_t coords(coordinate_dofs, shape);
    std::vector<double> detJ_scratch(2 * gdim * tdim);

    // Compute Jacobian, its inverse and the determinant
    cmdspan4_t dphi_c_full(coordinate_basis.data(), tab_shape);

    auto dphi_c_0 = stdex::submdspan(dphi_c_full, std::pair(1, tdim + 1), 0, stdex::full_extent, 0);
    dolfinx::fem::CoordinateElement::compute_jacobian(dphi_c_0, coords, J);
    dolfinx::fem::CoordinateElement::compute_jacobian_inverse(J, K);
    const double detJ
        = std::fabs(dolfinx::fem::CoordinateElement::compute_jacobian_determinant(J, detJ_scratch));

    // Get derivatives of element basis
    cmdspan3_t dphi(dphib.data(), tdim, basis_shape[1], basis_shape[2]);

    // Temporary variable for grad(phi) on physical cell
    std::vector<double> dphi_physb(
        std::reduce(shape_d.begin(), shape_d.end(), 1, std::multiplies{}));
    mdspan2_t dphi_phys(dphi_physb.data(), shape_d);

    // Main loop
    for (std::size_t q = 0; q < weights.size(); q++)
    {
      const double w0 = weights[q] * detJ;
      // Precompute J^-T * dphi
      std::fill(dphi_physb.begin(), dphi_physb.end(), 0);
      for (int i = 0; i < ndofs_cell; i++)
      {
        for (int j = 0; j < bs; j++)
        {
          double acc = 0;
          for (int k = 0; k < tdim; k++)
            acc += K(k, j) * dphi(q, i, k);
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
template <typename T>
dolfinx_cuas::kernel_fn<T> generate_kernel(dolfinx_cuas::Kernel type, int P, int bs,
                                           dolfinx_cuas::QuadratureRule& quadrature_rule)
{
  switch (P)
  {
  case 1:
    switch (bs)
    {
    case 1:
      return generate_tet_kernel<1, 1, T>(type, quadrature_rule);
    case 2:
      return generate_tet_kernel<1, 2, T>(type, quadrature_rule);
    case 3:
      return generate_tet_kernel<1, 3, T>(type, quadrature_rule);
    default:
      throw std::runtime_error("Can only have block size from 1 to 3.");
    }
  case 2:
    switch (bs)
    {
    case 1:
      return generate_tet_kernel<2, 1, T>(type, quadrature_rule);
    case 2:
      return generate_tet_kernel<2, 2, T>(type, quadrature_rule);
    case 3:
      return generate_tet_kernel<2, 3, T>(type, quadrature_rule);
    default:
      throw std::runtime_error("Can only have block size from 1 to 3.");
    }
  case 3:
    switch (bs)
    {
    case 1:
      return generate_tet_kernel<3, 1, T>(type, quadrature_rule);
    case 2:
      return generate_tet_kernel<3, 2, T>(type, quadrature_rule);
    case 3:
      return generate_tet_kernel<3, 3, T>(type, quadrature_rule);
    default:
      throw std::runtime_error("Can only have block size from 1 to 3.");
    }
  case 4:
    switch (bs)
    {
    case 1:
      return generate_tet_kernel<4, 1, T>(type, quadrature_rule);
    case 2:
      return generate_tet_kernel<4, 2, T>(type, quadrature_rule);
    case 3:
      return generate_tet_kernel<4, 3, T>(type, quadrature_rule);
    default:
      throw std::runtime_error("Can only have block size from 1 to 3.");
    }
  case 5:
    switch (bs)
    {
    case 1:
      return generate_tet_kernel<5, 1, T>(type, quadrature_rule);
    case 2:
      return generate_tet_kernel<5, 2, T>(type, quadrature_rule);
    case 3:
      return generate_tet_kernel<5, 3, T>(type, quadrature_rule);
    default:
      throw std::runtime_error("Can only have block size from 1 to 3.");
    }
  default:
    throw std::runtime_error("Custom kernel only supported up to 5th order");
  }
}
} // namespace dolfinx_cuas
