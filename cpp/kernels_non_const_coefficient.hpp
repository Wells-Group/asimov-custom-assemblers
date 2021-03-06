// Copyright (C) 2021 Jørgen S. Dokken, Igor A. Baratta, Sarah Roggendorf
//
// This file is part of DOLFINx_CUAS
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "kernels.hpp"
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx/fem/Function.h>
#include <string>

namespace
{
/// Create integration kernel for Pth order Lagrange elements
/// @param[in] type The kernel type (Mass or Stiffness)
/// @return The integration kernel
template <int P, typename T>
dolfinx_cuas::kernel_fn<T>
generate_coefficient_kernel(dolfinx_cuas::Kernel type,
                            std::vector<std::shared_ptr<const dolfinx::fem::Function<T>>> coeffs,
                            dolfinx_cuas::QuadratureRule& q_rule)
{
  // Problem specific parameters
  basix::element::family family = basix::element::family::P;
  basix::cell::type cell = basix::cell::type::tetrahedron;
  constexpr std::int32_t gdim = 3;
  constexpr std::int32_t tdim = 3;
  constexpr std::int32_t d = 4;
  constexpr std::int32_t ndofs_cell = (P + 1) * (P + 2) * (P + 3) / 6;

  // Get quadrature points and quadrature weights
  const std::vector<double>& weights = q_rule.weights_ref()[0];
  const xt::xarray<double>& points = q_rule.points_ref()[0];

  // Create Finite element for test and trial functions and tabulate shape functions
  basix::FiniteElement element
      = basix::create_element(family, cell, P, basix::element::lagrange_variant::gll_warped);
  std::array<std::size_t, 4> basis_shape = element.tabulate_shape(1, points.shape(0));
  xt::xtensor<double, 4> basis(basis_shape);
  std::array<std::size_t, 2> pts_shape = {points.shape(0), points.shape(1)};
  element.tabulate(1, basix::impl::cmdspan2_t(points.data(), pts_shape),
                   basix::impl::mdspan4_t(basis.data(), basis_shape));
  xt::xtensor<double, 2> phi = xt::view(basis, 0, xt::all(), xt::all(), 0);

  // Create Finite elements for coefficient functions and tabulate shape functions
  int num_coeffs = coeffs.size();
  std::vector<int> offsets(num_coeffs + 1);
  offsets[0] = 0;
  for (int i = 1; i < num_coeffs + 1; i++)
    offsets[i] = offsets[i - 1] + coeffs[i - 1]->function_space()->element()->space_dimension();
  xt::xtensor<double, 2> phi_coeffs({weights.size(), offsets[num_coeffs]});
  for (int i = 0; i < num_coeffs; i++)
  {
    std::shared_ptr<const dolfinx::fem::FiniteElement> coeff_element
        = coeffs[i]->function_space()->element();
    // NOTE: Assuming value size 1
    xt::xtensor<double, 4> coeff_basis({1, weights.size(), coeff_element->space_dimension(), 1});
    coeff_element->tabulate(coeff_basis, points, 0);
    auto phi_i = xt::view(phi_coeffs, xt::all(), xt::range(offsets[i], offsets[i + 1]));
    phi_i = xt::view(coeff_basis, 0, xt::all(), xt::all(), 0);
  }

  // Get coordinate element from dolfinx
  basix::FiniteElement coordinate_element
      = basix::create_element(family, cell, 1, basix::element::lagrange_variant::gll_warped);
  std::array<std::size_t, 4> tab_shape = coordinate_element.tabulate_shape(1, points.shape(0));
  xt::xtensor<double, 4> coordinate_basis(tab_shape);
  coordinate_element.tabulate(1, basix::impl::cmdspan2_t(points.data(), pts_shape),
                              basix::impl::mdspan4_t(coordinate_basis.data(), tab_shape));

  xt::xtensor<double, 2> dphi0_c
      = xt::view(coordinate_basis, xt::range(1, tdim + 1), 0, xt::all(), 0);

  assert(ndofs_cell == static_cast<std::int32_t>(phi.shape(1)));

  // Mass Matrix using quadrature formulation
  // =====================================================================================
  dolfinx_cuas::kernel_fn<T> mass_coeff
      = [=](T* A, const T* c, const T* w, const double* coordinate_dofs,
            const int* entity_local_index, const std::uint8_t* quadrature_permutation)
  {
    // Get geometrical data
    xt::xtensor<double, 2> J = xt::zeros<double>({gdim, tdim});
    std::array<std::size_t, 2> shape = {d, gdim};
    xt::xtensor<double, 2> coord = xt::adapt(coordinate_dofs, gdim * d, xt::no_ownership(), shape);

    // Compute Jacobian, its inverse and the determinant
    dolfinx::fem::CoordinateElement::compute_jacobian(dphi0_c, coord, J);
    double detJ = std::fabs(dolfinx::fem::CoordinateElement::compute_jacobian_determinant(J));

    // Main loop
    for (std::size_t q = 0; q < weights.size(); q++)
    {
      T w0 = 0;
      //  For each coefficient (assumed scalar valued), compute
      //  sum_{i=0}^{num_functions}sum_{j=0}^{num_dofs} c^j phi^j(x_q)
      for (int i = 0; i < num_coeffs; i++)
      {
        for (int j = offsets[i]; j < offsets[i + 1]; j++)
          w0 += c[j] * phi_coeffs(q, j);
      }
      w0 *= weights[q] * detJ;
      for (int i = 0; i < ndofs_cell; i++)
      {
        T w1 = w0 * phi.unchecked(q, i);
        for (int j = 0; j < ndofs_cell; j++)
          A[i * ndofs_cell + j] += w1 * phi.unchecked(q, j);
      }
    }
  };

  switch (type)
  {
  case dolfinx_cuas::Kernel::Mass:
    return mass_coeff;
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
/// @return The integration kernel
template <typename T>
dolfinx_cuas::kernel_fn<T>
generate_coeff_kernel(dolfinx_cuas::Kernel type,
                      std::vector<std::shared_ptr<const dolfinx::fem::Function<T>>> coeffs, int P,
                      dolfinx_cuas::QuadratureRule& q_rule)
{
  switch (P)
  {
  case 1:
    return generate_coefficient_kernel<1>(type, coeffs, q_rule);
  case 2:
    return generate_coefficient_kernel<2>(type, coeffs, q_rule);
  case 3:
    return generate_coefficient_kernel<3>(type, coeffs, q_rule);
  case 4:
    return generate_coefficient_kernel<4>(type, coeffs, q_rule);
  case 5:
    return generate_coefficient_kernel<5>(type, coeffs, q_rule);
  default:
    throw std::runtime_error("Custom kernel only supported up to 5th order");
  }
}
} // namespace dolfinx_cuas
