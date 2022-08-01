// Copyright (C) 2021-2022 Jørgen S. Dokken, Igor A. Baratta, Sarah Roggendorf
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
  constexpr std::array<std::size_t, 2> coordinate_shape = {d, gdim};

  // Get quadrature points and quadrature weights
  const std::vector<double>& weights = q_rule.weights_ref()[0];
  const xt::xarray<double>& points = q_rule.points_ref()[0];

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
  coordinate_element.tabulate(1, cmdspan2_t(points.data(), pts_shape),
                              mdspan4_t(coordinate_basis.data(), tab_shape));

  assert(ndofs_cell == static_cast<std::int32_t>(tab_shape[2]));

  // Fetch finite elements for coefficient functions and tabulate shape functions
  int num_coeffs = coeffs.size();
  std::vector<int> offsets(num_coeffs + 1);
  offsets[0] = 0;
  for (int i = 1; i < num_coeffs + 1; i++)
    offsets[i] = offsets[i - 1] + coeffs[i - 1]->function_space()->element()->space_dimension();
  std::vector<double> phi_coeffsb(weights.size() * offsets.back());
  mdspan2_t phi_coeffs(phi_coeffsb.data(), weights.size(), offsets.back());
  for (int i = 0; i < num_coeffs; i++)
  {
    std::shared_ptr<const dolfinx::fem::FiniteElement> coeff_element
        = coeffs[i]->function_space()->element();
    std::array<std::size_t, 4> coeff_shape
        = coeff_element->basix_element().tabulate_shape(0, weights.size());
    assert(coeff_shape.back() == 1);
    std::vector<double> coeff_basis(
        std::reduce(coeff_shape.cbegin(), coeff_shape.cend(), 1, std::multiplies()));
    coeff_element->tabulate(coeff_basis, std::span(points.data(), points.size()),
                            {points.shape(0), points.shape(1)}, 0);
    cmdspan4_t cb(coeff_basis.data(), coeff_shape);
    auto phi_i
        = stdex::submdspan(phi_coeffs, stdex::full_extent, std::pair(offsets[i], offsets[i + 1]));
    for (std::size_t j = 0; j < phi_i.extent(0); ++j)
      for (std::size_t k = 0; k < phi_i.extent(1); ++k)
        phi_i(j, k) = cb(0, j, k, 0);
  }

  // Mass Matrix using quadrature formulation
  // =====================================================================================
  dolfinx_cuas::kernel_fn<T> mass_coeff
      = [=](T* A, const T* c, const T* w, const double* coordinate_dofs,
            const int* entity_local_index, const std::uint8_t* quadrature_permutation)
  {
    // Create buffers for jacobian (inverse and determinant) computations
    std::vector<double> Jb(gdim * tdim);
    std::vector<double> Kb(tdim * gdim);
    mdspan2_t J(Jb.data(), gdim, tdim);
    mdspan2_t K(Kb.data(), tdim, gdim);
    cmdspan2_t coords(coordinate_dofs, coordinate_shape);
    std::vector<double> detJ_scratch(2 * gdim * tdim);

    // Compute Jacobian, its inverse and the determinant
    cmdspan4_t dphi_c_full(coordinate_basis.data(), tab_shape);

    auto dphi_c_0 = stdex::submdspan(dphi_c_full, std::pair(1, tdim + 1), 0, stdex::full_extent, 0);
    dolfinx::fem::CoordinateElement::compute_jacobian(dphi_c_0, coords, J);
    dolfinx::fem::CoordinateElement::compute_jacobian_inverse(J, K);
    const double detJ
        = std::fabs(dolfinx::fem::CoordinateElement::compute_jacobian_determinant(J, detJ_scratch));

    // Get basis function views
    cmdspan2_t phi_coeffs(phi_coeffsb.data(), weights.size(), offsets.back());
    cmdspan4_t full_basis(basisb.data(), basis_shape);
    auto phi = stdex::submdspan(full_basis, 0, stdex::full_extent, stdex::full_extent, 0);

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
        T w1 = w0 * phi(q, i);
        for (int j = 0; j < ndofs_cell; j++)
          A[i * ndofs_cell + j] += w1 * phi(q, j);
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
