
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
namespace dolfinx_cuas
{

template <typename T>
kernel_fn<T> generate_surface_kernel(std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
                                     dolfinx_cuas::Kernel type,
                                     dolfinx_cuas::QuadratureRule& quadrature_rule)
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
  const std::vector<xt::xtensor<double, 2>>& q_points = quadrature_rule.points_ref();
  const std::vector<std::vector<double>>& q_weights = quadrature_rule.weights_ref();

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
  kernel_fn<T> mass = [=](T* A, const T* c, const T* w, const double* coordinate_dofs,
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

  kernel_fn<T> mass_nonaffine
      = [=](T* A, const T* c, const T* w, const double* coordinate_dofs,
            const int* entity_local_index, const std::uint8_t* quadrature_permutation)
  {
    std::size_t facet_index = size_t(*entity_local_index);

    // Get basis values
    auto [basis, shape] = basis_values[facet_index];
    auto [cbasis, cshape] = coordinate_basis_values[facet_index];

    // Reshape coordinate dofs
    cmdspan2_t coords(coordinate_dofs, cd_shape);
    auto c_view = stdex::submdspan(coords, stdex::full_extent, std::pair{0, gdim});

    // Compute Jacobian and determinant on facet
    std::vector<double> Jb(gdim * tdim);
    std::vector<double> Kb(tdim * gdim);
    mdspan2_t J(Jb.data(), gdim, tdim);
    mdspan2_t K(Kb.data(), tdim, gdim);
    std::vector<double> detJ_scratch(2 * gdim * tdim);

    // Extract reference Jacobian at facet
    cmdspan3_t reference_jacobians(ref_jac.data(), jac_shape);
    auto J_f = stdex::submdspan(reference_jacobians, facet_index, stdex::full_extent,
                                stdex::full_extent);
    std::vector<double> J_totb(J.extent(0) * J_f.extent(1));
    mdspan2_t J_tot(J_totb.data(), J.extent(0), J_f.extent(1));

    // Get number of dofs per cell
    const std::vector<double>& weights = q_weights[facet_index];

    // Extract basis values of test/trial function
    cmdspan4_t phi_full(basis.data(), shape);
    auto phi_f = stdex::submdspan(phi_full, 0, stdex::full_extent, stdex::full_extent, 0);
    cmdspan4_t phi_c_full(cbasis.data(), cshape);

    // Loop over quadrature points
    for (std::size_t q = 0; q < weights.size(); q++)
    {

      // Extract the first derivative of the coordinate element (cell) of degrees of freedom
      // on the facet

      auto dphi_c_q
          = stdex::submdspan(phi_c_full, std::pair(1, tdim + 1), q, stdex::full_extent, 0);
      std::fill(Jb.begin(), Jb.end(), 0);
      dolfinx::fem::CoordinateElement::compute_jacobian(dphi_c_q, c_view, J);
      std::fill(J_totb.begin(), J_totb.end(), 0);
      dolfinx::math::dot(J, J_f, J_tot);

      // NOTE: Remove once https://github.com/FEniCS/dolfinx/pull/2291 is merged
      std::fill(detJ_scratch.begin(), detJ_scratch.end(), 0);
      const double detJ = std::fabs(
          dolfinx::fem::CoordinateElement::compute_jacobian_determinant(J_tot, detJ_scratch));

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
  kernel_fn<T> stiffness
      = [=](T* A, const T* c, const T* w, const double* coordinate_dofs,
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
    auto dphi_f = stdex::submdspan(phi_full, std::pair(1, tdim + 1), stdex::full_extent,
                                   stdex::full_extent, 0);

    std::vector<double> dphi_physb(gdim * shape[2]);
    mdspan2_t dphi_phys(dphi_physb.data(), gdim, shape[2]);
    // Loop over quadrature points.
    for (std::size_t q = 0; q < weights.size(); q++)
    {
      // Scale for integral. NOTE: for non-simplices detJ is detJ[q]
      const double w0 = weights[q] * detJ;

      // Precompute J^-T * dphi
      std::fill(dphi_physb.begin(), dphi_physb.end(), 0);
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

  kernel_fn<T> sym_grad
      = [=](T* A, const T* c, const T* w, const double* coordinate_dofs,
            const int* entity_local_index, const std::uint8_t* quadrature_permutation)
  {
    std::size_t facet_index = size_t(*entity_local_index);
    assert(is_affine);

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
    auto dphi_f = stdex::submdspan(phi_full, std::pair(1, tdim + 1), stdex::full_extent,
                                   stdex::full_extent, 0);

    std::vector<double> dphi_physb(gdim * shape[2]);
    mdspan2_t dphi_phys(dphi_physb.data(), gdim, shape[2]);

    // Loop over quadrature points
    for (std::size_t q = 0; q < weights.size(); q++)
    {
      // Create for integral. NOTE: for non-simplices detJ is detJ[q]
      const double w0 = weights[q] * detJ;

      // Precompute J^-T * dphi
      std::fill(dphi_physb.begin(), dphi_physb.end(), 0);
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
  kernel_fn<T> normal
      = [=](T* A, const T* c, const T* w, const double* coordinate_dofs,
            const int* entity_local_index, const std::uint8_t* quadrature_permutation)
  {
    assert(bs == tdim);
    std::size_t facet_index = size_t(*entity_local_index);
    assert(is_affine);

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

    // Compute normal of physical facet using a normalized covariant Piola transform
    // n_phys = J^{-T} n_ref / ||J^{-T} n_ref||
    // See for instance DOI: 10.1137/08073901X
    cmdspan2_t normals_f(facet_normals.data(), normal_shape);
    std::vector<double> n_phys(gdim);
    for (std::size_t i = 0; i < gdim; i++)
      for (std::size_t j = 0; j < tdim; j++)
        n_phys[i] += K(j, i) * normals_f(facet_index, j);
    double n_norm = 0;
    for (std::size_t i = 0; i < gdim; i++)
      n_norm += n_phys[i] * n_phys[i];
    n_norm = std::sqrt(n_norm);
    std::for_each(n_phys.begin(), n_phys.end(), [n_norm](auto& n) { n /= n_norm; });

    // Get number of dofs per cell
    const std::vector<double>& weights = q_weights[facet_index];

    // Extract basis values of test/trial function
    cmdspan4_t phi_full(basis.data(), shape);
    auto phi_f = stdex::submdspan(phi_full, 0, stdex::full_extent, stdex::full_extent, 0);
    auto dphi_f = stdex::submdspan(phi_full, std::pair(1, tdim + 1), stdex::full_extent,
                                   stdex::full_extent, 0);

    std::vector<double> dphi_physb(gdim * shape[2]);
    mdspan2_t dphi_phys(dphi_physb.data(), gdim, shape[2]);

    // Loop over quadrature points
    for (std::size_t q = 0; q < weights.size(); q++)
    {
      // Create for integral. NOTE: for non-simplices detJ is detJ[q]
      const double w0 = weights[q] * detJ;

      // Precompute J^-T * dphi
      std::fill(dphi_physb.begin(), dphi_physb.end(), 0);
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
            block_invariant_cont += dphi_phys(s, i) * n_phys[s] * phi_f(q, j);
          block_invariant_cont *= w0;

          for (int l = 0; l < bs; ++l)
          {
            const std::size_t row = (l + j * bs) * (num_local_dofs * bs);
            A[row + i * bs + l] += block_invariant_cont;

            // Add dphi^j/dx_k dphi^i/dx_l
            for (int b = 0; b < bs; ++b)
              A[row + i * bs + b] += w0 * dphi_phys(l, i) * n_phys[b] * phi_f(q, j);
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
