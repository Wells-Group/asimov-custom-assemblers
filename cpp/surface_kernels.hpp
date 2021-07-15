
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

  // Creacte _qp_ref_facet (quadrature points on reference facet)
  auto mesh = V->mesh();
  auto [qp_ref_facet, qw_ref_facet] = create_reference_facet_qp(mesh, quadrature_degree);

  // Starting with implementing the following term in Jacobian:
  // u*v*ds
  // Mesh info
  const int gdim = mesh->geometry().dim(); // geometrical dimension
  const int tdim = mesh->topology().dim(); // topological dimension
  const int fdim = tdim - 1;               // topological dimesnion of facet

  // FIXME: Need basix element public in mesh
  // int degree = mesh->geometry().cmap().degree();
  int degree = 1; // element degree

  const dolfinx::mesh::CellType dolfinx_cell = mesh->topology().cell_type();
  const basix::cell::type basix_cell
      = basix::cell::str_to_type(dolfinx::mesh::to_string(dolfinx_cell)); // basix cell type
  auto dolfinx_facet
      = dolfinx::mesh::cell_entity_type(dolfinx_cell, fdim);        // dolfinx facet cell type
  auto dolfinx_facet_str = dolfinx::mesh::to_string(dolfinx_facet); // facet cell type as string
  auto basix_facet = basix::cell::str_to_type(dolfinx_facet_str);   // basix facet cell type
  auto facets = basix::cell::topology(basix_cell)[tdim - 1];

  // Create facet quadrature points
  auto quadrature_points
      = basix::quadrature::make_quadrature("default", basix_facet, quadrature_degree).first;

  basix::FiniteElement surface_element
      = basix::create_element("Lagrange", dolfinx_facet_str, degree);
  basix::FiniteElement basix_element
      = basix::create_element("Lagrange", dolfinx::mesh::to_string(dolfinx_cell), degree);
  const int num_coordinate_dofs = basix_element.dim();
  // tabulate on reference facet
  auto f_tab = surface_element.tabulate(1, quadrature_points);
  xt::xtensor<double, 2> dphi0_f
      = xt::round(xt::view(f_tab, xt::range(1, tdim + 1), 0, xt::all(), 0));

  // tabulate on reference cell
  // not quite the right quadrature points if jacobian non-constant
  auto qp_cell = xt::view(qp_ref_facet, 0, xt::all(), xt::all());

  auto c_tab = basix_element.tabulate(1, qp_cell);
  xt::xtensor<double, 2> dphi0_c
      = xt::round(xt::view(c_tab, xt::range(1, tdim + 1), 0, xt::all(), 0));

  std::uint32_t num_quadrature_pts = qp_ref_facet.shape(1);
  std::uint32_t num_facets = qp_ref_facet.shape(0);
  std::shared_ptr<const dolfinx::fem::FiniteElement> element = V->element();
  int bs = element->block_size();
  std::uint32_t num_local_dofs = element->space_dimension() / bs;
  xt::xtensor<double, 3> phi({num_facets, num_quadrature_pts, num_local_dofs});
  xt::xtensor<double, 4> dphi({num_facets, tdim, num_quadrature_pts, num_local_dofs});
  xt::xtensor<double, 4> cell_tab({tdim + 1, num_quadrature_pts, num_local_dofs, bs});
  auto ref_jacobians = basix::cell::facet_jacobians(basix_cell);
  const xt::xtensor<double, 2> x = basix::cell::geometry(basix_cell);
  // tabulate at quadrature points on facet
  for (int i = 0; i < num_facets; ++i)
  {
    auto phi_i = xt::view(phi, i, xt::all(), xt::all());
    auto dphi_i = xt::view(dphi, i, xt::all(), xt::all(), xt::all());
    auto q_facet = xt::view(qp_ref_facet, i, xt::all(), xt::all());
    element->tabulate(cell_tab, q_facet, 1);
    phi_i = xt::view(cell_tab, 0, xt::all(), xt::all(), 0);
    dphi_i = xt::view(cell_tab, xt::range(1, tdim + 1), xt::all(), xt::all(), 0);
  }

  auto q_weights = qw_ref_facet;
  kernel_fn mass = [facets, dphi0_f, phi, gdim, tdim, fdim, bs, q_weights, num_coordinate_dofs](
                       double* A, const double* c, const double* w, const double* coordinate_dofs,
                       const int* entity_local_index, const std::uint8_t* quadrature_permutation)
  {
    // Compute Jacobian at each quadrature point
    xt::xtensor<double, 2> J = xt::zeros<double>({gdim, fdim});

    // NOTE: DOLFINx has 3D input coordinate dofs
    std::array<std::size_t, 2> shape = {num_coordinate_dofs, 3};
    xt::xtensor<double, 2> coord
        = xt::adapt(coordinate_dofs, num_coordinate_dofs * 3, xt::no_ownership(), shape);
    dolfinx_cuas::math::compute_jacobian(dphi0_f,
                                         xt::view(coord, xt::keep(facets[*entity_local_index])), J);
    double detJ = std::fabs(dolfinx_cuas::math::compute_determinant(J));
    // Get number of dofs per cell
    std::int32_t ndofs_cell = phi.shape(2);
    // Main loop
    for (std::size_t q = 0; q < phi.shape(1); q++)
    {
      double w0 = q_weights[q] * detJ;

      for (int i = 0; i < ndofs_cell; i++)
      {
        double w1 = w0 * phi(*entity_local_index, q, i);
        for (int j = 0; j < ndofs_cell; j++)
        {
          double value = w1 * phi(*entity_local_index, q, j);
          for (int k = 0; k < bs; k++)
          {
            A[(k + i * bs) * (ndofs_cell * bs) + k + j * bs] += value;
          }
        }
      }
    }
  };

  kernel_fn stiffness
      = [facets, dphi0_f, dphi, gdim, tdim, fdim, bs, dphi0_c, q_weights, num_coordinate_dofs](
            double* A, const double* c, const double* w, const double* coordinate_dofs,
            const int* entity_local_index, const std::uint8_t* quadrature_permutation)
  {
    // Compute Jacobian at each quadrature point: currently assumed to be constant...
    xt::xtensor<double, 2> J_facet = xt::zeros<double>({gdim, fdim});
    xt::xtensor<double, 2> J = xt::zeros<double>({gdim, tdim});
    xt::xtensor<double, 2> K = xt::zeros<double>({tdim, gdim});

    // NOTE: DOLFINx has 3D input coordinate dofs
    std::array<std::size_t, 2> shape = {num_coordinate_dofs, 3};
    xt::xtensor<double, 2> coord
        = xt::adapt(coordinate_dofs, num_coordinate_dofs * 3, xt::no_ownership(), shape);
    dolfinx_cuas::math::compute_jacobian(
        dphi0_f, xt::view(coord, xt::keep(facets[*entity_local_index])), J_facet);
    dolfinx_cuas::math::compute_jacobian(dphi0_c, coord, J);
    dolfinx_cuas::math::compute_inv(J, K);

    // Get number of dofs per cell
    std::int32_t ndofs_cell = dphi.shape(3);

    double detJ = std::fabs(dolfinx_cuas::math::compute_determinant(J_facet));

    xt::xtensor<double, 2> temp({gdim, ndofs_cell});
    // Main loop

    for (std::size_t q = 0; q < dphi.shape(2); q++)
    {
      double w0 = q_weights[q] * detJ; //

      // precompute J^-T * dphi in temporary array temp
      for (int i = 0; i < ndofs_cell; i++)
      {

        for (int j = 0; j < gdim; j++)
        {
          temp(j, i) = 0;
          for (int k = 0; k < tdim; k++)
          {
            temp(j, i) += K(k, j) * dphi(*entity_local_index, k, q, i);
          }
        }
      }

      for (int i = 0; i < ndofs_cell; i++)
      {
        for (int j = 0; j < ndofs_cell; j++)
        {
          double value = 0;
          for (int k = 0; k < gdim; k++)
          {
            value += temp(k, i) * temp(k, j) * w0;
          }
          for (int k = 0; k < bs; k++)
          {
            A[(k + i * bs) * (ndofs_cell * bs) + k + j * bs] += value;
          }
        }
      }
    }
  };

  kernel_fn contact_jac
      = [facets, dphi0_f, dphi, gdim, tdim, fdim, bs, dphi0_c, q_weights, num_coordinate_dofs](
            double* A, const double* c, const double* w, const double* coordinate_dofs,
            const int* entity_local_index, const std::uint8_t* quadrature_permutation)
  {
    assert(bs == tdim);
    // Compute Jacobian at each quadrature point: currently assumed to be constant...
    xt::xtensor<double, 2> J_facet = xt::zeros<double>({gdim, fdim});
    xt::xtensor<double, 2> J = xt::zeros<double>({gdim, tdim});
    xt::xtensor<double, 2> K = xt::zeros<double>({tdim, gdim});

    // NOTE: DOlFINx assumes 3D coordinate dofs input
    std::array<std::size_t, 2> shape = {num_coordinate_dofs, 3};
    xt::xtensor<double, 2> coord
        = xt::adapt(coordinate_dofs, num_coordinate_dofs * 3, xt::no_ownership(), shape);

    dolfinx_cuas::math::compute_jacobian(
        dphi0_f, xt::view(coord, xt::keep(facets[*entity_local_index])), J_facet);

    dolfinx_cuas::math::compute_jacobian(dphi0_c, coord, J);
    dolfinx_cuas::math::compute_inv(J, K);
    // Get number of dofs per cell
    std::int32_t ndofs_cell = dphi.shape(3);

    double detJ = std::fabs(dolfinx_cuas::math::compute_determinant(J_facet));

    xt::xtensor<double, 2> temp({gdim, ndofs_cell});
    // Main loop
    for (std::size_t q = 0; q < dphi.shape(2); q++)
    {
      double w0 = q_weights[q] * detJ; //

      // precompute J^-T * dphi in temporary array temp
      for (int i = 0; i < ndofs_cell; i++)
      {

        for (int j = 0; j < gdim; j++)
        {
          temp(j, i) = 0;
          for (int k = 0; k < tdim; k++)
          {
            temp(j, i) += K(k, j) * dphi(*entity_local_index, k, q, i);
          }
        }
      }
      // This currently corresponds to the term sym(grad(u)):sym(grad(v)) (see
      // https://www.overleaf.com/2212919918tbbqtnmnrynf for details)
      for (int i = 0; i < ndofs_cell; i++)
      {
        for (int j = 0; j < ndofs_cell; j++)
        {
          // Compute sum_t dphi^j/dx_t dphi^i/dx_t
          double value = 0;
          for (int t = 0; t < gdim; t++)
            value += 0.5 * w0 * temp(t, i) * temp(t, j);

          for (int k = 0; k < bs; ++k)
          {
            A[(k + i * bs) * (ndofs_cell * bs) + (j * bs + k)] += value;
            for (int l = 0; l < bs; ++l)
            {
              // Add dphi^j/dx_k dphi^i/dx_l
              A[(k + i * bs) * (ndofs_cell * bs) + (j * bs + l)]
                  += 0.5 * w0 * temp(l, i) * temp(k, j);
            }
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
    return contact_jac;
  default:
    throw std::runtime_error("unrecognized kernel");
  }
}

} // namespace dolfinx_cuas