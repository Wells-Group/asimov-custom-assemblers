
// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINx_CUAS
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <basix/cell.h>
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx/mesh/Mesh.h>
#include <xtensor-blas/xlinalg.hpp>

namespace dolfinx_cuas
{

/// Given a mesh and an entity dimension, return the corresponding basix element of the entity
/// @param[in] mesh The mesh
/// @param[in] dim Dimension of entity
/// @return The basix element
basix::FiniteElement mesh_to_basix_element(std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
                                           const int dim)
{
  // FIXME: Support of higher order cmap
  // Fixed in https://github.com/FEniCS/dolfinx/pull/1618
  // const int degree = mesh->geometry().cmap().degree();
  const int degree = 1; // element degree

  const int tdim = mesh->topology().dim(); // topological dimension
  const int fdim = tdim - 1;               // topological dimension of facet

  // Get necessary DOLFINx and basix facet and cell types
  const dolfinx::mesh::CellType dolfinx_cell = mesh->topology().cell_type();
  if (dim == fdim)
  {
    const dolfinx::mesh::CellType dolfinx_facet
        = dolfinx::mesh::cell_entity_type(dolfinx_cell, fdim);
    const std::string dolfinx_facet_str = dolfinx::mesh::to_string(dolfinx_facet);
    return basix::create_element("Lagrange", dolfinx_facet_str, degree);
  }
  if (dim == tdim)
  {
    return basix::create_element("Lagrange", dolfinx::mesh::to_string(dolfinx_cell), degree);
  }
  else
    throw std::runtime_error("Does not support elements of edges and vertices");
}

/// Convert DOLFINx CellType to basix cell type
const basix::cell::type to_basix_celltype(dolfinx::mesh::CellType celltype)
{
  return basix::cell::str_to_type(dolfinx::mesh::to_string(celltype));
}

// Compute quadrature points and weights on all facets of the reference cell
/// by pushing them forward from the reference facet.
/// @param[in] mesh The mesh
/// @param[in] quadrature_degree Degree of quadrature rule
std::pair<xt::xtensor<double, 3>, std::vector<double>>
create_reference_facet_qp(std::shared_ptr<const dolfinx::mesh::Mesh> mesh, int quadrature_degree)
{
  // Mesh info
  const int tdim = mesh->topology().dim(); // topological dimension
  const int fdim = tdim - 1;               // topological dimesnion of facet

  const basix::cell::type basix_cell = to_basix_celltype(mesh->topology().cell_type());

  // Create basix facet coordinate element
  const basix::FiniteElement surface_element = mesh_to_basix_element(mesh, fdim);

  // Create facet quadrature points
  const basix::cell::type basix_facet = surface_element.cell_type();
  std::pair<xt::xarray<double>, std::vector<double>> quadrature
      = basix::quadrature::make_quadrature("default", basix_facet, quadrature_degree);

  // Tabulate facet coordinate functions
  auto c_tab = surface_element.tabulate(0, quadrature.first);
  xt::xtensor<double, 2> phi_s = xt::view(c_tab, 0, xt::all(), xt::all(), 0);

  // Create reference topology and geometry
  auto facet_topology = basix::cell::topology(basix_cell)[fdim];
  const xt::xtensor<double, 2> ref_geom = basix::cell::geometry(basix_cell);

  // Push forward quadrature points on reference facet to reference cell
  const std::uint32_t num_facets = facet_topology.size();
  const std::uint32_t num_quadrature_pts = quadrature.first.shape(0);
  xt::xtensor<double, 3> qp_ref_facet({num_facets, num_quadrature_pts, ref_geom.shape(1)});
  for (int i = 0; i < num_facets; ++i)
  {
    auto facet = facet_topology[i];
    auto coords = xt::view(ref_geom, xt::keep(facet), xt::all());
    auto q_facet = xt::view(qp_ref_facet, i, xt::all(), xt::all());
    q_facet = xt::linalg::dot(phi_s, coords);
  }
  return {qp_ref_facet, quadrature.second};
}

} // namespace dolfinx_cuas