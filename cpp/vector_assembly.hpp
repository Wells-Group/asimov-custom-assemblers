// Copyright (C) 2021 Sarah Roggendorf
//
// This file is part of DOLFINx_CUAS
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "kernels.hpp"
#include "utils.hpp"
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/assembler.h>
#include <functional>
#include <xtl/xspan.hpp>

// Helper functions for assembly in DOLFINx
namespace
{
/// Assemble vector over exterior facets
/// Provides easier interface to dolfinx::fem::impl::assemble_exterior_facets
/// @param[in,out] b the vector to be assembled
/// @param[in] V the function space
/// @param[in] active_facets list of indices (local to process) of facets to be integrated over
/// @param[in] kernel the custom integration kernel
/// @param[in] coeffs coefficients used in the variational form
/// @param[in] cstride Number of coefficients per cell
/// @param[in] constants used in the variational form
template <typename T>
void assemble_exterior_facets(xtl::span<T> b, std::shared_ptr<dolfinx::fem::FunctionSpace> V,
                              const xtl::span<const std::int32_t>& active_facets,
                              dolfinx_cuas::kernel_fn<T>& kernel, const xtl::span<const T> coeffs,
                              int cstride, const xtl::span<const T>& constants)
{
  // Extract mesh
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = V->mesh();

  // Extract function space data (assuming same test and trial space)
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = V->dofmap();
  const dolfinx::graph::AdjacencyList<std::int32_t>& dofs = dofmap->list();
  const int bs = dofmap->bs();
  std::shared_ptr<const dolfinx::fem::FiniteElement> element = V->element();
  const std::function<void(const xtl::span<T>&, const xtl::span<const std::uint32_t>&, std::int32_t,
                           int)>
      apply_dof_transformation = element->get_dof_transformation_function<T>();

  // NOTE: Need to reconsider this when we get to jump integrals between disconnected interfaces
  const bool needs_transformation_data = element->needs_dof_transformations();

  // Get permutation data
  xtl::span<const std::uint32_t> cell_info;
  if (needs_transformation_data)
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
  }

  // Create facet tuple: cell_index (local to process) and facet_index (local to cell)
  int tdim = mesh->topology().dim();
  auto f_to_c = mesh->topology().connectivity(tdim - 1, tdim);
  assert(f_to_c);
  auto c_to_f = mesh->topology().connectivity(tdim, tdim - 1);
  assert(c_to_f);
  std::vector<std::int32_t> facets(2 * active_facets.size());
  for (std::size_t f = 0; f < active_facets.size(); ++f)
  {
    auto cells = f_to_c->links(active_facets[f]);
    assert(cells.size() == 1);
    auto cell_facets = c_to_f->links(cells[0]);
    const auto* it = std::find(cell_facets.begin(), cell_facets.end(), active_facets[f]);
    assert(it != cell_facets.end());
    facets[2 * f] = cells[0];
    facets[2 * f + 1] = std::distance(cell_facets.data(), it);
  }

  // Assemble using dolfinx
  dolfinx::fem::impl::assemble_exterior_facets<T>(apply_dof_transformation, b, *mesh, facets, dofs,
                                                  bs, kernel, constants, coeffs, cstride,
                                                  cell_info);
}

/// Assemble vector over cells
/// Provides easier interface to dolfinx::fem::impl::assemble_cells
/// @param[in,out] b the vector to be assembled
/// @param[in] V the function space
/// @param[in] active_cells list of indices (local to process) of cells to be integrated over
/// @param[in] kernel the custom integration kernel
/// @param[in] coeffs coefficients used in the variational form
/// @param[in] cstride Number of coefficients per cell
/// @param[in] constants used in the variational form
template <typename T>
void assemble_cells(xtl::span<T> b, std::shared_ptr<dolfinx::fem::FunctionSpace> V,
                    const xtl::span<const std::int32_t>& active_cells,
                    dolfinx_cuas::kernel_fn<T>& kernel, const xtl::span<const T> coeffs,
                    int cstride, const xtl::span<const T>& constants)
{
  // Extract mesh
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = V->mesh();

  // Extract function space data
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = V->dofmap();
  const dolfinx::graph::AdjacencyList<std::int32_t>& dofs = dofmap->list();
  const int bs = dofmap->bs();
  std::shared_ptr<const dolfinx::fem::FiniteElement> element = V->element();
  const std::function<void(const xtl::span<T>&, const xtl::span<const std::uint32_t>&, std::int32_t,
                           int)>
      apply_dof_transformation = element->get_dof_transformation_function<T>();

  // NOTE: Need to reconsider this when we get to jump integrals between disconnected interfaces
  const bool needs_transformation_data = element->needs_dof_transformations();

  // Get permutation data
  xtl::span<const std::uint32_t> cell_info;
  if (needs_transformation_data)
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
  }

  dolfinx::fem::impl::assemble_cells<T>(apply_dof_transformation, b, mesh->geometry(), active_cells,
                                        dofs, bs, kernel, constants, coeffs, cstride, cell_info);
}
} // namespace

namespace dolfinx_cuas
{

/// Assemble vector for given kernel function
/// @param[in,out] b the vector to be assembled
/// @param[in] V the function space
/// @param[in] active_entities list of indices (local to process) of entities to be integrated over
/// @param[in] kernel the custom integration kernel
/// @param[in] coefficients used in the variational form
/// @param[in] cstride Number of coefficients per cell
/// @param[in] constants used in the variational form
/// @param[in] type the integral type
template <typename T>
void assemble_vector(xtl::span<T> b, std::shared_ptr<dolfinx::fem::FunctionSpace> V,
                     const xtl::span<const std::int32_t>& active_entities,
                     dolfinx_cuas::kernel_fn<T>& kernel, const xtl::span<const T> coeffs,
                     int cstride, const xtl::span<const T>& constants,
                     dolfinx::fem::IntegralType type)
{

  // Assemble integral
  if (type == dolfinx::fem::IntegralType::cell)
    assemble_cells(b, V, active_entities, kernel, coeffs, cstride, constants);
  else if (type == dolfinx::fem::IntegralType::exterior_facet)
    assemble_exterior_facets(b, V, active_entities, kernel, coeffs, cstride, constants);
  else
    throw std::runtime_error("Unsupported integral type");
};

} // namespace dolfinx_cuas
