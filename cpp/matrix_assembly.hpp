// Copyright (C) 2021 Jørgen S. Dokken
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
#include <petscmat.h>
#include <xtl/xspan.hpp>

// Helper functions for assembly in DOLFINx
namespace
{

/// Assemble matrix over exterior facets
/// Provides easier interface to dolfinx::fem::impl::assemble_exterior_facets
/// @tparam T The scalar type
/// @tparam U The signature of the set local entries in matrix function
/// @param[in] mat_set the function for setting the values in the matrix
/// @param[in] V the function space
/// @param[in] active_facets list of indices (local to process) of facets to be integrated over
/// @param[in] kernel the custom integration kernel
/// @param[in] coeffs coefficients used in the variational form
/// @param[in] cstride Number of coefficients per cell
/// @param[in] constants used in the variational form
template <typename T, typename U>
void assemble_exterior_facets(U mat_set, std::shared_ptr<dolfinx::fem::FunctionSpace> V,
                              const std::vector<std::int8_t>& bc,
                              const std::span<const std::int32_t>& active_facets,
                              dolfinx_cuas::kernel_fn<T>& kernel, const std::span<const T> coeffs,
                              int cstride, const std::span<const T>& constants)
{
  // Extract mesh
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = V->mesh();

  // Extract function space data (assuming same test and trial space)
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = V->dofmap();
  const dolfinx::graph::AdjacencyList<std::int32_t>& dofs = dofmap->list();
  const int bs = dofmap->bs();
  std::shared_ptr<const dolfinx::fem::FiniteElement> element = V->element();
  const std::function<void(const std::span<T>&, const std::span<const std::uint32_t>&, std::int32_t,
                           int)>
      apply_dof_transformation = element->get_dof_transformation_function<T>();
  const std::function<void(const std::span<T>&, const std::span<const std::uint32_t>&, std::int32_t,
                           int)>
      apply_dof_transformation_to_transpose
      = element->get_dof_transformation_to_transpose_function<T>();

  // NOTE: Need to reconsider this when we get to jump integrals between disconnected interfaces
  const bool needs_transformation_data = element->needs_dof_transformations();

  // Get permutation data
  std::span<const std::uint32_t> cell_info;
  if (needs_transformation_data)
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = std::span(mesh->topology().get_cell_permutation_info());
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
    auto it = std::find(cell_facets.begin(), cell_facets.end(), active_facets[f]);
    assert(it != cell_facets.end());
    facets[2 * f] = cells[0];
    facets[2 * f + 1] = std::distance(cell_facets.begin(), it);
  }

  // Assemble using dolfinx
  dolfinx::fem::impl::assemble_exterior_facets(mat_set, *mesh, facets, apply_dof_transformation,
                                               dofs, bs, apply_dof_transformation_to_transpose,
                                               dofs, bs, bc, bc, kernel, coeffs, cstride, constants,
                                               cell_info);
}

/// Assemble vector over cells
/// Provides easier interface to dolfinx::fem::impl::assemble_cells
/// @tparam T The scalar type
/// @tparam U The signature of the set local entries in matrix function
/// @param[in] mat_set the function for setting the values in the matrix
/// @param[in] V the function space
/// @param[in] active_cells list of indices (local to process) of cells to be integrated over
/// @param[in] kernel the custom integration kernel
/// @param[in] coeffs coefficients used in the variational form
/// @param[in] cstride Number of coefficients per cell
/// @param[in] constants used in the variational form
template <typename T, typename U>
void assemble_cells(U mat_set, std::shared_ptr<dolfinx::fem::FunctionSpace> V,
                    const std::vector<std::int8_t>& bc,
                    const std::span<const std::int32_t>& active_cells,
                    dolfinx_cuas::kernel_fn<T>& kernel, const std::span<const T> coeffs,
                    int cstride, const std::span<const T>& constants)
{
  // Extract mesh
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = V->mesh();

  // Extract function space data
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = V->dofmap();
  const dolfinx::graph::AdjacencyList<std::int32_t>& dofs = dofmap->list();
  const int bs = dofmap->bs();
  std::shared_ptr<const dolfinx::fem::FiniteElement> element = V->element();
  const std::function<void(const std::span<T>&, const std::span<const std::uint32_t>&, std::int32_t,
                           int)>
      apply_dof_transformation = element->get_dof_transformation_function<T>();
  const std::function<void(const std::span<T>&, const std::span<const std::uint32_t>&, std::int32_t,
                           int)>
      apply_dof_transformation_to_transpose
      = element->get_dof_transformation_to_transpose_function<T>();

  // NOTE: Need to reconsider this when we get to jump integrals between disconnected interfaces
  const bool needs_transformation_data = element->needs_dof_transformations();

  // Get permutation data
  std::span<const std::uint32_t> cell_info;
  if (needs_transformation_data)
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = std::span(mesh->topology().get_cell_permutation_info());
  }

  // Assemble using dolfinx
  dolfinx::fem::impl::assemble_cells(mat_set, mesh->geometry(), active_cells,
                                     apply_dof_transformation, dofs, bs,
                                     apply_dof_transformation_to_transpose, dofs, bs, bc, bc,
                                     kernel, coeffs, cstride, constants, cell_info);
}
} // namespace

namespace dolfinx_cuas
{

/// Assemble matrix for given kernel function
/// @tparam T The scalar type
/// @tparam U The signature of the set local entries in matrix function
/// @param[in] mat_set the function for setting the values in the matrix
/// @param[in] V the function space
/// @param[in] bcs List of Dirichlet boundary conditions
/// @param[in] active_entities list of indices (local to process) of entities to be integrated over
/// @param[in] kernel the custom integration kernel
/// @param[in] coefficients used in the variational form
/// @param[in] cstride Number of coefficients per cell
/// @param[in] constants used in the variational form
/// @param[in] type the integral type
template <typename T, typename U>
void assemble_matrix(U mat_set, std::shared_ptr<dolfinx::fem::FunctionSpace> V,
                     const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>& bcs,
                     const std::span<const std::int32_t>& active_entities,
                     dolfinx_cuas::kernel_fn<T>& kernel, const std::span<const T> coeffs,
                     int cstride, const std::span<const T>& constants,
                     dolfinx::fem::IntegralType type)
{

  // Build dof marker (assuming same test and trial space)
  std::vector<std::int8_t> dof_marker;
  auto map = V->dofmap()->index_map;
  auto bs = V->dofmap()->index_map_bs();
  assert(map);
  std::int32_t dim = bs * (map->size_local() + map->num_ghosts());
  for (std::size_t k = 0; k < bcs.size(); ++k)
  {
    assert(bcs[k]);
    assert(bcs[k]->function_space());
    if (V->contains(*bcs[k]->function_space()))
    {
      dof_marker.resize(dim, false);
      bcs[k]->mark_dofs(dof_marker);
    }
  }

  // Assemble integral
  if (type == dolfinx::fem::IntegralType::cell)
    assemble_cells(mat_set, V, dof_marker, active_entities, kernel, coeffs, cstride, constants);
  else if (type == dolfinx::fem::IntegralType::exterior_facet)
    assemble_exterior_facets(mat_set, V, dof_marker, active_entities, kernel, coeffs, cstride,
                             constants);
  else
    throw std::runtime_error("Unsupported integral type");
};
} // namespace dolfinx_cuas
