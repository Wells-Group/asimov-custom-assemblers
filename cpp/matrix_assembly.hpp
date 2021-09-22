// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINx_CUAS
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/assembler.h>
#include <functional>
#include <petscmat.h>
#include <xtl/xspan.hpp>

using kernel_fn = std::function<void(double*, const double*, const double*, const double*,
                                     const int*, const std::uint8_t*)>;

// Helper functions for assembly in DOLFINx
namespace
{

/// Assemble matrix over exterior facets
/// Provides easier interface to dolfinx::fem::impl::assemble_exterior_facets
/// @param[in] mat_set the function for setting the values in the matrix
/// @param[in] V the function space
/// @param[in] active_facets list of indices (local to process) of facets to be integrated over
/// @param[in] kernel the custom integration kernel
/// @param[in] coeffs coefficients used in the variational form
/// @param[in] constants used in the variational form
void assemble_exterior_facets(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t, const std::int32_t*,
                            const PetscScalar*)>& mat_set,
    std::shared_ptr<dolfinx::fem::FunctionSpace> V, const std::vector<bool>& bc,
    const xtl::span<const std::int32_t>& active_facets, kernel_fn& kernel,
    const dolfinx::array2d<PetscScalar>& coeffs, const xtl::span<const PetscScalar>& constants)
{
  // Extract mesh
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = V->mesh();

  // Extract function space data (assuming same test and trial space)
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = V->dofmap();
  const dolfinx::graph::AdjacencyList<std::int32_t>& dofs = dofmap->list();
  const int bs = dofmap->bs();
  std::shared_ptr<const dolfinx::fem::FiniteElement> element = V->element();
  const std::function<void(const xtl::span<double>&, const xtl::span<const std::uint32_t>&,
                           std::int32_t, int)>
      apply_dof_transformation = element->get_dof_transformation_function<double>();
  const std::function<void(const xtl::span<double>&, const xtl::span<const std::uint32_t>&,
                           std::int32_t, int)>
      apply_dof_transformation_to_transpose
      = element->get_dof_transformation_to_transpose_function<double>();

  // NOTE: Need to reconsider this when we get to jump integrals between disconnected interfaces
  const bool needs_transformation_data = element->needs_dof_transformations();

  // Get permutation data
  xtl::span<const std::uint32_t> cell_info;
  if (needs_transformation_data)
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
  }
  // FIXME: Need to reconsider facet permutations for jump integrals
  auto get_perm = [](std::size_t) { return 0; };

  // Create facet tuple: cell_index (local to process) and facet_index (local to cell)
  int tdim = mesh->topology().dim();
  auto f_to_c = mesh->topology().connectivity(tdim - 1, tdim);
  assert(f_to_c);
  auto c_to_f = mesh->topology().connectivity(tdim, tdim - 1);
  assert(c_to_f);
  std::vector<std::pair<std::int32_t, int>> facets;
  facets.reserve(active_facets.size());
  for (auto facet : active_facets)
  {
    auto cells = f_to_c->links(facet);
    assert(cells.size() == 1);
    auto cell_facets = c_to_f->links(cells[0]);
    const auto* it = std::find(cell_facets.data(), cell_facets.data() + cell_facets.size(), facet);
    assert(it != (cell_facets.data() + cell_facets.size()));
    const int local_facet = std::distance(cell_facets.data(), it);
    facets.push_back({cells[0], local_facet});
  }

  // Assemble using dolfinx
  dolfinx::fem::impl::assemble_exterior_facets<PetscScalar>(
      mat_set, *mesh, facets, apply_dof_transformation, dofs, bs,
      apply_dof_transformation_to_transpose, dofs, bs, bc, bc, kernel, coeffs, constants, cell_info,
      get_perm);
}

/// Assemble vector over cells
/// Provides easier interface to dolfinx::fem::impl::assemble_cells
/// @param[in] mat_set the function for setting the values in the matrix
/// @param[in] V the function space
/// @param[in] active_cells list of indices (local to process) of cells to be integrated over
/// @param[in] kernel the custom integration kernel
/// @param[in] coeffs coefficients used in the variational form
/// @param[in] constants used in the variational form
void assemble_cells(const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                                            const std::int32_t*, const PetscScalar*)>& mat_set,
                    std::shared_ptr<dolfinx::fem::FunctionSpace> V, const std::vector<bool>& bc,
                    const xtl::span<const std::int32_t>& active_cells, kernel_fn& kernel,
                    const dolfinx::array2d<PetscScalar>& coeffs,
                    const xtl::span<const PetscScalar>& constants)
{
  // Extract mesh
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = V->mesh();

  // Extract function space data
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = V->dofmap();
  const dolfinx::graph::AdjacencyList<std::int32_t>& dofs = dofmap->list();
  const int bs = dofmap->bs();
  std::shared_ptr<const dolfinx::fem::FiniteElement> element = V->element();
  const std::function<void(const xtl::span<double>&, const xtl::span<const std::uint32_t>&,
                           std::int32_t, int)>
      apply_dof_transformation = element->get_dof_transformation_function<double>();
  const std::function<void(const xtl::span<double>&, const xtl::span<const std::uint32_t>&,
                           std::int32_t, int)>
      apply_dof_transformation_to_transpose
      = element->get_dof_transformation_to_transpose_function<double>();

  // NOTE: Need to reconsider this when we get to jump integrals between disconnected interfaces
  const bool needs_transformation_data = element->needs_dof_transformations();

  // Get permutation data
  xtl::span<const std::uint32_t> cell_info;
  if (needs_transformation_data)
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
  }

  // Assemble using dolfinx
  dolfinx::fem::impl::assemble_cells<PetscScalar>(mat_set, mesh->geometry(), active_cells,
                                                  apply_dof_transformation, dofs, bs,
                                                  apply_dof_transformation_to_transpose, dofs, bs,
                                                  bc, bc, kernel, coeffs, constants, cell_info);
}
} // namespace

namespace dolfinx_cuas
{

/// Assemble matrix for given kernel function
/// @param[in] mat_set the function for setting the values in the matrix
/// @param[in] V the function space
/// @param[in] bcs List of Dirichlet boundary conditions
/// @param[in] active_entities list of indices (local to process) of entities to be integrated over
/// @param[in] kernel the custom integration kernel
/// @param[in] coefficients used in the variational form
/// @param[in] constants used in the variational form
/// @param[in] type the integral type
void assemble_matrix(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t, const std::int32_t*,
                            const PetscScalar*)>& mat_set,
    std::shared_ptr<dolfinx::fem::FunctionSpace> V,
    const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<PetscScalar>>>& bcs,
    const xtl::span<const std::int32_t>& active_entities, kernel_fn& kernel,
    const dolfinx::array2d<PetscScalar>& coeffs, const xtl::span<const PetscScalar>& constants,
    dolfinx::fem::IntegralType type)
{

  // Build dof marker (assuming same test and trial space)
  std::vector<bool> dof_marker;
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
    assemble_cells(mat_set, V, dof_marker, active_entities, kernel, coeffs, constants);
  else if (type == dolfinx::fem::IntegralType::exterior_facet)
    assemble_exterior_facets(mat_set, V, dof_marker, active_entities, kernel, coeffs, constants);
  else
    throw std::runtime_error("Unsupported integral type");
};
} // namespace dolfinx_cuas
