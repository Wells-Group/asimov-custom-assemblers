// Copyright (C) 2021 Sarah Roggendorf
//
// This file is part of DOLFINx_CUAS
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "utils.hpp"
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/assembler.h>
#include <functional>
#include <xtl/xspan.hpp>

using kernel_fn = std::function<void(double*, const double*, const double*, const double*,
                                     const int*, const std::uint8_t*)>;

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
/// @param[in] constants used in the variational form
void assemble_exterior_facets(xtl::span<PetscScalar> b,
                              std::shared_ptr<dolfinx::fem::FunctionSpace> V,
                              const xtl::span<const std::int32_t>& active_facets, kernel_fn& kernel,
                              const dolfinx::array2d<PetscScalar>& coeffs,
                              const xtl::span<const PetscScalar>& constants)
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

  // NOTE: Need to reconsider this when we get to jump integrals between disconnected interfaces
  const bool needs_transformation_data = element->needs_dof_transformations();

  // Get permutation data
  xtl::span<const std::uint32_t> cell_info;
  if (needs_transformation_data)
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
  }

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
  dolfinx::fem::impl::assemble_exterior_facets<PetscScalar>(apply_dof_transformation, b, *mesh,
                                                            facets, dofs, bs, kernel, constants,
                                                            coeffs, cell_info, get_perm);
}

/// Assemble vector over cells
/// Provides easier interface to dolfinx::fem::impl::assemble_cells
/// @param[in,out] b the vector to be assembled
/// @param[in] V the function space
/// @param[in] active_cells list of indices (local to process) of cells to be integrated over
/// @param[in] kernel the custom integration kernel
/// @param[in] coeffs coefficients used in the variational form
/// @param[in] constants used in the variational form
void assemble_cells(xtl::span<PetscScalar> b, std::shared_ptr<dolfinx::fem::FunctionSpace> V,
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

  // NOTE: Need to reconsider this when we get to jump integrals between disconnected interfaces
  const bool needs_transformation_data = element->needs_dof_transformations();

  // Get permutation data
  xtl::span<const std::uint32_t> cell_info;
  if (needs_transformation_data)
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
  }

  dolfinx::fem::impl::assemble_cells<PetscScalar>(apply_dof_transformation, b, mesh->geometry(),
                                                  active_cells, dofs, bs, kernel, constants, coeffs,
                                                  cell_info);
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
/// @param[in] constants used in the variational form
/// @param[in] type the integral type
void assemble_vector(xtl::span<PetscScalar> b, std::shared_ptr<dolfinx::fem::FunctionSpace> V,
                     const xtl::span<const std::int32_t>& active_entities, kernel_fn& kernel,
                     const dolfinx::array2d<PetscScalar>& coeffs,
                     const xtl::span<const PetscScalar>& constants, dolfinx::fem::IntegralType type)
{

  // Assemble integral
  if (type == dolfinx::fem::IntegralType::cell)
    assemble_cells(b, V, active_entities, kernel, coeffs, constants);
  else if (type == dolfinx::fem::IntegralType::exterior_facet)
    assemble_exterior_facets(b, V, active_entities, kernel, coeffs, constants);
  else
    throw std::runtime_error("Unsupported integral type");
};

} // namespace dolfinx_cuas
