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

namespace dolfinx_cuas
{

void assemble_exterior_facets(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t, const std::int32_t*,
                            const PetscScalar*)>& mat_set,
    std::shared_ptr<const dolfinx::fem::Form<PetscScalar>> a,
    const xtl::span<const std::int32_t>& active_facets, kernel_fn& kernel)
{
  // Extract mesh
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = a->function_spaces().at(0)->mesh();

  // Extract function space data
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap0 = a->function_spaces().at(0)->dofmap();
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap1 = a->function_spaces().at(1)->dofmap();
  const dolfinx::graph::AdjacencyList<std::int32_t>& dofs0 = dofmap0->list();
  const int bs0 = dofmap0->bs();
  const dolfinx::graph::AdjacencyList<std::int32_t>& dofs1 = dofmap1->list();
  const int bs1 = dofmap1->bs();
  std::vector<bool> bc0;
  std::vector<bool> bc1;
  std::shared_ptr<const dolfinx::fem::FiniteElement> element0
      = a->function_spaces().at(0)->element();
  std::shared_ptr<const dolfinx::fem::FiniteElement> element1
      = a->function_spaces().at(1)->element();
  const std::function<void(const xtl::span<double>&, const xtl::span<const std::uint32_t>&,
                           std::int32_t, int)>
      apply_dof_transformation = element0->get_dof_transformation_function<double>();
  const std::function<void(const xtl::span<double>&, const xtl::span<const std::uint32_t>&,
                           std::int32_t, int)>
      apply_dof_transformation_to_transpose
      = element1->get_dof_transformation_to_transpose_function<double>();

  // Pack constants and coefficients
  const std::vector<double> constants = dolfinx::fem::pack_constants(*a);
  const dolfinx::array2d<double> coeffs = dolfinx::fem::pack_coefficients(*a);

  const bool needs_transformation_data = element0->needs_dof_transformations()
                                         or element1->needs_dof_transformations()
                                         or a->needs_facet_permutations();

  // Get permutation data
  xtl::span<const std::uint32_t> cell_info;
  if (needs_transformation_data)
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
  }

  const std::vector<std::uint8_t>& perms = mesh->topology().get_facet_permutations();

  // Assemble using dolfinx
  dolfinx::fem::impl::assemble_exterior_facets<PetscScalar>(
      mat_set, *mesh, active_facets, apply_dof_transformation, dofs0, bs0,
      apply_dof_transformation_to_transpose, dofs1, bs1, bc0, bc1, kernel, coeffs, constants,
      cell_info, perms);
}

void assemble_cells(const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                                            const std::int32_t*, const PetscScalar*)>& mat_set,
                    std::shared_ptr<const dolfinx::fem::Form<PetscScalar>> a,
                    const xtl::span<const std::int32_t>& active_cells, kernel_fn& kernel)
{
  // Extract mesh
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = a->function_spaces().at(0)->mesh();

  // Extract function space data
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap0 = a->function_spaces().at(0)->dofmap();
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap1 = a->function_spaces().at(1)->dofmap();
  const dolfinx::graph::AdjacencyList<std::int32_t>& dofs0 = dofmap0->list();
  const int bs0 = dofmap0->bs();
  const dolfinx::graph::AdjacencyList<std::int32_t>& dofs1 = dofmap1->list();
  const int bs1 = dofmap1->bs();
  std::vector<bool> bc0;
  std::vector<bool> bc1;
  std::shared_ptr<const dolfinx::fem::FiniteElement> element0
      = a->function_spaces().at(0)->element();
  std::shared_ptr<const dolfinx::fem::FiniteElement> element1
      = a->function_spaces().at(1)->element();
  const std::function<void(const xtl::span<double>&, const xtl::span<const std::uint32_t>&,
                           std::int32_t, int)>
      apply_dof_transformation = element0->get_dof_transformation_function<double>();
  const std::function<void(const xtl::span<double>&, const xtl::span<const std::uint32_t>&,
                           std::int32_t, int)>
      apply_dof_transformation_to_transpose
      = element1->get_dof_transformation_to_transpose_function<double>();

  // Pack constants and coefficients
  const std::vector<double> constants = dolfinx::fem::pack_constants(*a);
  const dolfinx::array2d<double> coeffs = dolfinx::fem::pack_coefficients(*a);

  const bool needs_transformation_data = element0->needs_dof_transformations()
                                         or element1->needs_dof_transformations()
                                         or a->needs_facet_permutations();

  // Get permutation data
  xtl::span<const std::uint32_t> cell_info;
  if (needs_transformation_data)
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
  }

  // Assemble using dolfinx
  dolfinx::fem::impl::assemble_cells<PetscScalar>(mat_set, mesh->geometry(), active_cells,
                                                  apply_dof_transformation, dofs0, bs0,
                                                  apply_dof_transformation_to_transpose, dofs1, bs1,
                                                  bc0, bc1, kernel, coeffs, constants, cell_info);
}

} // namespace dolfinx_cuas