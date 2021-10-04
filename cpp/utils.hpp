
// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINx_CUAS
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <basix/cell.h>
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/mesh/Mesh.h>

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
    // FIXME: This will not be correct for prism meshes, as we would need to create multiple basix
    // elements
    const dolfinx::mesh::CellType dolfinx_facet
        = dolfinx::mesh::cell_entity_type(dolfinx_cell, fdim, 0);
    return basix::create_element(basix::element::family::P,
                                 dolfinx::mesh::cell_type_to_basix_type(dolfinx_facet), degree,
                                 basix::element::lagrange_variant::equispaced);
  }
  if (dim == tdim)
  {
    return basix::create_element(basix::element::family::P,
                                 dolfinx::mesh::cell_type_to_basix_type(dolfinx_cell), degree,
                                 basix::element::lagrange_variant::equispaced);
  }
  else
    throw std::runtime_error("Does not support elements of edges and vertices");
}

/// Returns true if two PETSc matrices are element-wise equal within a tolerance.
/// @param[in] A PETSc Matrix to compare
/// @param[in] B PETSc Matrix to compare
bool allclose(Mat A, Mat B)
{
  MatInfo A_info;
  MatGetInfo(A, MAT_LOCAL, &A_info);

  MatInfo B_info;
  MatGetInfo(B, MAT_LOCAL, &B_info);

  if (B_info.nz_allocated != A_info.nz_allocated)
    return false;

  double* A_array;
  MatSeqAIJGetArray(A, &A_array);
  auto _A = xt::adapt(A_array, A_info.nz_allocated, xt::no_ownership(),
                      std::vector<std::size_t>{std::size_t(A_info.nz_allocated)});

  double* B_array;
  MatSeqAIJGetArray(B, &B_array);
  auto _B = xt::adapt(B_array, B_info.nz_allocated, xt::no_ownership(),
                      std::vector<std::size_t>{std::size_t(B_info.nz_allocated)});

  return xt::allclose(_A, _B);
}

/// Prepare coefficients (dolfinx.Function's) for assembly with custom kernels
/// by packing them as a 1D array, where the coefficients are packed cell-wise.
/// For each row, the first N_0 columns correspond to the values of the 0th function space with N_0
/// dofs. If function space is blocked, the coefficients are ordered in XYZ XYZ ordering.
/// @param[in] coeffs The coefficients to pack
/// @param[out] c The packed coefficients and the number of coeffs per cell
std::pair<std::vector<PetscScalar>, int>
pack_coefficients(std::vector<std::shared_ptr<const dolfinx::fem::Function<PetscScalar>>> coeffs)
{
  // Coefficient offsets
  std::vector<int> coeffs_offsets{0};
  for (const auto& c : coeffs)
  {
    if (!c)
      throw std::runtime_error("Not all form coefficients have been set.");
    coeffs_offsets.push_back(coeffs_offsets.back()
                             + c->function_space()->element()->space_dimension());
  }

  std::vector<const dolfinx::fem::DofMap*> dofmaps(coeffs.size());
  std::vector<const dolfinx::fem::FiniteElement*> elements(coeffs.size());
  std::vector<std::reference_wrapper<const std::vector<PetscScalar>>> v;
  v.reserve(coeffs.size());
  for (std::size_t i = 0; i < coeffs.size(); ++i)
  {
    elements[i] = coeffs[i]->function_space()->element().get();
    dofmaps[i] = coeffs[i]->function_space()->dofmap().get();
    v.push_back(coeffs[i]->x()->array());
  }

  // Get mesh
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = coeffs[0]->function_space()->mesh();
  assert(mesh);
  const int tdim = mesh->topology().dim();
  const std::int32_t num_cells = mesh->topology().index_map(tdim)->size_local()
                                 + mesh->topology().index_map(tdim)->num_ghosts();

  // Copy data into coefficient array
  std::vector<PetscScalar> c(num_cells * coeffs_offsets.back());
  const int cstride = coeffs_offsets.back();
  if (!coeffs.empty())
  {
    bool needs_dof_transformations = false;
    for (std::size_t coeff = 0; coeff < dofmaps.size(); ++coeff)
    {
      if (elements[coeff]->needs_dof_transformations())
      {
        needs_dof_transformations = true;
        mesh->topology_mutable().create_entity_permutations();
      }
    }

    // Iterate over coefficients
    xtl::span<const std::uint32_t> cell_info;
    if (needs_dof_transformations)
      cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
    for (std::size_t coeff = 0; coeff < dofmaps.size(); ++coeff)
    {
      const std::function<void(const xtl::span<PetscScalar>&, const xtl::span<const std::uint32_t>&,
                               std::int32_t, int)>
          transformation
          = elements[coeff]->get_dof_transformation_function<PetscScalar>(false, true);
      if (int bs = dofmaps[coeff]->bs(); bs == 1)
      {
        dolfinx::fem::impl::pack_coefficient<PetscScalar, 1>(
            xtl::span<PetscScalar>(c), cstride, v[coeff], cell_info, *dofmaps[coeff], num_cells,
            coeffs_offsets[coeff], elements[coeff]->space_dimension(), transformation);
      }
      else if (bs == 2)
      {
        dolfinx::fem::impl::pack_coefficient<PetscScalar, 2>(
            xtl::span<PetscScalar>(c), cstride, v[coeff], cell_info, *dofmaps[coeff], num_cells,
            coeffs_offsets[coeff], elements[coeff]->space_dimension(), transformation);
      }
      else if (bs == 3)
      {
        dolfinx::fem::impl::pack_coefficient<PetscScalar, 3>(
            xtl::span<PetscScalar>(c), cstride, v[coeff], cell_info, *dofmaps[coeff], num_cells,
            coeffs_offsets[coeff], elements[coeff]->space_dimension(), transformation);
      }
      else
      {
        dolfinx::fem::impl::pack_coefficient<PetscScalar>(
            xtl::span<PetscScalar>(c), cstride, v[coeff], cell_info, *dofmaps[coeff], num_cells,
            coeffs_offsets[coeff], elements[coeff]->space_dimension(), transformation);
      }
    }
  }

  return {std::move(c), cstride};
}

} // namespace dolfinx_cuas