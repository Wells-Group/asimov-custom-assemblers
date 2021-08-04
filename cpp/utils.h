
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
#include <xtensor-blas/xlinalg.hpp>

namespace dolfinx_cuas
{

/// Given a mesh and an entity dimension, return the corresponding basix element of the entity
/// @param[in] mesh The mesh
/// @param[in] dim Dimension of entity
/// @return The basix element
basix::FiniteElement mesh_to_basix_element(std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
                                           const int dim);

/// Convert DOLFINx CellType to basix cell type
const basix::cell::type to_basix_celltype(dolfinx::mesh::CellType celltype);

// Compute quadrature points and weights on all facets of the reference cell
/// by pushing them forward from the reference facet.
/// @param[in] mesh The mesh
/// @param[in] quadrature_degree Degree of quadrature rule
std::pair<xt::xtensor<double, 3>, std::vector<double>>
create_reference_facet_qp(std::shared_ptr<const dolfinx::mesh::Mesh> mesh, int quadrature_degree);

/// Returns true if two PETSc matrices are element-wise equal within a tolerance.
/// @param[in] A PETSc Matrix to compare
/// @param[in] B PETSc Matrix to compare
bool allclose(Mat A, Mat B);

/// Prepare coefficients (dolfinx.Function's) for assembly with custom kernels
/// by packing them as a 2D array, where the ith row maps to the ith local cell.
/// For each row, the first N_0 columns correspond to the values of the 0th function space with N_0
/// dofs. If function space is blocked, the coefficients are ordered in XYZ XYZ ordering.
/// @param[in] coeffs The coefficients to pack
/// @param[out] c The packed coefficients
dolfinx::array2d<PetscScalar>
pack_coefficients(std::vector<std::shared_ptr<const dolfinx::fem::Function<PetscScalar>>> coeffs);
} // namespace dolfinx_cuas