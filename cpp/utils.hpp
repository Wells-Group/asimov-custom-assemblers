
// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINx_CUAS
//
// SPDX-License-Identifier:    MIT

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
                                 basix::element::lagrange_variant::gll_warped);
  }
  if (dim == tdim)
  {
    return basix::create_element(basix::element::family::P,
                                 dolfinx::mesh::cell_type_to_basix_type(dolfinx_cell), degree,
                                 basix::element::lagrange_variant::gll_warped);
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

  PetscScalar* A_array;
  MatSeqAIJGetArray(A, &A_array);
  auto _A = xt::adapt(A_array, A_info.nz_allocated, xt::no_ownership(),
                      std::vector<std::size_t>{std::size_t(A_info.nz_allocated)});

  PetscScalar* B_array;
  MatSeqAIJGetArray(B, &B_array);
  auto _B = xt::adapt(B_array, B_info.nz_allocated, xt::no_ownership(),
                      std::vector<std::size_t>{std::size_t(B_info.nz_allocated)});

  return xt::allclose(_A, _B);
}

/// Allocate memory for packed coefficients based on an input list of coefficients and integral
/// entities
/// @brief Allocate storage for coefficients of a pair (integral_type,
/// id) from a fem::Form form
/// @param[in] coefficients The coefficients
/// @param[in] active_entities The integral entities, flattened row-major.
/// @param[in] integral_type Type of integral
/// @return A storage container and the column stride
template <typename T>
std::pair<std::vector<T>, int> allocate_coefficient_storage(
    std::vector<std::shared_ptr<const dolfinx::fem::Function<T>>> coefficients,
    xtl::span<const std::int32_t> active_entities, dolfinx::fem::IntegralType integral_type)
{

  std::size_t num_entities = 0;
  if (!coefficients.empty())
  {
    switch (integral_type)
    {
    case dolfinx::fem::IntegralType::cell:
      num_entities = active_entities.size();
      break;
    case dolfinx::fem::IntegralType::exterior_facet:
      num_entities = active_entities.size() / 2;
      break;
    case dolfinx::fem::IntegralType::interior_facet:
      num_entities = active_entities.size() / 2;
      break;
    default:
      throw std::runtime_error("Could not allocate coefficient data. Integral type not supported.");
    }
  }
  // Coefficient offsets
  int cstride = 0;
  if (!coefficients.empty())
  {
    std::vector<int> coeffs_offsets{0};
    for (const auto& c : coefficients)
    {
      if (!c)
        throw std::runtime_error("Not all form coefficients have been set.");
      coeffs_offsets.push_back(coeffs_offsets.back()
                               + c->function_space()->element()->space_dimension());
    }
    cstride = coeffs_offsets.back();
  }

  return {std::vector<T>(num_entities * cstride), cstride};
}

/// Pack coefficients for a list of functions over a set of active entities
/// @param[in] coeffs The list of coefficients to pack
/// @param[in] active_entities The list of active entities to pack. The data has been flattened
/// row-major.
/// @param[in] integral_type The integral type associated with the active entities. This determines
/// the shape of the input entities (which has been flattened row-major). Cell integrals indicates
/// that active_entities is a list of cell indices, exterior_facet indicates a tuple (cell index,
/// local_facet_index), while interior_factet indicates a quadruplet (cell_0, local_facet_0, cell_1,
/// local_facet_1).
/// @returns A tuple (coeffs, stride) where coeffs is a 1D array containing the coefficients
/// packed for all the active entities, and stride is how many coeffs there are per entity.
template <typename T>
std::pair<std::vector<T>, int>
pack_coefficients(std::vector<std::shared_ptr<const dolfinx::fem::Function<T>>> coeffs,
                  xtl::span<const std::int32_t> active_entities,
                  dolfinx::fem::IntegralType integral_type)
{

  // NOTE: We could move coefficent data allocation outside packing to optimize performance.
  // Get coefficient storage
  std::pair<std::vector<T>, int> coeff_data
      = allocate_coefficient_storage(coeffs, active_entities, integral_type);
  auto c = std::get<0>(coeff_data);
  auto cstride = std::get<1>(coeff_data);
  // Get coefficient offset
  std::vector<int> offsets = {0};
  for (const auto& coeff : coeffs)
  {
    if (!coeff)
      throw std::runtime_error("Not all coefficients have been set.");
    offsets.push_back(offsets.back() + coeff->function_space()->element()->space_dimension());
  }

  // Get mesh
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = coeffs[0]->function_space()->mesh();
  assert(mesh);
  const int tdim = mesh->topology().dim();
  const std::int32_t num_cells = mesh->topology().index_map(tdim)->size_local()
                                 + mesh->topology().index_map(tdim)->num_ghosts();

  xtl::span<const std::uint32_t> cell_info = dolfinx::fem::impl::get_cell_orientation_info(coeffs);

  switch (integral_type)
  {
  case dolfinx::fem::IntegralType::cell:
  {
    auto fetch_cell = [](auto entity) { return entity.front(); };
    // Iterate over coefficients
    for (std::size_t coeff = 0; coeff < coeffs.size(); ++coeff)
    {
      dolfinx::fem::impl::pack_coefficient_entity(xtl::span(c), cstride, *coeffs[coeff], cell_info,
                                                  active_entities, 1, fetch_cell, offsets[coeff]);
    }
    break;
  }
  case dolfinx::fem::IntegralType::exterior_facet:
  {

    // Create lambda function fetching cell index from exterior facet entity
    auto fetch_cell = [](auto& entity) { return entity.front(); };

    // Iterate over coefficients
    for (std::size_t coeff = 0; coeff < coeffs.size(); ++coeff)
    {
      dolfinx::fem::impl::pack_coefficient_entity(xtl::span(c), cstride, *coeffs[coeff], cell_info,
                                                  active_entities, 2, fetch_cell, offsets[coeff]);
    }

    break;
  }
  case dolfinx::fem::IntegralType::interior_facet:
  {
    // Lambda functions to fetch cell index from interior facet entity
    auto fetch_cell0 = [](auto& entity) { return entity[0]; };
    auto fetch_cell1 = [](auto& entity) { return entity[2]; };

    // Iterate over coefficients
    for (std::size_t coeff = 0; coeff < coeffs.size(); ++coeff)
    {
      // Pack coefficient ['+']
      dolfinx::fem::impl::pack_coefficient_entity(xtl::span(c), 2 * cstride, *coeffs[coeff],
                                                  cell_info, active_entities, 4, fetch_cell0,
                                                  2 * offsets[coeff]);
      // Pack coefficient ['-']
      dolfinx::fem::impl::pack_coefficient_entity(xtl::span(c), 2 * cstride, *coeffs[coeff],
                                                  cell_info, active_entities, 4, fetch_cell1,
                                                  offsets[coeff] + offsets[coeff + 1]);
    }
    break;
  }
  default:
    throw std::runtime_error("Could not pack coefficient. Integral type not supported.");
  }

  return {std::move(c), cstride};
}

/// Compute the active entities in DOLFINx format for a given integral type over a set of entities
/// If the integral type is cell, return the input, if it is exterior facets, return a list of
/// pairs (cell, local_facet_index), and if it is interior facets, return a list of tuples
/// (cell_0, local_facet_index_0, cell_1, local_facet_index_1) for each entity.
/// @param[in] mesh The mesh
/// @param[in] entities List of mesh entities
/// @param[in] integral The type of integral
std::vector<std::int32_t> compute_active_entities(std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
                                                  xtl::span<const std::int32_t> entities,
                                                  dolfinx::fem::IntegralType integral)
{

  std::size_t entity_size;
  switch (integral)
  {
  case dolfinx::fem::IntegralType::cell:
  {
    std::vector<std::int32_t> active_entities(entities.size());
    std::transform(entities.begin(), entities.end(), active_entities.begin(),
                   [](std::int32_t cell) { return cell; });
    return active_entities;
  }
  case dolfinx::fem::IntegralType::exterior_facet:
  {
    std::vector<std::int32_t> active_entities(2 * entities.size());
    const dolfinx::mesh::Topology& topology = mesh->topology();
    int tdim = topology.dim();
    auto f_to_c = topology.connectivity(tdim - 1, tdim);
    assert(f_to_c);
    auto c_to_f = topology.connectivity(tdim, tdim - 1);
    assert(c_to_f);
    for (std::int32_t f = 0; f < entities.size(); f++)
    {
      assert(f_to_c->num_links(entities[f]) == 1);
      const std::int32_t cell = f_to_c->links(entities[f])[0];
      auto cell_facets = c_to_f->links(cell);

      auto facet_it = std::find(cell_facets.begin(), cell_facets.end(), entities[f]);
      assert(facet_it != cell_facets.end());
      active_entities[2 * f] = cell;
      active_entities[2 * f + 1] = std::distance(cell_facets.begin(), facet_it);
    }
    return active_entities;
  }
  case dolfinx::fem::IntegralType::interior_facet:
  {
    std::vector<std::int32_t> active_entities(4 * entities.size());
    const dolfinx::mesh::Topology& topology = mesh->topology();
    int tdim = topology.dim();
    auto f_to_c = topology.connectivity(tdim - 1, tdim);
    assert(f_to_c);
    auto c_to_f = topology.connectivity(tdim, tdim - 1);
    assert(c_to_f);
    std::array<std::pair<std::int32_t, int>, 2> interior_facets;
    for (std::int32_t f = 0; f < entities.size(); f++)
    {
      assert(f_to_c->num_links(entities[f]) == 2);
      auto cells = f_to_c->links(entities[f]);
      for (std::int32_t i = 0; i < 2; i++)
      {
        auto cell_facets = c_to_f->links(cells[i]);
        auto facet_it = std::find(cell_facets.begin(), cell_facets.end(), entities[f]);
        assert(facet_it != cell_facets.end());
        int local_f = std::distance(cell_facets.begin(), facet_it);
        interior_facets[i] = {cells[i], local_f};
      }
      active_entities[4 * f] = interior_facets[0].first;
      active_entities[4 * f + 1] = interior_facets[0].second;
      active_entities[4 * f + 2] = interior_facets[1].first;
      active_entities[4 * f + 3] = interior_facets[1].second;
    }
    return active_entities;
  }
  default:
    throw std::runtime_error("Unknown integral type");
  }
  return {};
}
} // namespace dolfinx_cuas