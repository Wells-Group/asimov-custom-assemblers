
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

/// Allocate memory for packed coefficients based on an input list of coefficients and a set of
/// integral entities
template <typename T>
std::pair<std::vector<T>, int> allocate_coefficient_storage(
    std::vector<std::shared_ptr<const dolfinx::fem::Function<T>>> coefficients,
    std::variant<tcb::span<const std::int32_t>, tcb::span<const std::pair<std::int32_t, int>>,
                 tcb::span<const std::tuple<std::int32_t, int, std::int32_t, int>>>
        active_entities)
{
  // Compute number of active entities
  std::size_t num_entities;
  std::visit(
      [&](auto&& entities)
      {
        using U = std::decay_t<decltype(entities)>;
        if constexpr (std::is_same_v<U, tcb::span<const std::int32_t>>)
          num_entities = entities.size();
        else if constexpr (std::is_same_v<U, tcb::span<const std::pair<std::int32_t, int>>>)
          num_entities = entities.size();
        else if constexpr (std::is_same_v<
                               U,
                               tcb::span<const std::tuple<std::int32_t, int, std::int32_t, int>>>)
          num_entities = 2 * entities.size();
        else
        {
          throw std::runtime_error(
              "Could not pack coefficient. Input entity type is not supported.");
        }
      },
      active_entities);

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
/// @param[in] active_entities The list of active entities to pack
/// @returns A tuple (coeffs, stride) where coeffs is a 1D array containing the coefficients
/// packed for all the active entities, and stride is how many coeffs there are per entity.
template <typename T>
std::pair<std::vector<T>, int> pack_coefficients(
    std::vector<std::shared_ptr<const dolfinx::fem::Function<T>>> coeffs,
    std::variant<tcb::span<const std::int32_t>, tcb::span<const std::pair<std::int32_t, int>>,
                 tcb::span<const std::tuple<std::int32_t, int, std::int32_t, int>>>
        active_entities)
{

  // NOTE: We could move coefficent data allocation outside packing to optimize performance.
  // Get coefficient storage
  std::pair<std::vector<T>, int> coeff_data = allocate_coefficient_storage(coeffs, active_entities);
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

  // Copy data into coefficient array
  if (!coeffs.empty())
  {
    xtl::span<const std::uint32_t> cell_info
        = dolfinx::fem::impl::get_cell_orientation_info(coeffs);

    // TODO see if this can be simplified with templating
    std::visit(
        [&](auto&& entities)
        {
          using U = std::decay_t<decltype(entities)>;
          if constexpr (std::is_same_v<U, tcb::span<const std::int32_t>>)
          {
            // Iterate over coefficients
            auto fetch_cell = [](auto entity) { return entity; };
            for (std::size_t coeff = 0; coeff < coeffs.size(); ++coeff)
            {
              dolfinx::fem::impl::pack_coefficient_entity(xtl::span<T>(c), cstride, *coeffs[coeff],
                                                          cell_info, entities, fetch_cell,
                                                          offsets[coeff]);
            }
          }
          else if constexpr (std::is_same_v<U, tcb::span<const std::pair<std::int32_t, int>>>)
          {
            // Create lambda function fetching cell index from exterior facet entity
            auto fetch_cell
                = [](const std::pair<std::int32_t, int>& entity) { return entity.first; };

            // Iterate over coefficients
            for (std::size_t coeff = 0; coeff < coeffs.size(); ++coeff)
            {
              dolfinx::fem::impl::pack_coefficient_entity(xtl::span<T>(c), cstride, *coeffs[coeff],
                                                          cell_info, entities, fetch_cell,
                                                          offsets[coeff]);
            }
          }
          else if constexpr (std::is_same_v<
                                 U,
                                 tcb::span<const std::tuple<std::int32_t, int, std::int32_t, int>>>)
          {

            // Lambda functions to fetch cell index from interior facet entity
            auto fetch_cell0 = [](const std::tuple<std::int32_t, int, std::int32_t, int>& entity)
            { return std::get<0>(entity); };
            auto fetch_cell1 = [](const std::tuple<std::int32_t, int, std::int32_t, int>& entity)
            { return std::get<2>(entity); };

            // Iterate over coefficients
            for (std::size_t coeff = 0; coeff < coeffs.size(); ++coeff)
            {
              // Pack coefficient ['+']
              dolfinx::fem::impl::pack_coefficient_entity(xtl::span<T>(c), 2 * cstride,
                                                          *coeffs[coeff], cell_info, entities,
                                                          fetch_cell0, 2 * offsets[coeff]);
              // Pack coefficient ['-']
              dolfinx::fem::impl::pack_coefficient_entity(
                  xtl::span<T>(c), 2 * cstride, *coeffs[coeff], cell_info, entities, fetch_cell1,
                  offsets[coeff] + offsets[coeff + 1]);
            }
          }
          else
          {
            throw std::runtime_error(
                "Could not pack coefficient. Input entity type is not supported.");
          }
        },
        active_entities);
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
std::variant<std::vector<std::int32_t>, std::vector<std::pair<std::int32_t, int>>,
             std::vector<std::tuple<std::int32_t, int, std::int32_t, int>>>
compute_active_entities(std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
                        tcb::span<const std::int32_t> entities, dolfinx::fem::IntegralType integral)
{

  // Determine variant type by integral
  std::variant<std::vector<std::int32_t>, std::vector<std::pair<std::int32_t, int>>,
               std::vector<std::tuple<std::int32_t, int, std::int32_t, int>>>
      active_entities;
  switch (integral)
  {
  case dolfinx::fem::IntegralType::cell:
    active_entities = std::vector<std::int32_t>(entities.size());
    break;
  case dolfinx::fem::IntegralType::exterior_facet:
    active_entities = std::vector<std::pair<std::int32_t, int>>(entities.size());
    break;
  case dolfinx::fem::IntegralType::interior_facet:
    active_entities
        = std::vector<std::tuple<std::int32_t, int, std::int32_t, int>>(entities.size());
    break;
  default:
    throw std::runtime_error("Unknown integral type");
  }

  std::visit(
      [&](auto&& output)
      {
        const dolfinx::mesh::Topology& topology = mesh->topology();
        using U = std::decay_t<decltype(output)>;
        if constexpr (std::is_same_v<U, std::vector<std::int32_t>>)
        {
          // Do nothing if cell integral
          std::transform(entities.begin(), entities.end(), output.begin(),
                         [](std::int32_t cell) { return cell; });
        }
        else if constexpr (std::is_same_v<U, std::vector<std::pair<std::int32_t, int>>>)
        {
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
            int local_f = std::distance(cell_facets.begin(), facet_it);
            output[f] = {cell, local_f};
          }
        }
        else if constexpr (std::is_same_v<
                               U, std::vector<std::tuple<std::int32_t, int, std::int32_t, int>>>)
        {
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
            output[f] = {interior_facets[0].first, interior_facets[0].second,
                         interior_facets[1].first, interior_facets[1].second};
          }
        }
      },
      active_entities);
  return active_entities;
}

} // namespace dolfinx_cuas