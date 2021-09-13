
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
// Helper function to get a std::vector of (cell, local_facet) pairs
// corresponding to a given facet index.
// @param[in] f Facet index
// @param[in] f_to_c Facet to cell connectivity
// @param[in] c_to_f Cell to facet connectivity
// @return Vector of (cell, local_facet) pairs
template <int num_cells>
static std::array<std::pair<std::int32_t, int>, num_cells>
get_cell_local_facet_pairs(std::int32_t f, const xtl::span<const std::int32_t>& cells,
                           const dolfinx::graph::AdjacencyList<std::int32_t>& c_to_f)
{
  // Loop over cells sharing facet
  assert(cells.size() == num_cells);
  std::array<std::pair<std::int32_t, int>, num_cells> cell_local_facet_pairs;
  for (int c = 0; c < num_cells; ++c)
  {
    // Get local index of facet with respect to the cell
    std::int32_t cell = cells[c];
    auto cell_facets = c_to_f.links(cell);
    auto facet_it = std::find(cell_facets.begin(), cell_facets.end(), f);
    assert(facet_it != cell_facets.end());
    int local_f = std::distance(cell_facets.begin(), facet_it);
    cell_local_facet_pairs[c] = {cell, local_f};
  }

  return cell_local_facet_pairs;
}
/// Convert DOLFINx CellType to basix cell type
const basix::cell::type to_basix_celltype(dolfinx::mesh::CellType celltype)
{
  return basix::cell::str_to_type(dolfinx::mesh::to_string(celltype));
}

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
    // FIXME: assuming all facets have the same cell type
    const dolfinx::mesh::CellType dolfinx_facet
        = dolfinx::mesh::cell_entity_type(dolfinx_cell, fdim, 0);
    return basix::create_element(basix::element::family::P, to_basix_celltype(dolfinx_facet),
                                 degree, basix::lattice::type::equispaced);
  }
  if (dim == tdim)
  {
    return basix::create_element(basix::element::family::P, to_basix_celltype(dolfinx_cell), degree,
                                 basix::lattice::type::equispaced);
  }
  else
    throw std::runtime_error("Does not support elements of edges and vertices");
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
/// by packing them as a 2D array, where the ith row maps to the ith local cell.
/// For each row, the first N_0 columns correspond to the values of the 0th function space with N_0
/// dofs. If function space is blocked, the coefficients are ordered in XYZ XYZ ordering.
/// @param[in] coeffs The coefficients to pack
/// @param[out] c The packed coefficients
dolfinx::array2d<PetscScalar>
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
  dolfinx::array2d<PetscScalar> c(num_cells, coeffs_offsets.back());
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
            c, v[coeff], cell_info, *dofmaps[coeff], num_cells, coeffs_offsets[coeff],
            elements[coeff]->space_dimension(), transformation);
      }
      else if (bs == 2)
      {
        dolfinx::fem::impl::pack_coefficient<PetscScalar, 2>(
            c, v[coeff], cell_info, *dofmaps[coeff], num_cells, coeffs_offsets[coeff],
            elements[coeff]->space_dimension(), transformation);
      }
      else if (bs == 3)
      {
        dolfinx::fem::impl::pack_coefficient<PetscScalar, 3>(
            c, v[coeff], cell_info, *dofmaps[coeff], num_cells, coeffs_offsets[coeff],
            elements[coeff]->space_dimension(), transformation);
      }
      else
      {
        dolfinx::fem::impl::pack_coefficient<PetscScalar>(
            c, v[coeff], cell_info, *dofmaps[coeff], num_cells, coeffs_offsets[coeff],
            elements[coeff]->space_dimension(), transformation);
      }
    }
  }

  return c;
}

/// Prepare a coefficient (dolfinx::fem::Function) for assembly with custom kernels
/// by packing them as a 2D array, where the ith row maps to the ith local cell.
/// There are num_q_points*block_size*value_size columns, where
/// column[q * (block_size * value_size) + k + c] = sum_i c^i[k] * phi^i(x_q)[c]
/// where c^i[k] is the ith coefficient's kth vector component, phi^i(x_q)[c] is the ith basis
/// function's c-th value compoenent at the quadrature point x_q.
/// @param[in] coeff The coefficient to pack
/// @param[out] c The packed coefficients
dolfinx::array2d<PetscScalar>
pack_coefficient_quadrature(std::shared_ptr<const dolfinx::fem::Function<PetscScalar>> coeff,
                            const int q)
{
  const dolfinx::fem::DofMap* dofmap = coeff->function_space()->dofmap().get();
  const dolfinx::fem::FiniteElement* element = coeff->function_space()->element().get();
  const std::vector<double>& data = coeff->x()->array();

  // Get mesh
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = coeff->function_space()->mesh();
  assert(mesh);
  const std::size_t tdim = mesh->topology().dim();
  const std::size_t gdim = mesh->geometry().dim();
  const std::int32_t num_cells = mesh->topology().index_map(tdim)->size_local()
                                 + mesh->topology().index_map(tdim)->num_ghosts();

  // Get dof transformations
  const bool needs_dof_transformations = element->needs_dof_transformations();
  xtl::span<const std::uint32_t> cell_info;
  if (needs_dof_transformations)
  {
    cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
    mesh->topology_mutable().create_entity_permutations();
  }
  const std::function<void(const xtl::span<PetscScalar>&, const xtl::span<const std::uint32_t>&,
                           std::int32_t, int)>
      transformation = element->get_dof_transformation_function<PetscScalar>();

  // Tabulate element at quadrature points
  // NOTE: Assuming no derivatives for now, should be reconsidered later
  const std::string cell_type = dolfinx::mesh::to_string(mesh->topology().cell_type());
  const std::size_t num_dofs = element->space_dimension();
  const std::size_t bs = dofmap->bs();
  const std::size_t vs = element->reference_value_size() / element->block_size();

  // Tabulate function at quadrature points
  auto [points, weights]
      = basix::quadrature::make_quadrature("default", basix::cell::str_to_type(cell_type), q);
  const std::size_t num_points = weights.size();
  xt::xtensor<double, 4> coeff_basis({1, num_points, num_dofs, vs});
  element->tabulate(coeff_basis, points, 0);
  dolfinx::array2d<PetscScalar> c(num_cells, vs * bs * num_points, 0);
  auto basis_reference_values = xt::view(coeff_basis, 0, xt::all(), xt::all(), xt::all());

  if (needs_dof_transformations)
  {
    // Prepare basis function data structures
    xt::xtensor<double, 3> basis_values({num_points, num_dofs / bs, vs});
    xt::xtensor<double, 3> cell_basis_values({num_points, num_dofs / bs, vs});

    // Get geometry data
    const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap = mesh->geometry().dofmap();

    // FIXME: Add proper interface for num coordinate dofs
    const std::size_t num_dofs_g = x_dofmap.num_links(0);
    const xt::xtensor<double, 2>& x_g = mesh->geometry().x();

    // Prepare geometry data structures
    xt::xtensor<double, 2> X({num_points, tdim});
    xt::xtensor<double, 3> J = xt::zeros<double>({num_points, gdim, tdim});
    xt::xtensor<double, 3> K = xt::zeros<double>({num_points, tdim, gdim});
    xt::xtensor<double, 1> detJ = xt::zeros<double>({num_points});
    xt::xtensor<double, 2> coordinate_dofs = xt::zeros<double>({num_dofs_g, gdim});

    // Get coordinate map
    const dolfinx::fem::CoordinateElement& cmap = mesh->geometry().cmap();

    // Compute first derivative of basis function of coordinate map
    xt::xtensor<double, 4> cmap_basis_functions = cmap.tabulate(1, points);
    xt::xtensor<double, 4> dphi_c = xt::view(cmap_basis_functions, xt::xrange(1, int(tdim) + 1),
                                             xt::all(), xt::all(), xt::all());

    for (std::int32_t cell = 0; cell < num_cells; ++cell)
    {

      // NOTE Add two separate loops here, one for and one without dof transforms

      // Get cell geometry (coordinate dofs)
      auto x_dofs = x_dofmap.links(cell);
      for (std::size_t i = 0; i < num_dofs_g; ++i)
        for (std::size_t j = 0; j < gdim; ++j)
          coordinate_dofs(i, j) = x_g(x_dofs[i], j);
      cmap.compute_jacobian(dphi_c, coordinate_dofs, J);
      cmap.compute_jacobian_inverse(J, K);
      cmap.compute_jacobian_determinant(J, detJ);

      // Permute the reference values to account for the cell's orientation
      cell_basis_values = basis_reference_values;
      for (std::size_t q = 0; q < num_points; ++q)
      {
        transformation(
            xtl::span(cell_basis_values.data() + q * num_dofs / bs * vs, num_dofs / bs * vs),
            cell_info, cell, vs);
      }
      // Push basis forward to physical element
      element->transform_reference_basis(basis_values, cell_basis_values, J, detJ, K);

      // Sum up quadrature contributions
      auto cell_coeff = c.row(cell);
      auto dofs = dofmap->cell_dofs(cell);
      for (std::size_t i = 0; i < dofs.size(); ++i)
      {
        const int pos_v = bs * dofs[i];

        for (int q = 0; q < num_points; ++q)
          for (int k = 0; k < bs; ++k)
            for (int c = 0; c < vs; c++)
              cell_coeff[q * (bs * vs) + k + c] += basis_values(q, i, c) * data[pos_v + k];
      }
    }
  }
  else
  {
    for (std::int32_t cell = 0; cell < num_cells; ++cell)
    {

      // Sum up quadrature contributions
      auto cell_coeff = c.row(cell);
      auto dofs = dofmap->cell_dofs(cell);
      for (std::size_t i = 0; i < dofs.size(); ++i)
      {
        const int pos_v = bs * dofs[i];

        for (int q = 0; q < num_points; ++q)
          for (int k = 0; k < bs; ++k)
            for (int c = 0; c < vs; c++)
              cell_coeff[q * (bs * vs) + k + c]
                  += basis_reference_values(q, i, c) * data[pos_v + k];
      }
    }
  }
  return c;
}

/// Prepare a coefficient (dolfinx::fem::Function) for assembly with custom kernels
/// by packing them as a 2D array, where the ith row maps to the ith facet int active_facets.
/// There are num_q_points*block_size*value_size columns, where
/// column[q * (block_size * value_size) + k + c] = sum_i c^i[k] * phi^i(x_q)[c]
/// where c^i[k] is the ith coefficient's kth vector component, phi^i(x_q)[c] is the ith basis
/// function's c-th value compoenent at the quadrature point x_q.
/// @param[in] coeff The coefficient to pack
/// @param[in] active_facets List of active facets
/// @param[in] q the quadrature degree
/// @param[out] c The packed coefficients
dolfinx::array2d<PetscScalar>
pack_coefficient_facet(std::shared_ptr<const dolfinx::fem::Function<PetscScalar>> coeff, int q,
                       const xtl::span<const std::int32_t>& active_facets)
{
  const dolfinx::fem::DofMap* dofmap = coeff->function_space()->dofmap().get();
  const dolfinx::fem::FiniteElement* element = coeff->function_space()->element().get();
  const std::vector<double>& data = coeff->x()->array();

  // Get mesh
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = coeff->function_space()->mesh();
  assert(mesh);
  const std::size_t tdim = mesh->topology().dim();
  const std::size_t gdim = mesh->geometry().dim();
  const std::size_t fdim = tdim - 1;
  const std::int32_t num_facets = active_facets.size();

  // Connectivity to evaluate at quadrature points
  // FIXME: Move create_connectivity out of this function and call before calling the function...
  mesh->topology_mutable().create_connectivity(fdim, tdim);
  auto f_to_c = mesh->topology().connectivity(fdim, tdim);
  mesh->topology_mutable().create_connectivity(tdim, fdim);
  auto c_to_f = mesh->topology().connectivity(tdim, fdim);

  // Get dof transformations
  const bool needs_dof_transformations = element->needs_dof_transformations();
  xtl::span<const std::uint32_t> cell_info;
  if (needs_dof_transformations)
  {
    cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
    mesh->topology_mutable().create_entity_permutations();
  }
  const std::function<void(const xtl::span<PetscScalar>&, const xtl::span<const std::uint32_t>&,
                           std::int32_t, int)>
      transformation = element->get_dof_transformation_function<PetscScalar>();

  // Tabulate element at quadrature points
  // NOTE: Assuming no derivatives for now, should be reconsidered later
  const std::string cell_type = dolfinx::mesh::to_string(mesh->topology().cell_type());
  const std::size_t num_dofs = element->space_dimension();
  const std::size_t bs = dofmap->bs();
  const std::size_t vs = element->reference_value_size() / element->block_size();

  // Tabulate function at quadrature points
  auto [points, weights] = create_reference_facet_qp(mesh, q);
  const std::size_t num_points = weights.size();
  const std::size_t num_local_facets = points.shape(0);
  xt::xtensor<double, 4> coeff_basis({1, num_points, num_dofs / bs, vs});
  xt::xtensor<double, 4> basis_reference_values({num_local_facets, num_points, num_dofs / bs, vs});

  for (int i = 0; i < num_local_facets; i++)
  {
    auto q_facet = xt::view(points, i, xt::all(), xt::all());
    element->tabulate(coeff_basis, q_facet, 0);
    auto basis_ref = xt::view(basis_reference_values, i, xt::all(), xt::all(), xt::all());
    basis_ref = xt::view(coeff_basis, 0, xt::all(), xt::all(), xt::all());
  }

  dolfinx::array2d<PetscScalar> c(num_facets, vs * bs * num_points, 0);

  if (needs_dof_transformations)
  {
    // Prepare basis function data structures
    xt::xtensor<double, 3> basis_values({num_points, num_dofs / bs, vs});
    xt::xtensor<double, 3> cell_basis_values({num_points, num_dofs / bs, vs});

    // Get geometry data
    const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap = mesh->geometry().dofmap();

    // FIXME: Add proper interface for num coordinate dofs
    const std::size_t num_dofs_g = x_dofmap.num_links(0);
    const xt::xtensor<double, 2>& x_g = mesh->geometry().x();

    // Prepare geometry data structures
    xt::xtensor<double, 2> X({num_points, tdim});
    xt::xtensor<double, 3> J = xt::zeros<double>({num_points, gdim, tdim});
    xt::xtensor<double, 3> K = xt::zeros<double>({num_points, tdim, gdim});
    xt::xtensor<double, 1> detJ = xt::zeros<double>({num_points});
    xt::xtensor<double, 2> coordinate_dofs = xt::zeros<double>({num_dofs_g, gdim});

    // Get coordinate map
    const dolfinx::fem::CoordinateElement& cmap = mesh->geometry().cmap();

    xt::xtensor<double, 5> dphi_c({num_local_facets, int(tdim), num_points, num_dofs_g / bs, 1});
    for (int i = 0; i < num_local_facets; i++)
    {
      auto q_facet = xt::view(points, i, xt::all(), xt::all());
      xt::xtensor<double, 4> cmap_basis_functions = cmap.tabulate(1, q_facet);
      auto dphi_ci = xt::view(dphi_c, i, xt::all(), xt::all(), xt::all(), xt::all());
      dphi_ci = xt::view(cmap_basis_functions, xt::xrange(1, int(tdim) + 1), xt::all(), xt::all(),
                         xt::all());
    }

    for (int facet = 0; facet < num_facets; facet++)
    {

      // NOTE Add two separate loops here, one for and one without dof transforms

      // FIXME: Assuming exterior facets
      // get cell/local facet index
      int global_facet = active_facets[facet]; // extract facet
      auto cells = f_to_c->links(global_facet);
      // since the facet is on the boundary it should only link to one cell
      assert(cells.size() == 1);
      auto cell = cells[0]; // extract cell

      // find local index of facet
      auto cell_facets = c_to_f->links(cell);
      auto local_facet = std::find(cell_facets.begin(), cell_facets.end(), global_facet);
      const std::int32_t local_index = std::distance(cell_facets.data(), local_facet);
      // Get cell geometry (coordinate dofs)
      auto x_dofs = x_dofmap.links(cell);

      for (std::size_t i = 0; i < num_dofs_g; ++i)
        for (std::size_t j = 0; j < gdim; ++j)
          coordinate_dofs(i, j) = x_g(x_dofs[i], j);

      auto dphi_ci = xt::view(dphi_c, local_index, xt::all(), xt::all(), xt::all(), xt::all());

      cmap.compute_jacobian(dphi_ci, coordinate_dofs, J);
      cmap.compute_jacobian_inverse(J, K);
      cmap.compute_jacobian_determinant(J, detJ);

      // Permute the reference values to account for the cell's orientation
      cell_basis_values
          = xt::view(basis_reference_values, local_index, xt::all(), xt::all(), xt::all());
      for (std::size_t q = 0; q < num_points; ++q)
      {
        transformation(
            xtl::span(cell_basis_values.data() + q * num_dofs / bs * vs, num_dofs / bs * vs),
            cell_info, cell, vs);
      }
      // Push basis forward to physical element
      element->transform_reference_basis(basis_values, cell_basis_values, J, detJ, K);

      // Sum up quadrature contributions
      auto facet_coeff = c.row(facet);
      auto dofs = dofmap->cell_dofs(cell);
      for (std::size_t i = 0; i < dofs.size(); ++i)
      {
        const int pos_v = bs * dofs[i];

        for (int q = 0; q < num_points; ++q)
          for (int k = 0; k < bs; ++k)
            for (int c = 0; c < vs; c++)
              facet_coeff[q * (bs * vs) + k + c] += basis_values(q, i, c) * data[pos_v + k];
      }
    }
  }
  else
  {

    for (int facet = 0; facet < num_facets; facet++)
    { // Sum up quadrature contributions
      // FIXME: Assuming exterior facets
      // get cell/local facet index
      int global_facet = active_facets[facet]; // extract facet
      auto cells = f_to_c->links(global_facet);
      // since the facet is on the boundary it should only link to one cell
      assert(cells.size() == 1);
      auto cell = cells[0]; // extract cell

      // find local index of facet
      auto cell_facets = c_to_f->links(cell);
      auto local_facet = std::find(cell_facets.begin(), cell_facets.end(), global_facet);
      const std::int32_t local_index = std::distance(cell_facets.data(), local_facet);

      auto facet_coeff = c.row(facet);
      auto dofs = dofmap->cell_dofs(cell);
      for (std::size_t i = 0; i < dofs.size(); ++i)
      {
        const int pos_v = bs * dofs[i];

        for (int q = 0; q < num_points; ++q)
          for (int k = 0; k < bs; ++k)
            for (int l = 0; l < vs; l++)
            {
              facet_coeff[q * (bs * vs) + k + l]
                  += basis_reference_values(local_index, q, i, l) * data[pos_v + k];
            }
      }
    }
  }
  return c;
}

} // namespace dolfinx_cuas