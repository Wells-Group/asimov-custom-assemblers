#include "CsrMatrix.hpp"
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx.h>
#include <dolfinx/common/math.h>
#include <xtensor-blas/xlinalg.hpp>

using namespace dolfinx;

enum Kernel
{
  Mass,
  Stiffness
};

namespace dolfinx_cuas
{

auto assemble_matrix(const std::shared_ptr<fem::FunctionSpace>& V, Kernel kernel = Kernel::Mass)
{
  // create sparsity pattern and allocate data
  custom::la::CsrMatrix<double, std::int32_t> A
      = custom::la::create_csr_matrix<double, std::int32_t>(V);

  // Get topology information
  const auto& mesh = V->mesh();
  const std::int32_t tdim = mesh->topology().dim();

  // Get geometry information
  const std::int32_t gdim = mesh->geometry().dim();
  const xt::xtensor<double, 2>& x_g = mesh->geometry().x();
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh->geometry().dofmap();

  // Finite Element
  auto family = V->element()->family();
  auto cell = mesh::to_string(V->mesh()->topology().cell_type());
  int degree = 1;

  // Create quadrature rule
  // FIXME: Update quadrature degree extimation see python code
  int quad_degree = 2 * degree + 1;
  auto cell_type = basix::cell::str_to_type(cell);
  auto [points, weights] = basix::quadrature::make_quadrature("default", cell_type, quad_degree);

  // Create Finite element for test and trial functions
  basix::FiniteElement element = basix::create_element(family, cell, degree);
  xt::xtensor<double, 2> phi = xt::view(element.tabulate(0, points), 0, xt::all(), xt::all(), 0);

  // Get coordinate element from dolfinx
  basix::FiniteElement coordinate_element = basix::create_element(family, cell, 1);
  xt::xtensor<double, 4> coordinate_basis = element.tabulate(1, points);

  // FIXME: Add documentation about the dimensions of dphi0, transpose in relation to basix
  // definition for performance purposes
  xt::xtensor<double, 2> dphi0
      = xt::transpose(xt::view(coordinate_basis, xt::range(1, tdim + 1), 0, xt::all(), 0));

  std::int32_t ncells = mesh->topology().index_map(tdim)->size_local();
  std::int32_t ndofs_cell = element.dim();

  // FIXME: Should be really constexpr/ should be known at compile time
  const std::int32_t d = coordinate_element.dim();

  // NOTE: transposed in relation to dolfinx, so we avoid tranposing
  // it for multiplifcation
  //  N0  N1, ..., Nd
  // [x0, x1, ..., xd]
  // [y0, y1, ..., yd]
  // [z0, z2, ..., zd]
  xt::xtensor<double, 2> coordinate_dofs = xt::empty<double>({gdim, d});

  // Allocate Local data
  xt::xtensor<double, 2> J = xt::empty<double>({gdim, tdim});
  xt::xtensor<double, 2> Ae = xt::empty<double>({ndofs_cell, ndofs_cell});

  const auto& dofmap = V->dofmap();

  for (std::int32_t c = 0; c < ncells; c++)
  {
    // Gather cell coordinates
    auto x_dofs = x_dofmap.links(c);
    for (int i = 0; i < d; ++i)
      for (int j = 0; j < gdim; ++j)
        coordinate_dofs(j, i) = x_g(x_dofs[i], j);
    Ae.fill(0);

    J = xt::linalg::dot(coordinate_dofs, dphi0);
    double detJ = std::abs(xt::linalg::det(J));

    // Compute local matrix

    for (std::size_t q = 0; q < weights.size(); q++)
      for (int i = 0; i < ndofs_cell; i++)
        for (int j = 0; j < ndofs_cell; j++)
          Ae(i, j) += weights[q] * phi(q, i) * phi(q, j) * detJ;

    auto dofs = dofmap->cell_dofs(c);
    A.add_values(Ae, dofs);
  }

  return A;
}

template <std::int32_t Degree>
auto assemble_stiffness_matrix(const std::shared_ptr<fem::FunctionSpace>& V,
                               Kernel kernel = Kernel::Mass)
{
  // create sparsity pattern and allocate data
  custom::la::CsrMatrix<double, std::int32_t> A
      = custom::la::create_csr_matrix<double, std::int32_t>(V);

  // Get topology information
  const auto& mesh = V->mesh();
  constexpr std::int32_t tdim = 3; // mesh->topology().dim();

  // Get geometry information
  constexpr std::int32_t gdim = 3; // mesh->geometry().dim();
  const xt::xtensor<double, 2>& x_g = mesh->geometry().x();
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh->geometry().dofmap();

  // Finite Element
  auto family = V->element()->family();
  auto cell = mesh::to_string(V->mesh()->topology().cell_type());
  int degree = Degree;

  // Create quadrature rule
  // FIXME: Update quadrature degree extimation see python code
  int quad_degree = (degree - 1) + (degree - 1) + 1;
  auto cell_type = basix::cell::str_to_type(cell);
  auto [points, weights] = basix::quadrature::make_quadrature("default", cell_type, quad_degree);

  // Create Finite element for test and trial functions
  basix::FiniteElement element = basix::create_element(family, cell, degree);
  xt::xtensor<double, 4> tab_data = element.tabulate(1, points);
  xt::xtensor<double, 3> dphi = xt::view(tab_data, xt::range(1, tdim + 1), xt::all(), xt::all(), 0);

  // Get coordinate element from dolfinx
  basix::FiniteElement coordinate_element = basix::create_element("Lagrange", cell, 1);
  xt::xtensor<double, 4> coordinate_basis = coordinate_element.tabulate(1, points);

  // FIXME: Add documentation about the dimensions of dphi0, transpose in relation to basix
  // definition for performance purposes
  xt::xtensor<double, 2> dphi0_c
      = xt::transpose(xt::view(coordinate_basis, xt::range(1, tdim + 1), 0, xt::all(), 0));

  std::int32_t ncells = mesh->topology().index_map(tdim)->size_local();
  constexpr std::int32_t ndofs_cell
      = (Degree + 1) * (Degree + 2) * (Degree + 3) / 6; // element.dim();

  // FIXME: Should be really constexpr/ should be known at compile time
  constexpr std::int32_t d = 4;

  // Allocate Local data

  // NOTE: transposed in relation to dolfinx, so we avoid tranposing
  // it for multiplifcation
  //  N0  N1, ..., Nd
  // [x0, x1, ..., xd]
  // [y0, y1, ..., yd]
  // [z0, z2, ..., zd]
  xt::xtensor_fixed<double, xt::xshape<gdim, d>> coordinate_dofs = xt::empty<double>({gdim, d});
  xt::xtensor_fixed<double, xt::xshape<gdim, tdim>> J = xt::empty<double>({gdim, tdim});
  xt::xtensor_fixed<double, xt::xshape<tdim, gdim>> K = xt::empty<double>({tdim, gdim});
  xt::xtensor_fixed<double, xt::xshape<ndofs_cell, ndofs_cell>> Ae
      = xt::empty<double>({ndofs_cell, ndofs_cell});

  xt::xtensor_fixed<double, xt::xshape<tdim>> row_i;
  xt::xtensor_fixed<double, xt::xshape<tdim>> row_j;
  const auto& dofmap = V->dofmap();

  // transpose dphi
  xt::xtensor<double, 3> _dphi({dphi.shape(1), dphi.shape(2), dphi.shape(0)});
  for (std::int32_t k = 0; k < tdim; k++)
    for (std::size_t q = 0; q < weights.size(); q++)
      for (std::int32_t i = 0; i < ndofs_cell; i++)
        _dphi(q, i, k) = dphi(k, q, i);

  for (std::int32_t c = 0; c < ncells; c++)
  {
    // Gather cell coordinates
    auto x_dofs = x_dofmap.links(c);
    for (int i = 0; i < d; ++i)
      for (int j = 0; j < gdim; ++j)
        coordinate_dofs(j, i) = x_g(x_dofs[i], j);
    Ae.fill(0);

    // Compute local matrix
    J = xt::linalg::dot(coordinate_dofs, dphi0_c);
    dolfinx::math::inv(J, K);
    double detJ = std::abs(dolfinx::math::det(J));

    // Main loop
    for (std::size_t q = 0; q < weights.size(); q++)
    {
      for (std::int32_t i = 0; i < ndofs_cell; i++)
      {
        for (std::int32_t j = 0; j < ndofs_cell; j++)
        {
          double acc = 0;
          for (std::int32_t k = 0; k < gdim; k++)
            for (std::int32_t l = 0; l < tdim; l++)
              acc += K(k, l) * _dphi(q, i, k) * K(k, l) * _dphi(q, j, k);
          Ae(i, j) += weights[q] * acc * detJ;
        }
      }
    }

    auto dofs = dofmap->cell_dofs(c);
    A.add_values(Ae, dofs);
  }

  return A;
}

template <std::int32_t Degree>
auto assemble_nedelec_mass_matrix(const std::shared_ptr<fem::FunctionSpace>& V)
{
  // create sparsity pattern and allocate data
  custom::la::CsrMatrix<double, std::int32_t> A
      = custom::la::create_csr_matrix<double, std::int32_t>(V);

  // Get topology information
  const auto& mesh = V->mesh();
  constexpr std::int32_t tdim = 3; // mesh->topology().dim();

  // Get geometry information
  constexpr std::int32_t gdim = 3; // mesh->geometry().dim();
  const xt::xtensor<double, 2>& x_g = mesh->geometry().x();
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh->geometry().dofmap();

  // Finite Element
  auto family = V->element()->family();
  auto cell = mesh::to_string(V->mesh()->topology().cell_type());
  int degree = Degree;

  // Create quadrature rule
  // FIXME: Update quadrature degree extimation see python code
  int quad_degree = degree + degree + 1;
  auto cell_type = basix::cell::str_to_type(cell);
  auto [points, weights] = basix::quadrature::make_quadrature("default", cell_type, quad_degree);

  // Create Finite element for test and trial functions
  basix::FiniteElement element = basix::create_element(family, cell, degree);
  xt::xtensor<double, 4> tab_data = element.tabulate(0, points);
  xt::xtensor<double, 3> phi = xt::view(tab_data, 0, xt::all(), xt::all(), xt::all());
  std::int32_t value_size = phi.shape(2);

  // Get coordinate element from dolfinx
  basix::FiniteElement coordinate_element = basix::create_element("Lagrange", cell, 1);
  xt::xtensor<double, 4> coordinate_basis = coordinate_element.tabulate(1, points);

  // FIXME: Add documentation about the dimensions of dphi0, transpose in relation to basix
  // definition for performance purposes
  xt::xtensor<double, 2> dphi0_c
      = xt::transpose(xt::view(coordinate_basis, xt::range(1, tdim + 1), 0, xt::all(), 0));

  std::int32_t ncells = mesh->topology().index_map(tdim)->size_local();
  constexpr std::int32_t ndofs_cell = Degree * (Degree + 2) * (Degree + 3) / 2; // element.dim();

  // FIXME: Should be really constexpr/ should be known at compile time
  constexpr std::int32_t d = 4;

  // Allocate Local data

  // NOTE: transposed in relation to dolfinx, so we avoid tranposing
  // it for multiplifcation
  //  N0  N1, ..., Nd
  // [x0, x1, ..., xd]
  // [y0, y1, ..., yd]
  // [z0, z2, ..., zd]
  xt::xtensor_fixed<double, xt::xshape<gdim, d>> coordinate_dofs = xt::empty<double>({gdim, d});
  xt::xtensor_fixed<double, xt::xshape<gdim, tdim>> J = xt::empty<double>({gdim, tdim});
  xt::xtensor_fixed<double, xt::xshape<tdim, gdim>> K = xt::empty<double>({tdim, gdim});
  xt::xtensor_fixed<double, xt::xshape<ndofs_cell, ndofs_cell>> Ae
      = xt::empty<double>({ndofs_cell, ndofs_cell});
  const auto& dofmap = V->dofmap();

  xt::xtensor_fixed<double, xt::xshape<ndofs_cell, ndofs_cell>> Aref
      = xt::empty<double>({ndofs_cell, ndofs_cell});

  dolfinx::common::Timer t0("assemble mass nedelec");

  for (std::int32_t c = 0; c < ncells; c++)
  {
    // Gather cell coordinates
    auto x_dofs = x_dofmap.links(c);
    for (int i = 0; i < d; ++i)
      for (int j = 0; j < gdim; ++j)
        coordinate_dofs(j, i) = x_g(x_dofs[i], j);
    Ae.fill(0);

    // Compute local matrix
    J = xt::linalg::dot(coordinate_dofs, dphi0_c);
    double detJ = std::abs(dolfinx::math::det(J));

    // Main loop
    for (std::size_t q = 0; q < weights.size(); q++)
    {
      for (std::int32_t i = 0; i < ndofs_cell; i++)
      {
        for (std::int32_t j = 0; j < ndofs_cell; j++)
        {
          double acc = 0;
          for (std::int32_t k = 0; k < value_size; k++)
            acc += phi(q, i, k) * phi(q, j, k);
          Ae(i, j) += weights[q] * acc * detJ;
        }
      }
    }

    auto dofs = dofmap->cell_dofs(c);
    A.add_values(Ae, dofs);
  }

  t0.stop();

  return A;
}
} // namespace dolfinx_cuas