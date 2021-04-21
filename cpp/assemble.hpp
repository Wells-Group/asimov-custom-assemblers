#include "CsrMatrix.hpp"
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx.h>
#include <xtensor-sparse/xsparse_array.hpp>

using namespace dolfinx;

enum Kernel
{
  Mass,
  Stiffness
};

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

    J = xt::linalg::dot(coordinate_dofs, dphi0);
    double detJ = std::abs(xt::linalg::det(J));

    // Compute local matrix
    Ae.fill(0);
    for (std::size_t q = 0; q < weights.size(); q++)
      for (int i = 0; i < ndofs_cell; i++)
        for (int j = 0; j < ndofs_cell; j++)
          Ae(i, j) += weights[q] * phi(q, i) * phi(q, j) * detJ;

    auto dofs = dofmap->cell_dofs(c);
    A.add_values(Ae, dofs);
  }

  return A;
}