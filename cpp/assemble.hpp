#include "CsrMatrix.hpp"
#include "kernels.hpp"
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx.h>
#include <dolfinx/common/math.h>
#include <xtensor-blas/xlinalg.hpp>

using namespace dolfinx;

template <std::int32_t P, typename Matrix>
auto assemble_matrix(const std::shared_ptr<fem::FunctionSpace>& V, Matrix& A,
                     Kernel type = Kernel::Stiffness)
{
  // Get topology information
  const auto& mesh = V->mesh();
  constexpr std::int32_t tdim = 3; // mesh->topology().dim();

  // Get geometry information
  constexpr std::int32_t gdim = 3; // mesh->geometry().dim();
  const xt::xtensor<double, 2>& x_g = mesh->geometry().x();
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh->geometry().dofmap();

  // Finite Element
  std::string family = V->element()->family();
  std::string cell = mesh::to_string(V->mesh()->topology().cell_type());
  std::int32_t ncells = mesh->topology().index_map(tdim)->size_local();
  constexpr std::int32_t ndofs_cell = (P + 1) * (P + 2) * (P + 3) / 6;

  basix::FiniteElement element = basix::create_element(family, cell, P);
  const std::vector<std::uint32_t>& cell_info = mesh->topology().get_cell_permutation_info();

  // FIXME: Should be really constexpr/ should be known at compile time
  constexpr std::int32_t d = 4;

  xt::xtensor<double, 2> coordinate_dofs = xt::empty<double>({gdim, d});
  xt::xtensor<double, 2> Ae = xt::empty<double>({ndofs_cell, ndofs_cell});

  const auto& dofmap = V->dofmap();

  auto kernel = generate_kernel<P>(family, cell, type);

  std::vector<std::int32_t> dofs(ndofs_cell);

  double t_this = MPI_Wtime();
  for (std::int32_t c = 0; c < ncells; c++)
  {
    // Gather cell coordinates
    auto x_dofs = x_dofmap.links(c);
    for (int i = 0; i < d; ++i)
      for (int j = 0; j < gdim; ++j)
        coordinate_dofs(j, i) = x_g(x_dofs[i], j);

    Ae.fill(0);
    kernel(coordinate_dofs, Ae);

    // std::copy(cell_dofs.begin(), cell_dofs.end(), dofs.begin());
    std::iota(dofs.begin(), dofs.end(), 0);
    auto dofs_span = xtl::span<std::int32_t>(dofs);
    element.permute_dofs(dofs_span, cell_info[c]);

    A.add_values(Ae, dofs_span, c);
  }
  t_this = MPI_Wtime() - t_this;

  return t_this;
}