#include "lagrange.h"
#include "CooMatrix.hpp"
#include "assemble.hpp"
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx.h>

#ifndef ELEMENT_DEGREE
#define ELEMENT_DEGREE 1
#endif

using namespace dolfinx;
using insert_func_t = std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                                        const std::int32_t*, const double*)>;

int main(int argc, char* argv[])
{
  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_mpi(argc, argv);

  constexpr std::int32_t P = ELEMENT_DEGREE;

  MPI_Comm mpi_comm{MPI_COMM_WORLD};

  // std::shared_ptr<mesh::Mesh> mesh = std::make_shared<mesh::Mesh>(
  //     generation::BoxMesh::create(mpi_comm, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {1, 1, 1},
  //                                 mesh::CellType::tetrahedron, mesh::GhostMode::none));

  dolfinx::fem::CoordinateElement element(dolfinx::mesh::CellType::tetrahedron, 1);
  xt::xtensor<double, 2> geom{{0.1, 0., 0}, {1, 0., 0}, {0., 1, 0.}, {0., 0., 1.}};
  xt::xtensor<std::int64_t, 2> topo{{0, 1, 2, 3}};

  auto [data, offset] = dolfinx::graph::create_adjacency_data(topo);
  auto cells = dolfinx::graph::AdjacencyList<std::int64_t>(data, offset);
  std::shared_ptr<mesh::Mesh> mesh = std::make_shared<mesh::Mesh>(
      create_mesh(mpi_comm, cells, element, geom, dolfinx::mesh::GhostMode::none));

  mesh->topology().create_entity_permutations();

  const std::shared_ptr<fem::FunctionSpace>& V
      = fem::create_functionspace(functionspace_form_lagrange_a, "u", mesh);

  int tdim = mesh->topology().dim();
  const auto& topology = mesh->topology();
  int ncells = topology.index_map(tdim)->size_global();
  int ndofs_cell = V->element()->space_dimension();

  // Define variational forms
  auto a = std::make_shared<fem::Form<PetscScalar>>(fem::create_form<PetscScalar>(
      *form_lagrange_a, {V, V}, {},
      std::map<std::string, std::shared_ptr<const fem::Constant<double>>>(), {}));

  // create sparsity pattern and allocate data
  custom::la::CooMatrix<double, std::int32_t> B(ncells, ndofs_cell);

  std::int32_t cell = 0;
  insert_func_t insert_block = [&](std::int32_t nr, const std::int32_t* rows, const std::int32_t nc,
                                   const std::int32_t* cols, const double* data) {
    std::vector<std::size_t> shape({std::size_t(nr), std::size_t(nr)});
    xt::xtensor<double, 2> Ae = xt::adapt(data, nc * nr, xt::no_ownership(), shape);
    B.add_values(Ae, cell);
    cell++;
    return 0;
  };

  // Dolfinx Assemble
  double t_ffcx = MPI_Wtime();
  dolfinx::fem::assemble_matrix(insert_block, *a, {});
  t_ffcx = MPI_Wtime() - t_ffcx;
  std::cout << "\nffcx"
            << ", " << P << ", " << 1 << ", " << ncells << ", " << t_ffcx;

  // create sparsity pattern and allocate data
  custom::la::CooMatrix<double, std::int32_t> A(ncells, ndofs_cell);
  double t0 = assemble_matrix<P>(V, A, Kernel::Stiffness, Representation::Quadrature);
  bool check0 = xt::allclose(A.array(), B.array());
  std::cout << "\nquadrature"
            << ", " << P << ", " << check0 << ", " << ncells << ", " << t0;

  std::cout << "\n \n"
            << xt::view(A.array(), 0, xt::all(), xt::all())
                   - xt::view(B.array(), 0, xt::all(), xt::all())
            << "\n";

  // custom::la::CooMatrix<double, std::int32_t> C(ncells, ndofs_cell);
  // double t1 = assemble_matrix<P>(V, C, Kernel::Mass, Representation::Tensor);
  // bool check1 = xt::allclose(A.array(), B.array());
  // std::cout << "\ntensor"
  //           << ", " << P << ", " << check1 << ", " << ncells << ", " << t1;

  return 0;
}