#include "assemble.hpp"
#include "mass.h"
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx.h>
#include <dolfinx/fem/petsc.h>
#include <xtensor/xio.hpp>

using namespace dolfinx;

int main(int argc, char* argv[])
{
  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_petsc(argc, argv);

  MPI_Comm mpi_comm{MPI_COMM_WORLD};

  int mesh_string = 0;
  std::shared_ptr<mesh::Mesh> mesh;

  switch (mesh_string)
  {
  case 0:
    mesh = std::make_shared<mesh::Mesh>(
        generation::BoxMesh::create(mpi_comm, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {10, 10, 10},
                                    mesh::CellType::tetrahedron, mesh::GhostMode::none));
    break;
  case 1:
    mesh = std::make_shared<mesh::Mesh>(
        generation::RectangleMesh::create(mpi_comm, {{{0.0, 0.0}, {1.0, 1.0}}}, {500, 500},
                                          mesh::CellType::triangle, mesh::GhostMode::none));
    break;
  default:
    dolfinx::fem::CoordinateElement element(dolfinx::mesh::CellType::triangle, 1);
    xt::xtensor<double, 2> geom{{0.1, 0.}, {1, 0.}, {0., 1}};
    xt::xtensor<std::int64_t, 2> topo{{0, 1, 2}};

    auto [data, offset] = dolfinx::graph::create_adjacency_data(topo);
    auto cells = dolfinx::graph::AdjacencyList<std::int64_t>(data, offset);
    mesh = std::make_shared<mesh::Mesh>(
        create_mesh(mpi_comm, cells, element, geom, dolfinx::mesh::GhostMode::none));
  }

  const std::shared_ptr<fem::FunctionSpace>& V
      = fem::create_functionspace(functionspace_form_mass_a, "u", mesh);

  dolfinx::common::Timer t0("this");
  auto A_csr = assemble_stiffness_matrix<5>(V);
  t0.stop();

  // Define variational forms
  auto kappa = std::make_shared<fem::Constant<PetscScalar>>(1.0);
  auto a = std::make_shared<fem::Form<PetscScalar>>(
      fem::create_form<PetscScalar>(*form_mass_a, {V, V}, {}, {{"kappa", kappa}}, {}));

  dolfinx::common::Timer t1("dolfinx");
  la::PETScMatrix A = la::PETScMatrix(fem::create_matrix(*a), false);
  MatZeroEntries(A.mat());
  fem::assemble_matrix(la::PETScMatrix::set_block_fn(A.mat(), ADD_VALUES), *a, {});
  MatAssemblyBegin(A.mat(), MAT_FLUSH_ASSEMBLY);
  MatAssemblyEnd(A.mat(), MAT_FLUSH_ASSEMBLY);
  t1.stop();

  dolfinx::list_timings(mpi_comm, {dolfinx::TimingType::wall});

  return 0;
}