#include "assemble.hpp"
#include "mass.h"
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx.h>
#include <xtensor/xio.hpp>

using namespace dolfinx;

int main(int argc, char* argv[])
{
  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_mpi(argc, argv);

  MPI_Comm mpi_comm{MPI_COMM_WORLD};

  auto mesh = std::make_shared<mesh::Mesh>(
      generation::BoxMesh::create(mpi_comm, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {1, 1, 1},
                                  mesh::CellType::tetrahedron, mesh::GhostMode::none));

  // TODO: Is it possible to create a function space with a basix element?
  // should we propose an interface for that, and avoid using ffcx for custom
  // assemblers?
  const std::shared_ptr<fem::FunctionSpace>& V
      = fem::create_functionspace(functionspace_form_mass_a, "u", mesh);

  auto A = assemble_matrix(V, Kernel::Mass);

  return 0;
}