#include "problem.h"
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx_cuas/kernels.hpp>
#include <xtensor/xio.hpp>

using namespace dolfinx;

int main(int argc, char* argv[])
{
  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_petsc(argc, argv);

  MPI_Comm mpi_comm{MPI_COMM_WORLD};

  std::shared_ptr<mesh::Mesh> mesh = std::make_shared<mesh::Mesh>(
      generation::BoxMesh::create(mpi_comm, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {50, 50, 50},
                                  mesh::CellType::tetrahedron, mesh::GhostMode::none));

  mesh->topology().create_entity_permutations();

  const std::shared_ptr<fem::FunctionSpace>& V
      = fem::create_functionspace(functionspace_form_problem_a, "u", mesh);

  // Define variational forms
  auto kappa = std::make_shared<fem::Constant<PetscScalar>>(1.0);
  auto a = std::make_shared<fem::Form<PetscScalar>>(
      fem::create_form<PetscScalar>(*form_problem_a, {V, V}, {}, {{"kappa", kappa}}, {}));

  // Define active cells
  const std::int32_t tdim = mesh->topology().dim();
  const std::int32_t ncells = mesh->topology().index_map(tdim)->size_local();
  xt::xarray<std::int32_t> active_cells = xt::arange<std::int32_t>(0, ncells);

  // Extract function space data
  std::shared_ptr<const fem::DofMap> dofmap0 = a->function_spaces().at(0)->dofmap();
  std::shared_ptr<const fem::DofMap> dofmap1 = a->function_spaces().at(1)->dofmap();
  const graph::AdjacencyList<std::int32_t>& dofs0 = dofmap0->list();
  const int bs0 = dofmap0->bs();
  const graph::AdjacencyList<std::int32_t>& dofs1 = dofmap1->list();
  const int bs1 = dofmap1->bs();
  std::vector<bool> bc0;
  std::vector<bool> bc1;

  // Pack constants and coefficients
  const std::vector<double> constants = dolfinx::fem::pack_constants(*a);
  const array2d<double> coeffs = dolfinx::fem::pack_coefficients(*a);

  auto cell_info = mesh->topology().get_cell_permutation_info();

  auto kernel = dolfinx_cuas::generate_kernel("Lagrange", "tetrahedron", Kernel::Stiffness, 1);

  // Matrix to be used with custom assembler
  la::PETScMatrix A = la::PETScMatrix(fem::create_matrix(*a), false);
  MatZeroEntries(A.mat());
  
  // Matrix to be used with custom DOLFINx/FFCx
  la::PETScMatrix B = la::PETScMatrix(fem::create_matrix(*a), false);
  MatZeroEntries(B.mat());

  common::Timer t0("~Assemble Matrix Custom");
  dolfinx::fem::impl::assemble_cells<double>(la::PETScMatrix::set_block_fn(A.mat(), ADD_VALUES),
                                             mesh->geometry(), active_cells, dofs0, bs0, dofs1, bs1,
                                             bc0, bc1, kernel, coeffs, constants, cell_info);
  MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);
  t0.stop();

  common::Timer t1("~Assemble Matrix DOLINFx/FFCx");
  dolfinx::fem::assemble_matrix(la::PETScMatrix::set_block_fn(B.mat(), ADD_VALUES), *a, {});
  MatAssemblyBegin(B.mat(), MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(B.mat(), MAT_FINAL_ASSEMBLY);
  t1.stop();

  double normA;
  MatNorm(A.mat(), NORM_FROBENIUS, &normA);

  double normB;
  MatNorm(A.mat(), NORM_FROBENIUS, &normB);

  assert(xt::isclose(normA, normB));

  dolfinx::list_timings(mpi_comm, {dolfinx::TimingType::wall});

  return 0;
}