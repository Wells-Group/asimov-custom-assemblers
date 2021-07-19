// Copyright (C) 2021 Igor A. Baratta
//
// This file is part of DOLFINx_CUAS
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "volume.h"
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx_cuas/assembly.hpp>
#include <dolfinx_cuas/kernels.hpp>
#include <dolfinx_cuas/utils.hpp>

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
      = fem::create_functionspace(functionspace_form_volume_a, "u", mesh);

  // Define variational forms
  auto kappa = std::make_shared<fem::Constant<PetscScalar>>(1.0);
  auto a = std::make_shared<fem::Form<PetscScalar>>(
      fem::create_form<PetscScalar>(*form_volume_a, {V, V}, {}, {{"kappa", kappa}}, {}));

  // Matrix to be used with custom assembler
  la::PETScMatrix A = la::PETScMatrix(fem::create_matrix(*a), false);
  MatZeroEntries(A.mat());

  // Matrix to be used with custom DOLFINx/FFCx
  la::PETScMatrix B = la::PETScMatrix(fem::create_matrix(*a), false);
  MatZeroEntries(B.mat());

  // Generate Kernel
  auto kernel = dolfinx_cuas::generate_kernel(dolfinx_cuas::Kernel::MassTensor, 1);

  // Define active cells
  const std::int32_t tdim = mesh->topology().dim();
  const std::int32_t ncells = mesh->topology().index_map(tdim)->size_local();
  xt::xarray<std::int32_t> active_cells = xt::arange<std::int32_t>(0, ncells);

  common::Timer t0("~Assemble Matrix Custom");
  dolfinx_cuas::assemble_cells(la::PETScMatrix::set_block_fn(A.mat(), ADD_VALUES), a, active_cells,
                               kernel);
  MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);
  t0.stop();

  common::Timer t1("~Assemble Matrix DOLINFx/FFCx");
  dolfinx::fem::assemble_matrix(la::PETScMatrix::set_block_fn(B.mat(), ADD_VALUES), *a, {});
  MatAssemblyBegin(B.mat(), MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(B.mat(), MAT_FINAL_ASSEMBLY);
  t1.stop();

  assert(dolfinx_cuas::allclose(A.mat(), B.mat()));

  dolfinx::list_timings(mpi_comm, {dolfinx::TimingType::wall});

  return 0;
}
