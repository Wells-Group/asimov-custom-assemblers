// Copyright (C) 2021 Igor A. Baratta
//
// This file is part of DOLFINx_CUAS
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "volume.h"
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <boost/program_options.hpp>
#include <dolfinx.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx_cuas/assembly.hpp>
#include <dolfinx_cuas/kernels.hpp>
#include <dolfinx_cuas/utils.hpp>
#include <xtensor/xio.hpp>

using namespace dolfinx;
namespace po = boost::program_options;

int main(int argc, char* argv[])
{
  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_petsc(argc, argv);

  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "print usage message")(
      "kernel", po::value<std::string>()->default_value("mass"),
      "kernel (mass or stiffness)")("degree", po::value<int>()->default_value(1),
                                    "Degree of function space (1-5)");
  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("help"))
  {
    std::cout << desc << "\n";
    return 0;
  }
  const std::string problem_type = vm["kernel"].as<std::string>();
  const int degree = vm["degree"].as<int>();

  MPI_Comm mpi_comm{MPI_COMM_WORLD};

  std::shared_ptr<mesh::Mesh> mesh = std::make_shared<mesh::Mesh>(
      generation::BoxMesh::create(mpi_comm, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {25, 25, 25},
                                  mesh::CellType::tetrahedron, mesh::GhostMode::none));

  mesh->topology().create_entity_permutations();

  auto kappa = std::make_shared<fem::Constant<PetscScalar>>(1.0);

  // Define variational forms
  ufc_form form;
  std::shared_ptr<fem::FunctionSpace> V;
  dolfinx_cuas::Kernel kernel_type;
  if (problem_type == "mass")
  {
    kernel_type = dolfinx_cuas::Kernel::MassTensor;
    switch (degree)
    {
    case 1:
      V = fem::create_functionspace(functionspace_form_volume_a_mass1, "v_0", mesh);
      form = *form_volume_a_mass1;
      break;
    case 2:
      V = fem::create_functionspace(functionspace_form_volume_a_mass2, "v_0", mesh);
      form = *form_volume_a_mass2;
      break;
    case 3:
      V = fem::create_functionspace(functionspace_form_volume_a_mass3, "v_0", mesh);
      form = *form_volume_a_mass3;
      break;
    case 4:
      V = fem::create_functionspace(functionspace_form_volume_a_mass4, "v_0", mesh);
      form = *form_volume_a_mass4;
      break;
    case 5:
      V = fem::create_functionspace(functionspace_form_volume_a_mass5, "v_0", mesh);
      form = *form_volume_a_mass5;
      break;
    }
  }
  else if (problem_type == "stiffness")
  {
    kernel_type = dolfinx_cuas::Kernel::Stiffness;
    switch (degree)
    {
    case 1:
      V = fem::create_functionspace(functionspace_form_volume_a_stiffness1, "v_0", mesh);
      form = *form_volume_a_stiffness1;
      break;
    case 2:
      V = fem::create_functionspace(functionspace_form_volume_a_stiffness2, "v_0", mesh);
      form = *form_volume_a_stiffness2;
      break;
    case 3:
      V = fem::create_functionspace(functionspace_form_volume_a_stiffness3, "v_0", mesh);
      form = *form_volume_a_stiffness3;
      break;
    case 4:
      V = fem::create_functionspace(functionspace_form_volume_a_stiffness4, "v_0", mesh);
      form = *form_volume_a_stiffness4;
      break;
    case 5:
      V = fem::create_functionspace(functionspace_form_volume_a_stiffness5, "v_0", mesh);
      form = *form_volume_a_stiffness5;
      break;
    }
  }
  else
    throw std::runtime_error("Unsupported kernel");

  auto a = std::make_shared<fem::Form<PetscScalar>>(
      fem::create_form<PetscScalar>(form, {V, V}, {}, {{"kappa", kappa}}, {}));

  // Matrix to be used with custom assembler
  la::PETScMatrix A = la::PETScMatrix(fem::create_matrix(*a), false);
  MatZeroEntries(A.mat());

  // Matrix to be used with custom DOLFINx/FFCx
  la::PETScMatrix B = la::PETScMatrix(fem::create_matrix(*a), false);
  MatZeroEntries(B.mat());

  // Generate Kernel
  auto kernel = dolfinx_cuas::generate_kernel(kernel_type, degree, V->dofmap()->index_map_bs());

  // Define active cells
  const std::int32_t tdim = mesh->topology().dim();
  const std::int32_t ncells = mesh->topology().index_map(tdim)->size_local();
  xt::xarray<std::int32_t> active_cells = xt::arange<std::int32_t>(0, ncells);

  common::Timer t0("~Assemble Matrix Custom");
  std::array<std::size_t, 2> shape = {std::size_t(ncells), 0};
  const dolfinx::array2d<PetscScalar> coeffs(shape);
  const std::vector<PetscScalar> consts(0);
  dolfinx_cuas::assemble_matrix(la::PETScMatrix::set_block_fn(A.mat(), ADD_VALUES), V, {},
                                active_cells, kernel, coeffs, consts,
                                dolfinx::fem::IntegralType::cell);
  MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);
  t0.stop();

  common::Timer t1("~Assemble Matrix DOLINFx/FFCx");
  dolfinx::fem::assemble_matrix(la::PETScMatrix::set_block_fn(B.mat(), ADD_VALUES), *a, {});
  MatAssemblyBegin(B.mat(), MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(B.mat(), MAT_FINAL_ASSEMBLY);
  t1.stop();

  dolfinx::list_timings(mpi_comm, {dolfinx::TimingType::wall});

  if (!dolfinx_cuas::allclose(A.mat(), B.mat()))
    throw std::runtime_error("Matrices are not the same");

  return 0;
}
