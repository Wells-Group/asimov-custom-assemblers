// Copyright (C) 2021 Igor A. Baratta
//
// This file is part of DOLFINx_CUAS
//
// SPDX-License-Identifier:    MIT

#include "volume.h"
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <boost/program_options.hpp>
#include <dolfinx.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx_cuas/QuadratureRule.hpp>
#include <dolfinx_cuas/kernels.hpp>
#include <dolfinx_cuas/matrix_assembly.hpp>
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
      mesh::create_box(mpi_comm, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {10, 10, 10},
                       mesh::CellType::tetrahedron, mesh::GhostMode::none));

  mesh->topology().create_entity_permutations();

  auto kappa = std::make_shared<fem::Constant<PetscScalar>>(1.0);

  // Define variational forms
  ufcx_form form;
  std::shared_ptr<fem::FunctionSpace> V;
  dolfinx_cuas::Kernel kernel_type;

  int q_degree = 0;
  if (problem_type == "mass")
  {
    q_degree = 2 * degree;
    kernel_type = dolfinx_cuas::Kernel::Mass;
    std::vector spaces_mass = {functionspace_form_volume_a_mass1, functionspace_form_volume_a_mass2,
                               functionspace_form_volume_a_mass3, functionspace_form_volume_a_mass4,
                               functionspace_form_volume_a_mass5};
    std::vector forms_mass = {form_volume_a_mass1, form_volume_a_mass2, form_volume_a_mass3,
                              form_volume_a_mass4, form_volume_a_mass5};
    V = std::make_shared<fem::FunctionSpace>(
        fem::create_functionspace(spaces_mass[degree - 1], "v_0", mesh));
    form = *forms_mass[degree - 1];
  }
  else if (problem_type == "stiffness")
  {
    q_degree = 2 * (degree - 1);
    kernel_type = dolfinx_cuas::Kernel::Stiffness;
    std::vector spaces_stiffness
        = {functionspace_form_volume_a_stiffness1, functionspace_form_volume_a_stiffness2,
           functionspace_form_volume_a_stiffness3, functionspace_form_volume_a_stiffness4,
           functionspace_form_volume_a_stiffness5};
    std::vector forms_stiffness
        = {form_volume_a_stiffness1, form_volume_a_stiffness2, form_volume_a_stiffness3,
           form_volume_a_stiffness4, form_volume_a_stiffness5};
    V = std::make_shared<fem::FunctionSpace>(
        fem::create_functionspace(spaces_stiffness[degree - 1], "v_0", mesh));
    form = *forms_stiffness[degree - 1];
  }
  else
    throw std::runtime_error("Unsupported kernel");

  auto a = std::make_shared<fem::Form<PetscScalar>>(
      fem::create_form<PetscScalar>(form, {V, V}, {}, {{"kappa", kappa}}, {}));

  // Matrix to be used with custom assembler
  la::petsc::Matrix A = la::petsc::Matrix(fem::petsc::create_matrix(*a), false);
  MatZeroEntries(A.mat());

  // Matrix to be used with custom DOLFINx/FFCx
  la::petsc::Matrix B = la::petsc::Matrix(fem::petsc::create_matrix(*a), false);
  MatZeroEntries(B.mat());

  // Generate Kernel
  dolfinx_cuas::QuadratureRule q_rule(mesh->topology().cell_type(), q_degree,
                                      mesh->topology().dim(), basix::quadrature::type::Default);
  auto kernel = dolfinx_cuas::generate_kernel<PetscScalar>(kernel_type, degree,
                                                           V->dofmap()->index_map_bs(), q_rule);

  // Define active cells
  const std::int32_t tdim = mesh->topology().dim();
  const std::int32_t ncells = mesh->topology().index_map(tdim)->size_local();
  xt::xarray<std::int32_t> active_cells = xt::arange<std::int32_t>(0, ncells);
  const std::vector<PetscScalar> coeffs(0);
  const std::vector<PetscScalar> consts(0);

  common::Timer t0("~Assemble Matrix Custom");
  dolfinx_cuas::assemble_matrix<PetscScalar>(la::petsc::Matrix::set_block_fn(A.mat(), ADD_VALUES),
                                             V, {}, active_cells, kernel, coeffs, 0, consts,
                                             dolfinx::fem::IntegralType::cell);
  MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);
  t0.stop();

  {
    // Prepare constants and coefficients
    const auto constants = pack_constants(*a);
    const auto coeffs = pack_coefficients(*a);
    common::Timer t1("~Assemble Matrix DOLINFx/FFCx");
    dolfinx::fem::assemble_matrix(la::petsc::Matrix::set_block_fn(B.mat(), ADD_VALUES), *a,
                                  tcb::make_span(constants),
                                  dolfinx::fem::make_coefficients_span(coeffs), {});
    MatAssemblyBegin(B.mat(), MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(B.mat(), MAT_FINAL_ASSEMBLY);
    t1.stop();
  }

  dolfinx::list_timings(mpi_comm, {dolfinx::TimingType::wall});

  if (!dolfinx_cuas::allclose(A.mat(), B.mat()))
    throw std::runtime_error("Matrices are not the same");

  return 0;
}
