// Copyright (C) 2021-2022 Igor A. Baratta, Sarah Roggendorf, Jørgen S. Dokken
//
// This file is part of DOLFINx_CUAS
//
// SPDX-License-Identifier:    MIT

#include "problem.h"
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx.h>
#include <dolfinx/common/sort.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/mesh/utils.h>
#include <dolfinx_cuas/matrix_assembly.hpp>
#include <dolfinx_cuas/surface_kernels.hpp>
#include <xtensor/xio.hpp>

using namespace dolfinx;

int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);

  PetscInitialize(&argc, &argv, nullptr, nullptr);
  {
    MPI_Comm mpi_comm{MPI_COMM_WORLD};

    const int N = 5;
    std::shared_ptr<mesh::Mesh> mesh = std::make_shared<mesh::Mesh>(
        mesh::create_box(mpi_comm, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {N, N, N},
                         mesh::CellType::tetrahedron, mesh::GhostMode::none));
    mesh->topology().create_entity_permutations();
    // Locate boundary facets with x=0
    auto boundary = [](auto& x) -> xt::xtensor<bool, 1> { return xt::isclose(xt::row(x, 0), 0.0); };
    std::vector<std::int32_t> boundary_facets
        = dolfinx::mesh::locate_entities_boundary(*mesh, 2, boundary);
    dolfinx::radix_sort(xtl::span(boundary_facets));

    std::vector<std::int32_t> boundary_values(boundary_facets.size());
    std::fill(boundary_values.begin(), boundary_values.end(), 1);
    auto mt = std::make_shared<dolfinx::mesh::MeshTags<std::int32_t>>(mesh, 2, boundary_facets,
                                                                      boundary_values);

    // Generate function space
    const int Q = 1; // Degree of function space
    auto V = std::make_shared<fem::FunctionSpace>(
        fem::create_functionspace(functionspace_form_problem_a, "u", mesh));

    // Generate boundary kernel
    const std::int32_t tdim = mesh->topology().dim();

    dolfinx_cuas::QuadratureRule q_rule(mesh->topology().cell_type(), 2 * (Q - 1), tdim - 1);
    auto kernel = dolfinx_cuas::generate_surface_kernel<PetscScalar>(
        V, dolfinx_cuas::Kernel::SymGrad, q_rule);

    // Define variational forms
    auto kappa = std::make_shared<fem::Constant<PetscScalar>>(1.0);
    auto a = std::make_shared<fem::Form<PetscScalar>>(
        fem::create_form<PetscScalar>(*form_problem_a, {V, V}, {}, {{"kappa", kappa}},
                                      {{dolfinx::fem::IntegralType::exterior_facet, &(*mt)}}));

    // Define active cells
    const std::int32_t ncells = mesh->topology().index_map(tdim)->size_local();
    xt::xarray<std::int32_t> active_cells = xt::arange<std::int32_t>(0, ncells);

    // Matrix to be used with custom assembler
    la::petsc::Matrix A = la::petsc::Matrix(fem::petsc::create_matrix(*a), false);
    MatZeroEntries(A.mat());

    // Matrix to be used with custom DOLFINx/FFCx
    la::petsc::Matrix B = la::petsc::Matrix(fem::petsc::create_matrix(*a), false);
    MatZeroEntries(B.mat());
    common::Timer t0("~Assemble Matrix Custom");
    const std::vector<PetscScalar> coeffs(ncells * 0);
    const std::vector<PetscScalar> consts(0);
    dolfinx_cuas::assemble_matrix<PetscScalar>(la::petsc::Matrix::set_block_fn(A.mat(), ADD_VALUES),
                                               V, {}, boundary_facets, kernel, coeffs, 0, consts,
                                               dolfinx::fem::IntegralType::exterior_facet);
    MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);
    t0.stop();

    common::Timer t1("~Assemble Matrix DOLINFx/FFCx");
    dolfinx::fem::assemble_matrix(la::petsc::Matrix::set_block_fn(B.mat(), ADD_VALUES), *a, {});
    MatAssemblyBegin(B.mat(), MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(B.mat(), MAT_FINAL_ASSEMBLY);
    t1.stop();

    double normA;
    MatNorm(A.mat(), NORM_FROBENIUS, &normA);

    double normB;
    MatNorm(B.mat(), NORM_FROBENIUS, &normB);
    assert(xt::isclose(normA, normB));
    if (!dolfinx_cuas::allclose(A.mat(), B.mat()))
      throw std::runtime_error("Matrices are not the same");

    dolfinx::list_timings(mpi_comm, {dolfinx::TimingType::wall});
  }
  PetscFinalize();

  return 0;
}