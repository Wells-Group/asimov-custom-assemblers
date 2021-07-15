// Copyright (C) 2021 Igor A. Baratta, Sarah Roggendorf
//
// This file is part of DOLFINx_CUAS
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "problem.h"
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/mesh/utils.h>
#include <dolfinx_cuas/assembly.hpp>
#include <dolfinx_cuas/contact/Contact.hpp>
#include <xtensor/xio.hpp>

using namespace dolfinx;

int main(int argc, char* argv[])
{
  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_petsc(argc, argv);
  int tag = 1;

  MPI_Comm mpi_comm{MPI_COMM_WORLD};

  //   std::shared_ptr<mesh::Mesh> mesh = std::make_shared<mesh::Mesh>(
  //       generation::BoxMesh::create(mpi_comm, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {1, 1, 1},
  //                                   mesh::CellType::tetrahedron, mesh::GhostMode::none));
  auto xdmf = dolfinx::io::XDMFFile(mpi_comm, "mesh_3d.xdmf", "r");

  auto cell_type = xdmf.read_cell_type("Grid");

  //   std::shared_ptr<mesh::Mesh> mesh = std::make_shared<mesh::Mesh>(
  //       xdmf.read_mesh(dolfinx::fem::CoordinateElement(cell_type.first, cell_type.second),
  //                      mesh::GhostMode::none, "mesh"));
  std::shared_ptr<mesh::Mesh> mesh = std::make_shared<mesh::Mesh>(
      xdmf.read_mesh(dolfinx::fem::CoordinateElement(cell_type.first, cell_type.second),
                     mesh::GhostMode::none, "Grid"));
  mesh->topology().create_entity_permutations();

  auto xdmf2 = dolfinx::io::XDMFFile(mpi_comm, "mesh_3d_facets.xdmf", "r");
  //   std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>> mt
  //       = std::make_shared<dolfinx::mesh::MeshTags<std::int32_t>>(
  //           xdmf.read_meshtags(mesh, "mesh_tags"));
  std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>> mt
      = std::make_shared<dolfinx::mesh::MeshTags<std::int32_t>>(xdmf2.read_meshtags(mesh, "Grid"));
  //   mesh->topology().create_entity_permutations();

  //   auto left_boundary
  //       = [](auto& x) -> xt::xtensor<bool, 1> { return xt::isclose(xt::row(x, 0), 0.0); };

  //   std::vector<std::int32_t> left_facets
  //       = dolfinx::mesh::locate_entities_boundary(*mesh, 2, left_boundary);
  //   std::vector<std::int32_t> left_values(left_facets.size());
  //   std::fill(left_values.begin(), left_values.end(), 1);
  //   auto mt
  //       = std::make_shared<dolfinx::mesh::MeshTags<std::int32_t>>(mesh, 2, left_facets,
  //       left_values);

  auto left_facets = mt->find(tag);

  auto x_dofs = mesh->geometry().x();
  auto coord = dolfinx::mesh::entities_to_geometry(*mesh, 2, left_facets, false);
  const std::shared_ptr<fem::FunctionSpace>& V
      = fem::create_functionspace(functionspace_form_problem_a, "u", mesh);
  auto contact = dolfinx_cuas::contact::Contact(mt, tag, tag, V);
  contact.create_reference_facet_qp();
  auto kernel = contact.generate_surface_kernel(0, dolfinx_cuas::Kernel::Contact_Jac);

  // Define variational forms
  auto kappa = std::make_shared<fem::Constant<PetscScalar>>(1.0);
  auto a = std::make_shared<fem::Form<PetscScalar>>(
      fem::create_form<PetscScalar>(*form_problem_a, {V, V}, {}, {{"kappa", kappa}},
                                    {{dolfinx::fem::IntegralType::exterior_facet, &(*mt)}}));

  // Define active cells
  const std::int32_t tdim = mesh->topology().dim();
  const std::int32_t ncells = mesh->topology().index_map(tdim)->size_local();
  xt::xarray<std::int32_t> active_cells = xt::arange<std::int32_t>(0, ncells);

  // auto kernel = dolfinx_cuas::generate_kernel("Lagrange", "tetrahedron",
  // Kernel::Stiffness, 1);

  // Matrix to be used with custom assembler
  la::PETScMatrix A = la::PETScMatrix(fem::create_matrix(*a), false);
  MatZeroEntries(A.mat());

  // Matrix to be used with custom DOLFINx/FFCx
  la::PETScMatrix B = la::PETScMatrix(fem::create_matrix(*a), false);
  MatZeroEntries(B.mat());
  common::Timer t0("~Assemble Matrix Custom");
  dolfinx_cuas::assemble_exterior_facets(la::PETScMatrix::set_block_fn(A.mat(), ADD_VALUES), a,
                                         left_facets, kernel);
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
  MatNorm(B.mat(), NORM_FROBENIUS, &normB);
  std::cout << "norm A: " << normA << ", normB: " << normB << "\n";
  assert(xt::isclose(normA, normB));

  dolfinx::list_timings(mpi_comm, {dolfinx::TimingType::wall});

  return 0;
}