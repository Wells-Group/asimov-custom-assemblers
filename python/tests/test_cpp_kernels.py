# Copyright (C) 2021 Jørgen S. Dokken
#
# SPDX-License-Identifier:   LGPL-3.0-or-later

import dolfinx
import dolfinx_cuas
import dolfinx_cuas.cpp
import numpy as np
import pytest
import ufl
from mpi4py import MPI
from petsc4py import PETSc

kt = dolfinx_cuas.cpp.Kernel
it = dolfinx.cpp.fem.IntegralType
compare_matrices = dolfinx_cuas.utils.compare_matrices


@pytest.mark.parametrize("kernel_type", [kt.Mass, kt.Stiffness, kt.SymGrad])
def test_manifold(kernel_type):
    gdim = 3
    shape = "triangle"
    degree = 1
    cell = ufl.Cell(shape, geometric_dimension=gdim)
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, degree))

    x = np.array([[0.0, 0.0, 0.1], [2, 0., 0.0], [0, 1.5, 0.3]])
    cells = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, x, domain)
    mesh.topology.create_connectivity_all()
    facets = np.arange(mesh.topology.index_map(mesh.topology.dim - 1).size_local, dtype=np.int32)
    values = np.ones(len(facets), dtype=np.int32)
    ft = dolfinx.MeshTags(mesh, mesh.topology.dim - 1, facets, values)

    # Define variational form
    V = dolfinx.VectorFunctionSpace(mesh, ("CG", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)
    if kernel_type == kt.Mass:
        a = ufl.inner(u, v) * ds(1)
    elif kernel_type == kt.Stiffness:
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ds(1)
    elif kernel_type == kt.SymGrad:
        def epsilon(v):
            return ufl.sym(ufl.grad(v))
        a = ufl.inner(epsilon(u), epsilon(v)) * ds(1)
    else:
        raise RuntimeError("Unknown kernel")

    quadrature_degree = dolfinx_cuas.estimate_max_polynomial_degree(a)
    # Compile UFL form
    cffi_options = ["-Ofast", "-march=native"]
    a = dolfinx.fem.Form(a, jit_parameters={"cffi_extra_compile_args": cffi_options, "cffi_libraries": ["m"]})
    A = dolfinx.fem.create_matrix(a)

    # Normal assembly
    A.zeroEntries()
    dolfinx.fem.assemble_matrix(A, a)
    A.assemble()

    # Custom assembly
    num_local_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    consts = np.zeros(0)
    coeffs = np.zeros((num_local_cells, 0), dtype=PETSc.ScalarType)
    B = dolfinx.fem.create_matrix(a)
    kernel = dolfinx_cuas.cpp.generate_surface_kernel(V._cpp_object, kernel_type, quadrature_degree)
    B.zeroEntries()
    dolfinx_cuas.assemble_matrix(B, V, ft.indices, kernel, coeffs, consts, it.exterior_facet)
    B.assemble()

    # Compare matrices, first norm, then entries
    assert np.isclose(A.norm(), B.norm())


@pytest.mark.parametrize("kernel_type", [kt.Mass, kt.Stiffness, kt.SymGrad])
@pytest.mark.parametrize("dim", [2, 3])
def test_surface_kernels(dim, kernel_type):
    N = 30 if dim == 2 else 10
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, N, N) if dim == 2 else dolfinx.UnitCubeMesh(MPI.COMM_WORLD, N, N, N)

    # Find facets on boundary to integrate over
    facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1,
                                                   lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                                                           np.isclose(x[0], 1.0)))
    values = np.ones(len(facets), dtype=np.int32)
    ft = dolfinx.MeshTags(mesh, mesh.topology.dim - 1, facets, values)

    # Define variational form
    V = dolfinx.VectorFunctionSpace(mesh, ("CG", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)
    if kernel_type == kt.Mass:
        a = ufl.inner(u, v) * ds(1)
    elif kernel_type == kt.Stiffness:
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ds(1)
    elif kernel_type == kt.SymGrad:
        def epsilon(v):
            return ufl.sym(ufl.grad(v))
        a = ufl.inner(epsilon(u), epsilon(v)) * ds(1)
    else:
        raise RuntimeError("Unknown kernel")
    quadrature_degree = dolfinx_cuas.estimate_max_polynomial_degree(a)
    # Compile UFL form
    cffi_options = ["-Ofast", "-march=native"]
    a = dolfinx.fem.Form(a, jit_parameters={"cffi_extra_compile_args": cffi_options, "cffi_libraries": ["m"]})
    A = dolfinx.fem.create_matrix(a)

    # Normal assembly
    A.zeroEntries()
    dolfinx.fem.assemble_matrix(A, a)
    A.assemble()

    # Custom assembly
    num_local_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    consts = np.zeros(0)
    coeffs = np.zeros((num_local_cells, 0), dtype=PETSc.ScalarType)

    B = dolfinx.fem.create_matrix(a)
    kernel = dolfinx_cuas.cpp.generate_surface_kernel(V._cpp_object, kernel_type, quadrature_degree)
    B.zeroEntries()
    dolfinx_cuas.assemble_matrix(B, V, ft.indices, kernel, coeffs, consts, it.exterior_facet)
    B.assemble()

    # Compare matrices, first norm, then entries
    assert np.isclose(A.norm(), B.norm())
    compare_matrices(A, B)


@pytest.mark.parametrize("kernel_type", [kt.Normal])
@pytest.mark.parametrize("dim", [2, 3])
def test_normal_kernels(dim, kernel_type):
    N = 30 if dim == 2 else 10
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, N, N) if dim == 2 else dolfinx.UnitCubeMesh(MPI.COMM_WORLD, N, N, N)

    facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1,
                                                   lambda x: np.full(x.shape[1], True, dtype=bool))
    values = np.ones(len(facets), dtype=np.int32)
    # Find facets on boundary to integrate over2)
    ft = dolfinx.MeshTags(mesh, mesh.topology.dim - 1, facets, values)

    # Define variational form
    V = dolfinx.VectorFunctionSpace(mesh, ("CG", 1))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)

    n = ufl.FacetNormal(mesh)

    def epsilon(v):
        return ufl.sym(ufl.grad(v))

    a = 2 * ufl.inner(epsilon(u) * n, v) * ds(1)
    quadrature_degree = dolfinx_cuas.estimate_max_polynomial_degree(a)
    # Compile UFL form
    cffi_options = ["-Ofast", "-march=native"]
    a = dolfinx.fem.Form(a, jit_parameters={"cffi_extra_compile_args": cffi_options, "cffi_libraries": ["m"]})
    A = dolfinx.fem.create_matrix(a)

    # Normal assembly
    A.zeroEntries()
    dolfinx.fem.assemble_matrix(A, a)
    A.assemble()

    # Custom assembly
    num_local_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    consts = np.zeros(0)
    coeffs = np.zeros((num_local_cells, 0), dtype=PETSc.ScalarType)

    B = dolfinx.fem.create_matrix(a)
    kernel = dolfinx_cuas.cpp.generate_surface_kernel(V._cpp_object, kernel_type, quadrature_degree)
    B.zeroEntries()
    dolfinx_cuas.assemble_matrix(B, V, ft.indices, kernel, coeffs, consts, it.exterior_facet)

    B.assemble()

    # Compare matrices, first norm, then entries
    assert np.isclose(A.norm(), B.norm())
    compare_matrices(A, B)


@pytest.mark.parametrize("kernel_type", [kt.Mass, kt.Stiffness])
@pytest.mark.parametrize("P", [1, 2, 3, 4, 5])
def test_volume_kernels(kernel_type, P):
    N = 4
    mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, N, N, N)
    # Define variational form
    V = dolfinx.FunctionSpace(mesh, ("CG", P))
    bs = V.dofmap.index_map_bs

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh)
    if kernel_type == kt.Mass:
        a = ufl.inner(u, v) * dx
    elif kernel_type == kt.Stiffness:
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    else:
        raise RuntimeError("Unknown kernel")

    # Compile UFL form
    cffi_options = ["-Ofast", "-march=native"]
    a = dolfinx.fem.Form(a, jit_parameters={"cffi_extra_compile_args": cffi_options, "cffi_libraries": ["m"]})
    A = dolfinx.fem.create_matrix(a)

    # Normal assembly
    A.zeroEntries()
    dolfinx.fem.assemble_matrix(A, a)
    A.assemble()

    # Custom assembly
    num_local_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    active_cells = np.arange(num_local_cells, dtype=np.int32)
    B = dolfinx.fem.create_matrix(a)
    kernel = dolfinx_cuas.cpp.generate_kernel(kernel_type, P, bs)
    B.zeroEntries()
    consts = np.zeros(0)
    coeffs = np.zeros((num_local_cells, 0), dtype=PETSc.ScalarType)
    dolfinx_cuas.assemble_matrix(B, V, active_cells, kernel, coeffs, consts, it.cell)
    B.assemble()

    # Compare matrices, first norm, then entries
    assert np.isclose(A.norm(), B.norm())
    compare_matrices(A, B)


@pytest.mark.parametrize("kernel_type", [kt.TrEps, kt.SymGrad, kt.Stiffness, kt.Mass])
@pytest.mark.parametrize("P", [1, 2, 3, 4, 5])
def test_vector_cell_kernel(kernel_type, P):
    N = 5
    mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, N, N, N)
    V = dolfinx.VectorFunctionSpace(mesh, ("CG", P))
    bs = V.dofmap.index_map_bs
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh)

    def epsilon(v):
        return ufl.sym(ufl.grad(v))

    if kernel_type == kt.Mass:
        a = ufl.inner(u, v) * dx
    elif kernel_type == kt.Stiffness:
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    elif kernel_type == kt.TrEps:
        a = ufl.inner(ufl.tr(epsilon(u)) * ufl.Identity(len(u)), epsilon(v)) * dx
    elif kernel_type == kt.SymGrad:
        a = 2 * ufl.inner(epsilon(u), epsilon(v)) * dx
    else:
        raise RuntimeError("Unknown kernel")

    # Compile UFL form
    cffi_options = ["-Ofast", "-march=native"]
    a = dolfinx.fem.Form(a, jit_parameters={"cffi_extra_compile_args": cffi_options, "cffi_libraries": ["m"]})
    A = dolfinx.fem.create_matrix(a)

    # Normal assembly
    A.zeroEntries()
    dolfinx.fem.assemble_matrix(A, a)
    A.assemble()

    # Custom assembly
    num_local_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    active_cells = np.arange(num_local_cells, dtype=np.int32)
    consts = np.zeros(0)
    coeffs = np.zeros((num_local_cells, 0), dtype=PETSc.ScalarType)
    B = dolfinx.fem.create_matrix(a)
    kernel = dolfinx_cuas.cpp.generate_kernel(kernel_type, P, bs)
    B.zeroEntries()
    dolfinx_cuas.assemble_matrix(B, V, active_cells, kernel, coeffs, consts, it.cell)
    B.assemble()

    # Compare matrices, first norm, then entries
    assert np.isclose(A.norm(), B.norm())
    compare_matrices(A, B)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("vector", [True, False])
@pytest.mark.parametrize("P", [1, 2, 3, 4, 5])
def test_surface_non_affine(P, vector, dim):
    if dim == 3:
        x = np.array([[0, 0, 0], [0, 1, 0], [0, 0.2, 0.8], [0, 0.9, 0.7],
                      [0.7, 0.1, 0.2], [0.9, 0.9, 0.1], [0.8, 0.1, 0.9], [1, 1, 1]])

        cells = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)
        ct = "hexahedron"

        cell = ufl.Cell(ct, geometric_dimension=x.shape[1])
        domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 1))
        mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, x, domain)
    else:
        x = np.array([[0, 0], [0.5, 0], [0, 1], [0.6, 1],
                      [1, 0], [0.7, 1]])

        cells = np.array([[0, 1, 2, 3], [1, 4, 3, 5]], dtype=np.int32)
        ct = "quadrilateral"

        cell = ufl.Cell(ct, geometric_dimension=x.shape[1])
        domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 1))
        mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, x, domain)
    el = ufl.VectorElement("CG", mesh.ufl_cell(), P) if vector \
        else ufl.FiniteElement("CG", mesh.ufl_cell(), P)
    V = dolfinx.FunctionSpace(mesh, el)

    # Find facets on boundary to integrate over
    facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1,
                                                   lambda x: np.isclose(x[0], 0.0))

    values = np.ones(len(facets), dtype=np.int32)
    ft = dolfinx.MeshTags(mesh, mesh.topology.dim - 1, facets, values)

    # Define variational form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)
    a = ufl.inner(u, v) * ds(1)
    quadrature_degree = dolfinx_cuas.estimate_max_polynomial_degree(a)

    # Compile UFL form
    cffi_options = ["-Ofast", "-march=native"]
    a = dolfinx.fem.Form(a, jit_parameters={"cffi_extra_compile_args": cffi_options, "cffi_libraries": ["m"]})
    A = dolfinx.fem.create_matrix(a)

    # Normal assembly
    A.zeroEntries()
    dolfinx.fem.assemble_matrix(A, a)
    A.assemble()

    # Custom assembly
    B = dolfinx.fem.create_matrix(a)
    num_local_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    consts = np.zeros(0)
    coeffs = np.zeros((num_local_cells, 0), dtype=PETSc.ScalarType)
    kernel = dolfinx_cuas.cpp.generate_surface_kernel(V._cpp_object, kt.MassNonAffine, quadrature_degree)
    B.zeroEntries()
    dolfinx_cuas.assemble_matrix(B, V, ft.indices, kernel, consts, coeffs, it.exterior_facet)
    B.assemble()

    # Compare matrices, first norm, then entries
    assert np.isclose(A.norm(), B.norm())
    compare_matrices(A, B)
