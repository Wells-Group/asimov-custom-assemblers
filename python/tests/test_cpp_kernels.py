# Copyright (C) 2021 Jørgen S. Dokken
#
# SPDX-License-Identifier:   LGPL-3.0-or-later

import dolfinx
import dolfinx_cuas
import dolfinx_cuas.cpp
import numpy as np
import pytest
import scipy.sparse
import ufl
from mpi4py import MPI
from petsc4py import PETSc

kt = dolfinx_cuas.cpp.Kernel


def compare_matrices(A: PETSc.Mat, B: PETSc.Mat, atol: float = 1e-13):
    """
    Helper for comparing two PETSc matrices
    """
    # Create scipy CSR matrices
    ai, aj, av = A.getValuesCSR()
    A_sp = scipy.sparse.csr_matrix((av, aj, ai))
    bi, bj, bv = B.getValuesCSR()
    B_sp = scipy.sparse.csr_matrix((bv, bj, bi))
    # Compare matrices
    diff = np.abs(A_sp - B_sp)
    assert diff.max() <= atol


@pytest.mark.parametrize("kernel_type", [kt.Mass, kt.Stiffness, kt.Contact_Jac])
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
    elif kernel_type == kt.Contact_Jac:
        def epsilon(v):
            return ufl.sym(ufl.grad(v))
        a = ufl.inner(epsilon(u), epsilon(v)) * ds(1)
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
    B = dolfinx.fem.create_matrix(a)
    contact = dolfinx_cuas.cpp.Contact(ft, 1, 1, V._cpp_object)
    contact.create_reference_facet_qp()
    kernel = contact.generate_surface_kernel(0, kernel_type)
    B.zeroEntries()
    dolfinx_cuas.cpp.assemble_exterior_facets(B, a._cpp_object, ft.indices, kernel)
    B.assemble()

    # Compare matrices, first norm, then entries
    assert np.isclose(A.norm(), B.norm())
    compare_matrices(A, B)


@pytest.mark.parametrize("kernel_type", [kt.Mass, kt.Stiffness])
@pytest.mark.parametrize("P", [1, 2, 3, 4, 5])
def test_volume_kernels(kernel_type, P):
    N = 1
    mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, N, N, N)

    # Define variational form
    V = dolfinx.FunctionSpace(mesh, ("CG", P))
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
    print(active_cells)
    B = dolfinx.fem.create_matrix(a)
    kernel = dolfinx_cuas.cpp.generate_kernel(kernel_type, P)
    B.zeroEntries()
    dolfinx_cuas.cpp.assemble_cells(B, a._cpp_object, active_cells, kernel)
    B.assemble()

    # Compare matrices, first norm, then entries
    assert np.isclose(A.norm(), B.norm())
    compare_matrices(A, B)
