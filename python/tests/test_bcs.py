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


@pytest.mark.parametrize("P", [1, 2, 3, 4, 5])
def test_dirichlet_bc(P):
    N = 4
    mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, N, N, N)
    # Define variational form
    V = dolfinx.FunctionSpace(mesh, ("CG", P))
    bs = V.dofmap.index_map_bs

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh)
    kernel_type = kt.Stiffness
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx

    # Compile UFL form
    cffi_options = ["-Ofast", "-march=native"]
    a = dolfinx.fem.Form(a, jit_parameters={"cffi_extra_compile_args": cffi_options, "cffi_libraries": ["m"]})
    A = dolfinx.fem.create_matrix(a)

    # Define DirichletBC
    b_dofs = dolfinx.fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0))
    u_bc = dolfinx.Function(V)
    u_bc.x.array[:] = 1
    bcs = [dolfinx.DirichletBC(u_bc, b_dofs)]

    # Normal assembly
    A.zeroEntries()
    dolfinx.fem.assemble_matrix(A, a, bcs=bcs)
    A.assemble()

    # Custom assembly
    num_local_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    active_cells = np.arange(num_local_cells, dtype=np.int32)
    B = dolfinx.fem.create_matrix(a)
    q_rule = dolfinx_cuas.cpp.QuadratureRule(mesh, P + P, "default")
    kernel = dolfinx_cuas.cpp.generate_kernel(kernel_type, P, bs, q_rule)
    B.zeroEntries()
    consts = np.zeros(0)
    coeffs = np.zeros((num_local_cells, 0), dtype=PETSc.ScalarType)
    dolfinx_cuas.assemble_matrix(B, V, active_cells, kernel, coeffs,
                                 consts, dolfinx.cpp.fem.IntegralType.cell, bcs=bcs)
    B.assemble()

    # Compare matrices, first norm, then entries
    assert np.isclose(A.norm(), B.norm())
    compare_matrices(A, B)
