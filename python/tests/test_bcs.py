# Copyright (C) 2021 Jørgen S. Dokken
#
# SPDX-License-Identifier:   MIT

from dolfinx import fem
from dolfinx.mesh import create_unit_cube
import basix
import dolfinx_cuas
import dolfinx_cuas.cpp
import numpy as np
import pytest
import ufl
from mpi4py import MPI
from petsc4py import PETSc


it = fem.IntegralType
compare_matrices = dolfinx_cuas.utils.compare_matrices


@pytest.mark.parametrize("P", [1, 2, 3, 4, 5])
def test_dirichlet_bc(P):
    N = 4
    mesh = create_unit_cube(MPI.COMM_WORLD, N, N, N)
    # Define variational form
    V = fem.FunctionSpace(mesh, ("CG", P))
    bs = V.dofmap.index_map_bs

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx

    # Compile UFL form
    cffi_options = ["-Ofast", "-march=native"]
    a = fem.form(a, jit_params={"cffi_extra_compile_args": cffi_options, "cffi_libraries": ["m"]})
    A = fem.petsc.create_matrix(a)

    # Define DirichletBC
    b_dofs = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0))
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 1
    bcs = [fem.dirichletbc(u_bc, b_dofs)]

    # Normal assembly
    A.zeroEntries()
    fem.petsc.assemble_matrix(A, a, bcs=bcs)
    A.assemble()

    # Custom assembly
    num_local_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    active_cells = np.arange(num_local_cells, dtype=np.int32)
    B = fem.petsc.create_matrix(a)
    q_rule = dolfinx_cuas.QuadratureRule(mesh.topology.cell_type, P + P,
                                         mesh.topology.dim, basix.quadrature.string_to_type("default"))
    kernel_type = dolfinx_cuas.Kernel.Stiffness
    kernel = dolfinx_cuas.cpp.generate_kernel(kernel_type, P, bs, q_rule)
    B.zeroEntries()
    consts = np.zeros(0)
    coeffs = np.zeros((num_local_cells, 0), dtype=PETSc.ScalarType)
    dolfinx_cuas.assemble_matrix(B, V, active_cells, kernel, coeffs,
                                 consts, fem.IntegralType.cell, bcs=bcs)
    B.assemble()

    # Compare matrices, first norm, then entries
    assert np.isclose(A.norm(), B.norm())
    compare_matrices(A, B)
