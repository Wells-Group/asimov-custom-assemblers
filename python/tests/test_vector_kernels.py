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


@pytest.mark.parametrize("kernel_type", [kt.Rhs])
@pytest.mark.parametrize("P", [1, 2, 3, 4, 5])
def test_vector_kernels(kernel_type, P):
    N = 1
    mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, N, N, N)
    # Define variational form
    V = dolfinx.FunctionSpace(mesh, ("CG", P))

    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh)
    L = v * dx
    kernel_type = kt.Rhs

    # Compile UFL form
    cffi_options = ["-Ofast", "-march=native"]
    L = dolfinx.fem.Form(L, jit_parameters={"cffi_extra_compile_args": cffi_options, "cffi_libraries": ["m"]})
    b = dolfinx.fem.create_vector(L)

    # Normal assembly
    b.zeroEntries()
    dolfinx.fem.assemble_vector(b, L)
    b.assemble()

    # Custom assembly
    num_local_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    active_cells = np.arange(num_local_cells, dtype=np.int32)
    b2 = dolfinx.fem.create_vector(L)
    kernel = dolfinx_cuas.cpp.generate_vector_kernel(kernel_type, P)
    b2.zeroEntries()
    consts = np.zeros(0)
    coeffs = np.zeros((num_local_cells, 0), dtype=PETSc.ScalarType)
    dolfinx_cuas.assemble_vector(b2, V, active_cells, kernel, coeffs, consts, it.cell)
    b2.assemble()

    print(np.array(b2))
    print(np.array(b))
    assert(np.linalg.norm(b - b2) < 1e-13)
