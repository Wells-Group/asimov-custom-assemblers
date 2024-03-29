# Copyright (C) 2021 Sarah Roggendorf
#
# SPDX-License-Identifier:   MIT

import basix
import dolfinx_cuas.cpp
import dolfinx_cuas.utils
import numpy as np
import pytest
import ufl
from dolfinx import fem
from dolfinx import mesh as _mesh
from mpi4py import MPI


@pytest.mark.parametrize("kernel_type", [dolfinx_cuas.Kernel.Mass])
@pytest.mark.parametrize("P", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("Q", [1, 2, 3])
def test_volume_kernels(kernel_type, P, Q):
    N = 4
    mesh = _mesh.create_unit_cube(MPI.COMM_WORLD, N, N, N)
    # Define variational form
    V = fem.FunctionSpace(mesh, ("CG", P))

    def f(x):
        values = np.zeros((1, x.shape[1]))
        for i in range(x.shape[1]):
            values[0, i] = np.max(np.abs(x[:, i]))
        return values

    V2 = fem.FunctionSpace(mesh, ("DG", Q - 1))
    mu = fem.Function(V2)
    mu.interpolate(f)

    V3 = fem.FunctionSpace(mesh, ("CG", Q))
    lam = fem.Function(V3)
    lam.interpolate(f)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh)
    if kernel_type == dolfinx_cuas.Kernel.Mass:
        a = (mu + lam) * ufl.inner(u, v) * dx
    elif kernel_type == dolfinx_cuas.Kernel.Stiffness:
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    else:
        raise RuntimeError("Unknown kernel")

    # Compile UFL form
    cffi_options = ["-Ofast", "-march=native"]
    a = fem.form(a, jit_options={"cffi_extra_compile_args": cffi_options, "cffi_libraries": ["m"]})
    A = fem.petsc.create_matrix(a)

    # Normal assembly
    A.zeroEntries()
    fem.petsc.assemble_matrix(A, a)
    A.assemble()

    # Custom assembly
    num_local_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    active_cells = np.arange(num_local_cells, dtype=np.int32)
    B = fem.petsc.create_matrix(a)
    quadrature_degree = 2 * P + Q
    q_rule = dolfinx_cuas.QuadratureRule(
        mesh.topology.cell_type, quadrature_degree, mesh.topology.dim, basix.quadrature.string_to_type("default"))
    kernel = dolfinx_cuas.cpp.generate_coeff_kernel(kernel_type, [mu._cpp_object, lam._cpp_object], P, q_rule)
    B.zeroEntries()
    consts = np.zeros(0)
    coeffs = dolfinx_cuas.pack_coefficients([mu._cpp_object, lam._cpp_object], active_cells)
    dolfinx_cuas.assemble_matrix(B, V, active_cells, kernel, coeffs, consts, fem.IntegralType.cell)
    B.assemble()

    # Compare matrices, first norm, then entries
    assert np.isclose(A.norm(), B.norm())
    dolfinx_cuas.utils.compare_matrices(A, B)
