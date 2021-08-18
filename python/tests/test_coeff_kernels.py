# Copyright (C) 2021 Sarah Roggendorf
#
# SPDX-License-Identifier:   LGPL-3.0-or-later

import dolfinx
import dolfinx_cuas.cpp
import dolfinx_cuas.utils
import numpy as np
import pytest
import ufl
from mpi4py import MPI

kt = dolfinx_cuas.cpp.Kernel
it = dolfinx.cpp.fem.IntegralType


@pytest.mark.parametrize("kernel_type", [kt.Mass])
@pytest.mark.parametrize("P", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("Q", [1, 2, 3])
def test_volume_kernels(kernel_type, P, Q):
    N = 4
    mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, N, N, N)
    # Define variational form
    V = dolfinx.FunctionSpace(mesh, ("CG", P))

    def f(x):
        values = np.zeros((1, x.shape[1]))
        for i in range(x.shape[1]):
            values[0, i] = np.max(np.abs(x[:, i]))
        return values

    V2 = dolfinx.FunctionSpace(mesh, ("DG", Q - 1))
    mu = dolfinx.Function(V2)
    mu.interpolate(f)

    V3 = dolfinx.FunctionSpace(mesh, ("CG", Q))
    lam = dolfinx.Function(V3)
    lam.interpolate(f)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh)
    if kernel_type == kt.Mass:
        a = (mu + lam) * ufl.inner(u, v) * dx
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
    quadrature_degree = 2 * P + Q
    q_rule = dolfinx_cuas.cpp.QuadratureRule(mesh.topology.cell_type, quadrature_degree, "default")
    kernel = dolfinx_cuas.cpp.generate_coeff_kernel(kernel_type, [mu._cpp_object, lam._cpp_object], P, q_rule)
    B.zeroEntries()
    consts = np.zeros(0)
    coeffs = dolfinx_cuas.cpp.pack_coefficients([mu._cpp_object, lam._cpp_object])
    dolfinx_cuas.assemble_matrix(B, V, active_cells, kernel, coeffs, consts, it.cell)
    B.assemble()

    # Compare matrices, first norm, then entries
    assert np.isclose(A.norm(), B.norm())
    dolfinx_cuas.utils.compare_matrices(A, B)


if __name__ == "__main__":
    test_volume_kernels(kt.Mass, 2, 5)