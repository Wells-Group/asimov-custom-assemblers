# Copyright (C) 2021 Jørgen S. Dokken
#
# SPDX-License-Identifier:   LGPL-3.0-or-later

import basix
import dolfinx
import dolfinx_cuas.cpp
import numpy as np
import pytest
import ufl
from mpi4py import MPI


def test_pack_coeffs():
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 10, 10)
    V = dolfinx.VectorFunctionSpace(mesh, ("CG", 2))
    Q = dolfinx.VectorFunctionSpace(mesh, ("DG", 1))
    Z = dolfinx.FunctionSpace(mesh, ("DG", 0))
    v = dolfinx.Function(V)
    q = dolfinx.Function(Q)
    z = dolfinx.Function(Z)
    a = z * ufl.inner(v, z * q) * ufl.dx
    form = dolfinx.Form(a)._cpp_object
    coeffs = dolfinx.cpp.fem.pack_coefficients(form)
    coeffs_cuas = dolfinx_cuas.cpp.pack_coefficients(form.coefficients)
    assert np.allclose(coeffs, coeffs_cuas)


@pytest.mark.parametrize("quadrature_degree", range(1, 6))
@pytest.mark.parametrize("space", ["CG", "N1curl"])
def test_pack_coeff_at_quadrature(quadrature_degree, space):
    N = 100
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, N, N)
    if space == "CG":
        V = dolfinx.VectorFunctionSpace(mesh, (space, 1))
    elif space == "N1curl":
        V = dolfinx.FunctionSpace(mesh, (space, 1))
    else:
        raise RuntimeError("Unsupported space")

    v = dolfinx.Function(V)
    v.interpolate(lambda x: (x[1], -x[0]))

    # Pack coeffs with cuas
    import time
    start = time.time()
    coeffs_cuas = dolfinx_cuas.cpp.pack_coefficient_quadrature(v._cpp_object, quadrature_degree)
    end = time.time()
    # Use eval and push forward to evaluate dofs
    quadrature_points, wts = basix.make_quadrature("default", basix.CellType.triangle, quadrature_degree)
    x_g = mesh.geometry.x
    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    coord_dofs = mesh.geometry.dofmap
    tot = 0
    for cell in range(num_cells):
        xg = x_g[coord_dofs.links(cell)]
        x = mesh.geometry.cmap.push_forward(quadrature_points, xg)
        start2 = time.time()
        v_ex = v.eval(x, np.full(x.shape[0], cell))
        end2 = time.time()
        tot += end2 - start2
        assert(np.allclose(v_ex.reshape(-1), coeffs_cuas[cell]))
    print(end - start, tot, tot / (end - start))
