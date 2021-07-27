# Copyright (C) 2021 Jørgen S. Dokken
#
# SPDX-License-Identifier:   LGPL-3.0-or-later

import dolfinx
import dolfinx_cuas.cpp
import numpy as np
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
