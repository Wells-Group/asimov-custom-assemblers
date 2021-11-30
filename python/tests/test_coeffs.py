# Copyright (C) 2021 Jørgen S. Dokken
#
# SPDX-License-Identifier:   LGPL-3.0-or-later

from dolfinx import fem, generation
import dolfinx_cuas.cpp
import numpy as np
import ufl
from mpi4py import MPI
import pytest


@pytest.mark.parametrize("integral_type",
                         [fem.IntegralType.cell, fem.IntegralType.exterior_facet,
                          fem.IntegralType.interior_facet])
def test_pack_coeffs(integral_type):
    if integral_type == fem.IntegralType.cell:
        measure = ufl.dx
    elif integral_type == fem.IntegralType.exterior_facet:
        measure = ufl.ds
    elif integral_type == fem.IntegralType.interior_facet:
        measure = ufl.dS
    mesh = generation.UnitSquareMesh(MPI.COMM_WORLD, 1, 1)
    V = fem.VectorFunctionSpace(mesh, ("CG", 2))
    Q = fem.VectorFunctionSpace(mesh, ("DG", 1))
    Z = fem.FunctionSpace(mesh, ("DG", 0))
    v = fem.Function(V)
    v.interpolate(lambda x: (x[0], x[1]))
    q = fem.Function(Q)
    q.interpolate(lambda x: (3 * x[1], x[0]))
    z = fem.Function(Z)
    z.interpolate(lambda x: 5 * x[0] + 2 * x[1])
    if integral_type == fem.IntegralType.interior_facet:
        a = z("-") * ufl.inner(v("-"), z("+") * q("-")) * measure
    else:
        a = z * ufl.inner(v, z * q) * measure

    form = fem.Form(a)._cpp_object
    coeffs = fem.pack_coefficients(form)
    active_entities = form.domains(integral_type, -1)

    coeffs_cuas = dolfinx_cuas.cpp.pack_coefficients(form.coefficients, active_entities)
    assert np.allclose(coeffs[(integral_type, -1)], coeffs_cuas)
