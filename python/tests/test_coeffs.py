# Copyright (C) 2021 Jørgen S. Dokken
#
# SPDX-License-Identifier:   MIT

from dolfinx import fem, generation, mesh as dmesh
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
        dC = ufl.dx
    elif integral_type == fem.IntegralType.exterior_facet:
        dC = ufl.ds
    elif integral_type == fem.IntegralType.interior_facet:
        dC = ufl.dS
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
        a = z("-") * ufl.inner(v("-"), z("+") * q("-")) * dC
    else:
        a = z * ufl.inner(v, z * q) * dC

    form = fem.Form(a)._cpp_object
    coeffs = fem.pack_coefficients(form)
    active_entities = form.domains(integral_type, -1)
    coeffs_cuas = dolfinx_cuas.cpp.pack_coefficients(form.coefficients, active_entities)
    assert np.allclose(coeffs[(integral_type, -1)], coeffs_cuas)


@pytest.mark.parametrize("integral_type",
                         [fem.IntegralType.cell, fem.IntegralType.exterior_facet,
                          fem.IntegralType.interior_facet])
def test_entity_packing(integral_type):
    """
    Test conversion of entities from a set of local indices to DOLFINX formant

    """
    if integral_type == fem.IntegralType.cell:
        dC = ufl.dx
    elif integral_type == fem.IntegralType.exterior_facet:
        dC = ufl.ds
    elif integral_type == fem.IntegralType.interior_facet:
        dC = ufl.dS
    mesh = generation.UnitSquareMesh(MPI.COMM_WORLD, 6, 4)
    a = fem.Constant(mesh, 1) * dC
    form = fem.Form(a)._cpp_object
    active_entities = form.domains(integral_type, -1)

    if integral_type == fem.IntegralType.cell:
        entities = np.arange(mesh.topology.index_map(mesh.topology.dim).size_local,
                             dtype=np.int32)
    else:
        facet_marker = dmesh.compute_boundary_facets(mesh.topology)

    if integral_type == fem.IntegralType.exterior_facet:
        entities = np.flatnonzero(facet_marker)
    elif integral_type == fem.IntegralType.interior_facet:
        entities = np.flatnonzero(np.invert(facet_marker))
    new_entities = dolfinx_cuas.cpp.compute_active_entities(mesh, np.asarray(entities, dtype=np.int32),
                                                            integral_type)
    assert(np.allclose(active_entities, new_entities))
