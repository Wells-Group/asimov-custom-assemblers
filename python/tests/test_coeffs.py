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
@pytest.mark.parametrize("degree", range(1, 6))
@pytest.mark.parametrize("space", ["CG", "N1curl", "DG"])
def test_pack_coeff_at_quadrature(quadrature_degree, space, degree):
    N = 15
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, N, N)
    if space == "CG":
        V = dolfinx.VectorFunctionSpace(mesh, (space, degree))
    elif space == "N1curl":
        V = dolfinx.FunctionSpace(mesh, (space, degree))
    elif space == "DG":
        V = dolfinx.FunctionSpace(mesh, (space, degree - 1))
    else:
        raise RuntimeError("Unsupported space")

    v = dolfinx.Function(V)
    if space == "DG":
        v.interpolate(lambda x: x[0] < 0.5 + x[1])
    else:
        v.interpolate(lambda x: (x[1], -x[0]))

    # Pack coeffs with cuas
    coeffs_cuas = dolfinx_cuas.cpp.pack_coefficient_quadrature(v._cpp_object, quadrature_degree)

    # Use prepare quadrature points and geometry for eval
    quadrature_points, wts = basix.make_quadrature("default", basix.CellType.triangle, quadrature_degree)
    x_g = mesh.geometry.x
    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    coord_dofs = mesh.geometry.dofmap

    # Eval for each cell
    for cell in range(num_cells):
        xg = x_g[coord_dofs.links(cell)]
        x = mesh.geometry.cmap.push_forward(quadrature_points, xg)
        v_ex = v.eval(x, np.full(x.shape[0], cell))

        # Compare
        assert(np.allclose(v_ex.reshape(-1), coeffs_cuas[cell]))


@pytest.mark.parametrize("quadrature_degree", range(1, 6))
@pytest.mark.parametrize("degree", range(1, 6))
@pytest.mark.parametrize("space", ["CG", "DG"])
def test_pack_coeff_on_facet(quadrature_degree, space, degree):
    N = 15
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, N, N)
    if space == "CG":
        V = dolfinx.VectorFunctionSpace(mesh, (space, degree))
    elif space == "N1curl":
        V = dolfinx.FunctionSpace(mesh, (space, degree))
    elif space == "DG":
        V = dolfinx.FunctionSpace(mesh, (space, degree - 1))
    else:
        raise RuntimeError("Unsupported space")

    v = dolfinx.Function(V)
    if space == "DG":
        v.interpolate(lambda x: x[0] < 0.5 + x[1])
    else:
        v.interpolate(lambda x: (x[1], -x[0]))

    # Find facets on boundary to integrate over
    facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1,
                                                   lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                                                           np.isclose(x[0], 1.0)))

    # Pack coeffs with cuas
    coeffs_cuas = dolfinx_cuas.cpp.pack_coefficient_facet(v._cpp_object, quadrature_degree, facets)

    # Use prepare quadrature points and geometry for eval
    qp_test, wts = basix.make_quadrature("default", basix.CellType.triangle, quadrature_degree)
    x_g = mesh.geometry.x
    tdim = mesh.topology.dim
    fdim = tdim - 1
    num_cells = mesh.topology.index_map(tdim).size_local
    coord_dofs = mesh.geometry.dofmap

    # Connectivity to evaluate at quadrature points
    mesh.topology.create_connectivity(fdim, tdim)
    f_to_c = mesh.topology.connectivity(fdim, tdim)
    mesh.topology.create_connectivity(tdim, fdim)
    c_to_f = mesh.topology.connectivity(tdim, fdim)

    qp, w = dolfinx_cuas.cpp.create_reference_facet_qp(mesh, quadrature_degree)

    # Eval for each cell
    for index, facet in enumerate(facets):
        cell = f_to_c.links(facet)[0]
        xg = x_g[coord_dofs.links(cell)]
        # find local index of facet
        cell_facets = c_to_f.links(cell)
        local_index = np.where(cell_facets == facet)
        quadrature_points = np.zeros((qp.shape[1], qp.shape[2]))
        quadrature_points[:, :] = qp[local_index[0], :, :]
        x = mesh.geometry.cmap.push_forward(quadrature_points, xg)
        v_ex = v.eval(x, np.full(x.shape[0], cell))

        # Compare
        assert(np.allclose(v_ex.reshape(-1), coeffs_cuas[index]))
