# Copyright (C) 2021 Jørgen S. Dokken, Igor Baratta, Sarah Roggendorf
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
import dolfinx
import numpy as np
import pytest
import scipy.sparse
import scipy.sparse.linalg
import ufl.algorithms
import ufl
from dolfinx_assemblers import (assemble_matrix,
                                compute_reference_mass_matrix, compute_reference_stiffness_matrix,
                                estimate_max_polynomial_degree)
from mpi4py import MPI


@pytest.mark.parametrize("degree", range(1, 5))
@pytest.mark.parametrize("ct", ["quadrilateral", "triangle", "tetrahedron",
                                "hexahedron"])
@pytest.mark.parametrize("element", [ufl.FiniteElement, ufl.VectorElement])
@pytest.mark.parametrize("integral_type", ["mass", "stiffness"])
def test_cell_kernels(element, ct, degree, integral_type):
    """
    Test assembly of mass matrices on non-affine mesh
    """
    cell_type = dolfinx.cpp.mesh.to_type(ct)
    if cell_type == dolfinx.cpp.mesh.CellType.quadrilateral:
        x = np.array([[0, 0, 0.5], [1, 0, 1.4], [0, 1.3, 0.0], [1.2, 1, 0.0]])
        cells = np.array([[0, 1, 2, 3]], dtype=np.int32)
    elif cell_type == dolfinx.cpp.mesh.CellType.triangle:
        x = np.array([[0, 0, 0.5], [1.1, 0, 1.3], [0.3, 1.0, 0.9], [2, 1.5, 0.0]])
        cells = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int32)
    elif cell_type == dolfinx.cpp.mesh.CellType.tetrahedron:
        x = np.array([[0, 0, 0], [1.1, 0, 0], [0.3, 1.0, 0], [1, 1.2, 1.5], [2, 2, 1.5]])
        cells = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
    elif cell_type == dolfinx.cpp.mesh.CellType.hexahedron:
        x = np.array([[0, 0, 0], [1.1, 0, 0], [0.1, 1, 0], [1, 1.2, 0],
                      [0, 0, 1.2], [1.0, 0, 1], [0, 1, 1], [1, 1, 1]])
        cells = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)
    else:
        raise ValueError(f"Unsupported mesh type {ct}")
    cell = ufl.Cell(ct, geometric_dimension=x.shape[1])
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 1))
    mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, x, domain)
    el = ufl.FiniteElement("CG", mesh.ufl_cell(), degree)
    V = dolfinx.FunctionSpace(mesh, el)

    if integral_type == "mass":
        a_ = ufl.inner(ufl.TrialFunction(V), ufl.TestFunction(V)) * ufl.dx
        reference_code = compute_reference_mass_matrix
    elif integral_type == "stiffness":
        a_ = ufl.inner(ufl.grad(ufl.TrialFunction(V)), ufl.grad(ufl.TestFunction(V))) * ufl.dx
        reference_code = compute_reference_stiffness_matrix

    quadrature_degree = estimate_max_polynomial_degree(a_) + 1
    if integral_type == "stiffness" and element == ufl.VectorElement:
        print("Block size not implemented for stiffness matrix")
        return

    # FIXME: Once ffcx updated: change quadrature_degree -1 to quadrature_degree
    Aref = reference_code(V, quadrature_degree - 1)
    A = assemble_matrix(V, quadrature_degree, int_type=integral_type)
    ai, aj, av = Aref.getValuesCSR()
    Aref_sp = scipy.sparse.csr_matrix((av, aj, ai))
    matrix_error = scipy.sparse.linalg.norm(Aref_sp - A)
    print(f"Matrix error {matrix_error}")
    assert(matrix_error < 1e-13)
