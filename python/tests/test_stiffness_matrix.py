import dolfinx
import numpy as np
import pytest
import scipy.sparse
import scipy.sparse.linalg
import ufl
from dolfinx_assemblers import (assemble_stiffness_matrix,
                                compute_reference_stiffness_matrix)
from mpi4py import MPI
# np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


# @pytest.mark.parametrize("degree", [1, 2])
# @pytest.mark.parametrize("ct", ["quadrilateral", "triangle"])
@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("ct", ["triangle"])
def test_stiffness_matrix(ct, degree):
    """
    Test assembly of stiffness matrices on non-affine mesh
    """
    if dolfinx.cpp.mesh.to_type(ct) == dolfinx.cpp.mesh.CellType.quadrilateral:
        x = np.array([[0, 0], [1, 0], [0, 1.3], [1.2, 1]])
        cells = np.array([[0, 1, 2, 3]], dtype=np.int32)
        ufl_mesh = ufl.Mesh(ufl.VectorElement("Lagrange", "quadrilateral", 1))
    else:
        x = np.array([[0.1, 0], [1.1, 0.1], [0.0, 1.0], [2, 1.5]])
        cells = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
        ufl_mesh = ufl.Mesh(ufl.VectorElement("Lagrange", "triangle", 1))

    mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, x, ufl_mesh)
    el = ufl.FiniteElement("CG", ct, degree)

    V = dolfinx.FunctionSpace(mesh, el)
    quadrature_degree = 1
    Aref = compute_reference_stiffness_matrix(V, quadrature_degree)
    A = assemble_stiffness_matrix(V, quadrature_degree)
    ai, aj, av = Aref.getValuesCSR()
    Aref_sp = scipy.sparse.csr_matrix((av, aj, ai))
    matrix_error = scipy.sparse.linalg.norm(Aref_sp - A)
    assert(matrix_error < 1e-13)
