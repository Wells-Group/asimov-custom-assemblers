import dolfinx
import numpy as np
import pytest
import scipy.sparse
import scipy.sparse.linalg
import ufl.algorithms
import ufl
from dolfinx_assemblers import (assemble_mass_matrix,
                                compute_reference_mass_matrix, estimate_max_polynomial_degree)
from mpi4py import MPI


@pytest.mark.parametrize("degree", range(1, 5))
@pytest.mark.parametrize("ct", ["quadrilateral", "triangle", "tetrahedron",
                                "hexahedron"])
@pytest.mark.parametrize("element", [ufl.FiniteElement, ufl.VectorElement])
def test_mass_matrix(element, ct, degree):
    """
    Test assembly of mass matrices on non-affine mesh
    """
    cell_type = dolfinx.cpp.mesh.to_type(ct)
    if cell_type == dolfinx.cpp.mesh.CellType.quadrilateral:
        x = np.array([[0, 0], [1, 0], [0, 1.3], [1.2, 1]])
        cells = np.array([[0, 1, 2, 3]], dtype=np.int32)
        ufl_mesh = ufl.Mesh(ufl.VectorElement("Lagrange", "quadrilateral", 1))
    elif cell_type == dolfinx.cpp.mesh.CellType.triangle:
        x = np.array([[0, 0], [1.1, 0], [0.3, 1.0], [2, 1.5]])
        cells = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int32)

        ufl_mesh = ufl.Mesh(ufl.VectorElement("Lagrange", "triangle", 1))
    elif cell_type == dolfinx.cpp.mesh.CellType.tetrahedron:
        x = np.array([[0, 0, 0], [1.1, 0, 0], [0.3, 1.0, 0], [1, 1.2, 1.5], [2, 2, 1.5]])
        cells = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
        ufl_mesh = ufl.Mesh(ufl.VectorElement("Lagrange", "tetrahedron", 1))
    elif cell_type == dolfinx.cpp.mesh.CellType.hexahedron:
        x = np.array([[0, 0, 0], [1.1, 0, 0], [0.1, 1, 0], [1, 1.2, 0],
                      [0, 0, 1.2], [1.0, 0, 1], [0, 1, 1], [1, 1, 1]])
        cells = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)
        ufl_mesh = ufl.Mesh(ufl.VectorElement("Lagrange", "hexahedron", 1))
    else:
        raise ValueError(f"Unsupported mesh type {ct}")
    mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, x, ufl_mesh)
    el = element("CG", ct, degree)
    V = dolfinx.FunctionSpace(mesh, el)

    a_mass = ufl.inner(ufl.TrialFunction(V), ufl.TestFunction(V)) * ufl.dx
    quadrature_degree = estimate_max_polynomial_degree(a_mass) + 1
    Aref = compute_reference_mass_matrix(V)
    A = assemble_mass_matrix(V, quadrature_degree)
    ai, aj, av = Aref.getValuesCSR()
    Aref_sp = scipy.sparse.csr_matrix((av, aj, ai))
    matrix_error = scipy.sparse.linalg.norm(Aref_sp - A)
    assert(matrix_error < 1e-13)
