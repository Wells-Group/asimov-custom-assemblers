import dolfinx
import dolfinx_cuas
import dolfinx_cuas.cpp
import numpy as np
import pytest
import scipy.sparse
import ufl
from mpi4py import MPI
from petsc4py import PETSc

kt = dolfinx_cuas.cpp.Kernel


def compare_matrices(A: PETSc.Mat, B: PETSc.Mat, atol: float = 1e-13):
    ai, aj, av = A.getValuesCSR()
    A_sp = scipy.sparse.csr_matrix((av, aj, ai))
    bi, bj, bv = B.getValuesCSR()
    B_sp = scipy.sparse.csr_matrix((bv, bj, bi))

    print(A.norm(), B.norm())

    # Compare matrices
    diff = np.abs(A_sp - B_sp)

    assert diff.max() <= atol


# @pytest.mark.parametrize("kernel_type", [kt.Mass, kt.Stiffness, kt.Contact_Jac])
@pytest.mark.parametrize("dim", [2])
@pytest.mark.parametrize("kernel_type", [kt.Contact_Jac])
def test_sum_grad(dim, kernel_type):
    if dim == 2:
        N = 10
        mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, N, N)
    else:
        N = 5
        mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, N, N, N)

    V = dolfinx.VectorFunctionSpace(mesh, ("CG", 1))

    facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1,
                                                   lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                                                           np.isclose(x[0], 1.0)))

    values = np.ones(len(facets), dtype=np.int32)
    ft = dolfinx.MeshTags(mesh, mesh.topology.dim - 1, facets, values)

    def epsilon(v):
        return ufl.sym(ufl.grad(v))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)
    if kernel_type == kt.Mass:
        a = ufl.inner(u, v) * ds(1)
    elif kernel_type == kt.Stiffness:
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ds(1)
    elif kernel_type == kt.Contact_Jac:
        a = ufl.inner(epsilon(u), epsilon(v)) * ds(1)
    else:
        raise RuntimeError("Unknown kernel")

    a = dolfinx.fem.Form(a, jit_parameters={"cffi_extra_compile_args": ["-Ofast", "-march=native"],
                                            "cffi_libraries": ["m"]})
    A = dolfinx.fem.create_matrix(a)
    A.zeroEntries()

    # Normal assembly
    dolfinx.fem.assemble_matrix(A, a)
    A.assemble()
    print("!!!!----!!!!")
    # Custom assembly
    B = dolfinx.fem.create_matrix(a)
    B.zeroEntries()
    contact = dolfinx_cuas.cpp.Contact(ft, 1, 1, V._cpp_object)
    contact.create_reference_facet_qp()
    kernel = contact.generate_surface_kernel(0, kernel_type)

    dolfinx_cuas.cpp.assemble_exterior_facets(B, a._cpp_object, ft.indices, kernel)
    B.assemble()
    compare_matrices(A, B)
