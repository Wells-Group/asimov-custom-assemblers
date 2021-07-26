import dolfinx
import dolfinx_cuas.cpp
import dolfinx.io
import dolfinx.log
import numpy as np
import ufl
from mpi4py import MPI
import scipy.sparse
from petsc4py import PETSc

kt = dolfinx_cuas.cpp.Kernel


def compare_matrices(A: PETSc.Mat, B: PETSc.Mat, atol: float = 1e-12):
    """
    Helper for comparing two PETSc matrices
    """
    # Create scipy CSR matrices
    ai, aj, av = A.getValuesCSR()
    A_sp = scipy.sparse.csr_matrix((av, aj, ai), shape=A.getSize())
    bi, bj, bv = B.getValuesCSR()
    B_sp = scipy.sparse.csr_matrix((bv, bj, bi), shape=B.getSize())

    # Compare matrices
    diff = np.abs(A_sp - B_sp)
    assert diff.max() <= atol


def test_volume_kernels(kernel_type, P):
    N = 4
    mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, N, N, N)
    # Define variational form
    V = dolfinx.FunctionSpace(mesh, ("CG", P))
    bs = V.dofmap.index_map_bs

    def f(x):
        values = np.zeros((1, x.shape[1]))
        for i in range(x.shape[1]):
            values[0, i] = np.max(np.abs(x[:, i]))
        return values

    V2 = dolfinx.FunctionSpace(mesh, ("DG", 0))
    mu = dolfinx.Function(V2)
    mu.interpolate(f)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh)
    if kernel_type == kt.Mass:
        a = mu * ufl.inner(u, v) * dx
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
    kernel = dolfinx_cuas.cpp.generate_coeff_kernel(kernel_type, P, bs)
    B.zeroEntries()
    consts = np.zeros(0)
    coeffs = mu.vector.array.reshape(num_local_cells, 1)
    dolfinx_cuas.cpp.assemble_cells(B, V._cpp_object, active_cells, kernel, coeffs, consts)
    B.assemble()

    # Compare matrices, first norm, then entries
    assert np.isclose(A.norm(), B.norm())
    compare_matrices(A, B)


if __name__ == "__main__":
    test_volume_kernels(kt.Mass, 2)
