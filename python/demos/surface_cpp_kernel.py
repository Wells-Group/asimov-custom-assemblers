# Copyright (C) 2021 Sarah Roggendorf
#
# SPDX-License-Identifier:   LGPL-3.0-or-later

import argparse
import time

import dolfinx
import dolfinx_cuas
import dolfinx_cuas.cpp
import numpy as np
import scipy.sparse
import ufl
from mpi4py import MPI
from petsc4py import PETSc

kt = dolfinx_cuas.cpp.Kernel


def compare_matrices(A: PETSc.Mat, B: PETSc.Mat, atol: float = 1e-13):
    """
    Helper for comparing two PETSc matrices
    """
    # Create scipy CSR matrices
    ai, aj, av = A.getValuesCSR()
    A_sp = scipy.sparse.csr_matrix((av, aj, ai))
    bi, bj, bv = B.getValuesCSR()
    B_sp = scipy.sparse.csr_matrix((bv, bj, bi))
    # Compare matrices
    diff = np.abs(A_sp - B_sp)
    assert diff.max() <= atol


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom assembler of mass matrix using numba and Basix",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num-runs", default=5, type=np.int32, dest="runs",
                        help="Number of times to run the assembler")
    parser.add_argument("--degree", default=1, type=int, dest="degree",
                        help="Degree of Lagrange finite element space")
    parser.add_argument("--kernel", default="mass", type=str, dest="kernel_str",
                        help="Type of kernel. Choose from 'mass', 'stiffness', 'symgrad'")
    _simplex = parser.add_mutually_exclusive_group(required=False)
    _simplex.add_argument('--simplex', dest='simplex', action='store_true',
                          help="Use simplex mesh", default=False)
    _2D = parser.add_mutually_exclusive_group(required=False)
    _2D.add_argument('--3D', dest='threed', action='store_true', help="Use 3D mesh", default=False)
    _verbose = parser.add_mutually_exclusive_group(required=False)
    _verbose.add_argument('--verbose', dest='verbose', action='store_true',
                          help="Print matrices", default=False)
    _vector = parser.add_mutually_exclusive_group(required=False)
    _vector.add_argument('--vector', dest='vector', action='store_true',
                         help="Use vector finite elements", default=False)

    args = parser.parse_args()
    simplex = args.simplex
    threed = args.threed
    runs = args.runs
    verbose = args.verbose
    degree = args.degree
    vector = args.vector
    kernel_str = args.kernel_str

    # N = 10 if threed else 30
    # mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, N, N, N) if threed else dolfinx.UnitSquareMesh(MPI.COMM_WORLD, N, N)
    if not threed and not simplex:
        print('yay')
        x = np.array([[0, 0, 0], [0, 0, 1], [0.1, 0.7, 0.9], [0.2, 0.8, 0.1]])
        cells = np.array([[0, 1, 2, 3]], dtype=np.int32)
        ct = "quadrilateral"
    elif not threed and simplex:
        x = np.array([[0, 0, 0.5], [1.1, 0, 1.3], [0.0, 1.0, 0.9], [0.0, 1.5, 0.0]])
        cells = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int32)
        ct = "triangle"
    elif threed and simplex:
        x = np.array([[0, 0, 0], [1.1, 0, 0], [0.3, 1.0, 0], [1, 1.2, 1.5], [2, 2, 1.5]])
        cells = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
        ct = "tetrahedron"
    elif threed and not simplex:
        # x = np.array([[0, 0, 0], [0, 1, 0], [0, 0.2, 0.8], [0, 0.9, 0.7],
        #              [0.7, 0.1, 0.2], [0.9, 0.9, 0.1], [0.8, 0.1, 0.9], [1, 1, 1]])
        x = np.array([[0, 0, 0], [0.9, 0, 0], [0, 0.9, 0], [0.9, 0.9, 0],
                     [0, 0, 0.9], [0.9, 0, 0.9], [0, 0.9, 0.9], [0.9, 0.9, 0.9]])
        cells = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)
        ct = "hexahedron"

    cell = ufl.Cell(ct, geometric_dimension=x.shape[1])
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 1))
    mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, x, domain)
    el = ufl.FiniteElement("CG", mesh.ufl_cell(), degree)
    V = dolfinx.FunctionSpace(mesh, el)

    # Find facets on boundary to integrate over
    facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1,
                                                   lambda x: np.isclose(x[0], 0.0))
    values = np.ones(len(facets), dtype=np.int32)
    ft = dolfinx.MeshTags(mesh, mesh.topology.dim - 1, facets, values)

    # Define variational form
    V = dolfinx.VectorFunctionSpace(mesh, ("CG", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)
    if kernel_str == "mass":
        kernel_type = kt.Mass
        a = ufl.inner(u, v) * ds(1)
    elif kernel_str == "stiffness":
        kernel_type = kt.Stiffness
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ds(1)
    elif kernel_str == "symgrad":
        kernel_type = kt.SymGrad

        def epsilon(v):
            return ufl.sym(ufl.grad(v))
        a = ufl.inner(epsilon(u), epsilon(v)) * ds(1)
    else:
        raise RuntimeError("Unknown kernel")
    quadrature_degree = dolfinx_cuas.estimate_max_polynomial_degree(a)
    # Compile UFL form
    cffi_options = ["-Ofast", "-march=native"]
    a = dolfinx.fem.Form(a, jit_parameters={"cffi_extra_compile_args": cffi_options, "cffi_libraries": ["m"]})
    A = dolfinx.fem.create_matrix(a)

    # Normal assembly
    A.zeroEntries()
    dolfinx.fem.assemble_matrix(A, a)
    A.assemble()

    # Custom assembly
    B = dolfinx.fem.create_matrix(a)

    kernel = dolfinx_cuas.cpp.generate_surface_kernel(V._cpp_object, kernel_type, quadrature_degree)
    B.zeroEntries()
    dolfinx_cuas.cpp.assemble_exterior_facets(B, a._cpp_object, ft.indices, kernel)
    B.assemble()

    # Compare matrices, first norm, then entries
    # assert np.isclose(A.norm(), B.norm())
    compare_matrices(A, B)
