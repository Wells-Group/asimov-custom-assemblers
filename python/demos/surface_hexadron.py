# Copyright (C) 2021 Sarah Roggendorf
#
# SPDX-License-Identifier:   LGPL-3.0-or-later

import argparse
import time

import dolfinx
import dolfinx_cuas
import dolfinx.io
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
    parser.add_argument("--degree", default=1, type=int, dest="degree",
                        help="Degree of Lagrange finite element space")
    _vector = parser.add_mutually_exclusive_group(required=False)
    _vector.add_argument('--vector', dest='vector', action='store_true',
                         help="Use vector finite elements", default=False)

    args = parser.parse_args()
    degree = args.degree
    vector = args.vector

    x = np.array([[0, 0, 0], [0, 1, 0], [0, 0.2, 0.8], [0, 0.9, 0.7],
                 [0.7, 0.1, 0.2], [0.9, 0.9, 0.1], [0.8, 0.1, 0.9], [1, 1, 1]])
    # x = np.array([[0, 0, 0], [0.9, 0, 0], [0, 1.0, 0], [0.9, 0.9, 0],
    #              [0, 0, 0.9], [0.9, 0, 0.9], [0, 0.9, 0.9], [0.9, 0.9, 0.9]])
    cells = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)
    ct = "hexahedron"

    cell = ufl.Cell(ct, geometric_dimension=x.shape[1])
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 1))
    mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, x, domain)
    el = ufl.VectorElement("CG", mesh.ufl_cell(), degree) if vector \
        else ufl.FiniteElement("CG", mesh.ufl_cell(), degree)
    V = dolfinx.FunctionSpace(mesh, el)

    # Find facets on boundary to integrate over
    facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1,
                                                   lambda x: np.isclose(x[0], 0.0))

    values = np.ones(len(facets), dtype=np.int32)
    ft = dolfinx.MeshTags(mesh, mesh.topology.dim - 1, facets, values)

    # Define variational form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)
    a = ufl.inner(u, v) * ds(1)
    quadrature_degree = dolfinx_cuas.estimate_max_polynomial_degree(a)

    # For debugging
    # with dolfinx.io.XDMFFile(MPI.COMM_WORLD, 'hexahedron.xdmf', "w") as xdmf:
    #     xdmf.write_mesh(mesh)

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

    kernel = dolfinx_cuas.cpp.generate_surface_kernel(V._cpp_object, kt.MassNonAffine, quadrature_degree)
    B.zeroEntries()
    dolfinx_cuas.cpp.assemble_exterior_facets(B, a._cpp_object, ft.indices, kernel)
    B.assemble()

    # Compare matrices, first norm, then entries
    assert np.isclose(A.norm(), B.norm())
    compare_matrices(A, B)
