# Copyright (C) 2021 Jørgen S. Dokken, Igor Baratta
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import argparse
import time

import dolfinx
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import ufl
from dolfinx_assemblers import (assemble_mass_matrix,
                                compute_reference_mass_matrix, estimate_max_polynomial_degree)
from mpi4py import MPI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom assembler of mass matrix using numba and Basix",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num-runs", default=5, type=np.int32, dest="runs",
                        help="Number of times to run the assembler")
    parser.add_argument("--degree", default=1, type=int, dest="degree",
                        help="Degree of Lagrange finite element space")
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

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    if threed:
        if simplex:
            ct = dolfinx.cpp.mesh.CellType.tetrahedron
        else:
            ct = dolfinx.cpp.mesh.CellType.hexahedron
        N = 30
        mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, N, N, N, cell_type=ct)

    else:
        if simplex:
            ct = dolfinx.cpp.mesh.CellType.triangle
        else:
            ct = dolfinx.cpp.mesh.CellType.quadrilateral
        N = 500
        mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, N, N, cell_type=ct)

    cell_str = dolfinx.cpp.mesh.to_string(mesh.topology.cell_type)
    el = ufl.VectorElement("CG", cell_str, degree) if vector else ufl.FiniteElement("CG", cell_str, degree)

    V = dolfinx.FunctionSpace(mesh, el)
    a_mass = ufl.inner(ufl.TrialFunction(V), ufl.TestFunction(V)) * ufl.dx
    quadrature_degree = quadrature_degree = estimate_max_polynomial_degree(a_mass) + 1

    dolfin_times = np.zeros(runs - 1)
    numba_times = np.zeros(runs - 1)
    for i in range(runs):
        start = time.time()
        Aref = compute_reference_mass_matrix(V, quadrature_degree)
        end = time.time()
        print(f"{i}: DOLFINx {end-start:.2e}")
        if i > 0:
            dolfin_times[i - 1] = end - start
        start = time.time()
        A = assemble_mass_matrix(V, quadrature_degree)
        end = time.time()
        if i > 0:
            numba_times[i - 1] = end - start

        print(f"{i}: Numba {end-start:.2e}")
    print(f"num dofs {V.dofmap.index_map.size_local}",
          f"numba/dolfin: {np.sum(numba_times) / np.sum(dolfin_times)}")
    if verbose:
        print(f"Reference:\n {Aref[:,:]}")
        print(f"Solution:\n {A.toarray()}")
    ai, aj, av = Aref.getValuesCSR()
    Aref_sp = scipy.sparse.csr_matrix((av, aj, ai))
    matrix_error = scipy.sparse.linalg.norm(Aref_sp - A)
    print(f"Norm of matrix error {matrix_error}")
