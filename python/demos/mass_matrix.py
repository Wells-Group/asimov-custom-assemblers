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
                                compute_reference_mass_matrix)
from mpi4py import MPI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom assembler of mass matrix using numba and Basix",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num-runs", default=5, type=np.int32, dest="runs",
                        help="Number of times to run the assembler")
    parser.add_argument("--degree", default=1, type=int, dest="degree",
                        help="Degree of Lagrange finite element space")
    _ct = parser.add_mutually_exclusive_group(required=False)
    _ct.add_argument('--quad', dest='quad', action='store_true',
                     help="Use quadrilateral mesh", default=False)
    _verbose = parser.add_mutually_exclusive_group(required=False)
    _verbose.add_argument('--verbose', dest='verbose', action='store_true',
                          help="Print matrices", default=False)

    args = parser.parse_args()
    quad = args.quad
    runs = args.runs
    verbose = args.verbose
    degree = args.degree

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    ct = dolfinx.cpp.mesh.CellType.quadrilateral if quad else dolfinx.cpp.mesh.CellType.triangle
    N = 500
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, N, N, cell_type=ct)

    cell_str = "quadrilateral" if quad else "triangle"
    el = ufl.FiniteElement("CG", cell_str, degree)
    quadrature_degree = 2 * el.degree() + 1
    V = dolfinx.FunctionSpace(mesh, el)
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
    print(f"numba/dolfin: {np.sum(numba_times) / np.sum(dolfin_times)}")
    if verbose:
        print(f"Reference:\n {Aref[:,:]}")
        print(f"Solution:\n {A.toarray()}")
    ai, aj, av = Aref.getValuesCSR()
    Aref_sp = scipy.sparse.csr_matrix((av, aj, ai))
    matrix_error = scipy.sparse.linalg.norm(Aref_sp - A)
    print(f"Norm of matrix error {matrix_error}")
