# Copyright (C) 2021 Sarah Roggendorf
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import argparse
import time

import dolfinx
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import ufl
from dolfinx_assemblers import (assemble_matrix,
                                compute_reference_mass_matrix, estimate_max_polynomial_degree)
from mpi4py import MPI



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom assembler of stiffness matrix using numba and Basix",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num-runs", default=2, type=np.int32, dest="runs",
                        help="Number of times to run the assembler")
    parser.add_argument("--degree", default=1, type=int, dest="degree",
                        help="Degree of Lagrange finite element space")
    _threed = parser.add_mutually_exclusive_group(required=False)
    _threed.add_argument('--3D', dest='threed', action='store_true',
                          help="Use quadrilateral mesh", default=False)

    _verbose = parser.add_mutually_exclusive_group(required=False)
    _verbose.add_argument('--verbose', dest='verbose', action='store_true',
                          help="Print matrices", default=False)

    args = parser.parse_args()
    threed = args.threed
    runs = args.runs
    verbose = args.verbose
    degree = args.degree

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    if threed:
        x = np.array([[0, 0, 0], [1.1, 0, 0], [0.3, 1.0, 0], [2, 1., 0]])
        cells = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int32)

        ufl_mesh = ufl.Mesh(ufl.VectorElement("Lagrange", "triangle", 1))
    else:
        x = np.array([[0, 0], [1.1, 0], [0.3, 1.0], [2, 1.5]])
        cells = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int32)

        ufl_mesh = ufl.Mesh(ufl.VectorElement("Lagrange", "triangle", 1))

    mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, x, ufl_mesh)


    el = ufl.FiniteElement("CG", 'triangle', degree)
    V = dolfinx.FunctionSpace(mesh, el)
    a_ = ufl.inner(ufl.TrialFunction(V), ufl.TestFunction(V)) * ufl.dx
    quadrature_degree = estimate_max_polynomial_degree(a_) + 1

    dolfin_times = np.zeros(runs - 1)
    numba_times = np.zeros(runs - 1)
    jit_parameters = {"cffi_extra_compile_args": ["-Ofast", "-march=native"], "cffi_verbose": False}
    for i in range(runs):
        start = time.time()
        # FIXME: Once ffcx updated: change quadrature_degree -1 to quadrature_degree
        Aref = compute_reference_mass_matrix(V, quadrature_degree - 1, jit_parameters)
        end = time.time()
        print(f"{i}: DOLFINx {end-start:.2e}")
        if i > 0:
            dolfin_times[i - 1] = end - start
        start = time.time()
        A = assemble_matrix(V, quadrature_degree, "mass")
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




