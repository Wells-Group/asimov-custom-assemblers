# Copyright (C) 2021 Jørgen S. Dokken, Igor Baratta, Sarah Roggendorf
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import dolfinx
import numba
import numpy as np
from petsc4py import PETSc
import scipy.sparse
import ufl

"""
Utilities for assembly
"""

__all__ = ["estimate_max_polynomial_degree",
           "pack_facet_info", "expand_dofmap", "create_csr_sparsity_pattern", "compare_matrices"]


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
    if diff.max() > atol:
        print(diff.max())
    assert diff.max() <= atol


def pack_facet_info(mesh: dolfinx.cpp.mesh.Mesh, mt: dolfinx.MeshTags, index: int):
    """
    Given a mesh, meshtag and an index, compute the triplet
    (facet index (local to process), cell index(local to process), facet index (local to cell) )
    """
    # FIXME: Should be moved to dolfinx C++ layer
    # Set up data required for exterior facet assembly
    tdim = mesh.topology.dim
    fdim = mesh.topology.dim - 1

    mesh.topology.create_connectivity(tdim, fdim)
    mesh.topology.create_connectivity(fdim, tdim)
    c_to_f = mesh.topology.connectivity(tdim, fdim)
    f_to_c = mesh.topology.connectivity(fdim, tdim)

    assert(mt.dim == fdim)
    active_facets = mt.indices[mt.values == index]
    facet_info = pack_facet_info_numba(active_facets,
                                       (c_to_f.array, c_to_f.offsets),
                                       (f_to_c.array, f_to_c.offsets))
    g_indices = dolfinx.cpp.mesh.entities_to_geometry(mesh, fdim,
                                                      np.array(active_facets, dtype=np.int32),
                                                      False)
    return facet_info, g_indices


@numba.njit(fastmath=True, cache=True)
def pack_facet_info_numba(active_facets, c_to_f, f_to_c):
    """
    Given a list of external facets get the owning cell and local facet
    index
    """
    facet_info = np.zeros((len(active_facets), 2), dtype=np.int64)
    c_to_f_pos, c_to_f_offs = c_to_f
    f_to_c_pos, f_to_c_offs = f_to_c

    for j, facet in enumerate(active_facets):
        cells = f_to_c_pos[f_to_c_offs[facet]:f_to_c_offs[facet + 1]]
        assert(len(cells) == 1)
        local_facets = c_to_f_pos[c_to_f_offs[cells[0]]: c_to_f_offs[cells[0] + 1]]
        # Should be wrapped in convenience numba function
        local_index = np.flatnonzero(facet == local_facets)[0]
        facet_info[j, :] = [cells[0], local_index]
    return facet_info


@numba.njit(cache=True)
def expand_dofmap(dofmap: np.ndarray, block_size: int, expanded_dofmap: np.ndarray):
    """
    Expand dofmap for a given block size
    """
    num_cells, num_dofs_per_cell = dofmap.shape
    for i in range(num_cells):
        for j in range(num_dofs_per_cell):
            for k in range(block_size):
                expanded_dofmap[i, j * block_size + k] = dofmap[i, j] * block_size + k


def create_csr_sparsity_pattern(num_cells: int, num_dofs_per_cell: int, dofmap: np.ndarray):
    """
    Create a csr matrix given a flattened dofmap and the number of cells and dofs per cell
    """
    rows = np.repeat(dofmap, num_dofs_per_cell)
    cols = np.tile(np.reshape(dofmap, (num_cells, num_dofs_per_cell)), num_dofs_per_cell)
    return rows, cols.ravel()


@numba.njit(cache=True)
def compute_determinant(A: np.ndarray, detJ: np.ndarray):
    """
    Compute the determinant of A matrix with max dimension 3 on any axis
    """
    num_rows = A.shape[0]
    num_cols = A.shape[1]
    if num_rows == num_cols:
        if num_rows == 1:
            detJ = A[0]
        elif num_rows == 2:
            detJ[0] = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        elif num_rows == 3:
            detJ[0] = A[0, 0] * A[1, 1] * A[2, 2] + A[0, 1] * A[1, 2] * A[2, 0]\
                + A[0, 2] * A[1, 0] * A[2, 1] - A[2, 0] * A[1, 1] * A[0, 2]\
                - A[2, 1] * A[1, 2] * A[0, 0] - A[2, 2] * A[1, 0] * A[0, 1]
        else:
            # print(f"Matrix has invalid size {num_rows}x{num_cols}")
            assert(False)
    else:
        # det(A^T A) = det(A) det(A)
        ATA = A.T @ A
        num_rows = ATA.shape[0]
        num_cols = ATA.shape[1]
        if num_rows == 1:
            detJ[0] = ATA[0, 0]
        elif num_rows == 2:
            detJ[0] = ATA[0, 0] * ATA[1, 1] - ATA[0, 1] * ATA[1, 0]
        elif num_rows == 3:
            detJ[0] = ATA[0, 0] * ATA[1, 1] * ATA[2, 2] + ATA[0, 1] * ATA[1, 2] * ATA[2, 0]\
                + ATA[0, 2] * ATA[1, 0] * ATA[2, 1] - ATA[2, 0] * ATA[1, 1] * ATA[0, 2]\
                - ATA[2, 1] * ATA[1, 2] * ATA[0, 0] - ATA[2, 2] * ATA[1, 0] * ATA[0, 1]
        else:
            # print(f"Matrix has invalid size {num_rows}x{num_cols}")
            assert(False)
        detJ[0] = np.sqrt(detJ[0])


@numba.njit(cache=True)
def square_inv(A: np.ndarray, Ainv: np.ndarray, detJ: np.ndarray):
    """
    Compute the inverse of A square matrix (1x1, 2x2, 3x3 only)
    """
    num_rows = A.shape[0]
    num_cols = A.shape[1]
    if num_rows == num_cols:
        if num_rows == 1:
            detJ = A[0]
            Ainv[0] = 1. / A[0]
        elif num_rows == 2:
            detJ[0] = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
            Ainv[0, 0] = A[1, 1] / detJ[0]
            Ainv[0, 1] = -A[0, 1] / detJ[0]
            Ainv[1, 0] = -A[1, 0] / detJ[0]
            Ainv[1, 1] = A[0, 0] / detJ[0]
        elif num_rows == 3:
            detJ[0] = A[0, 0] * A[1, 1] * A[2, 2] + A[0, 1] * A[1, 2] * A[2, 0]\
                + A[0, 2] * A[1, 0] * A[2, 1] - A[2, 0] * A[1, 1] * A[0, 2]\
                - A[2, 1] * A[1, 2] * A[0, 0] - A[2, 2] * A[1, 0] * A[0, 1]
            Ainv[0, 0] = (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]) / detJ[0]
            Ainv[0, 1] = -(A[0, 1] * A[2, 2] - A[0, 2] * A[2, 1]) / detJ[0]
            Ainv[0, 2] = (A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]) / detJ[0]
            Ainv[1, 0] = -(A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0]) / detJ[0]
            Ainv[1, 1] = (A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0]) / detJ[0]
            Ainv[1, 2] = -(A[0, 0] * A[1, 2] - A[0, 2] * A[1, 0]) / detJ[0]
            Ainv[2, 0] = (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]) / detJ[0]
            Ainv[2, 1] = -(A[0, 0] * A[2, 1] - A[0, 1] * A[2, 0]) / detJ[0]
            Ainv[2, 2] = (A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]) / detJ[0]
        else:
            # print(f"Matrix has invalid size {num_rows}x{num_cols}")
            assert(False)
    else:
        # print(f"Matrix has invalid size {num_rows}x{num_cols}")
        assert(False)


@numba.njit(cache=True)
def compute_inverse(A: np.ndarray, Ainv: np.ndarray, detJ: np.ndarray):
    """
    Compute the inverse of A matrix with max dimension 3 on any axis
    """
    num_rows = A.shape[0]
    num_cols = A.shape[1]
    if num_rows == num_cols:
        square_inv(A, Ainv, detJ)
    else:
        # Moore Penrose Pseudo inverse A^{-1} = (A^T A)^{-1} A^T
        AT = A.T.copy()
        ATA = AT @ A
        num_rows = ATA.shape[0]
        num_cols = ATA.shape[1]
        ATAinv = np.zeros((num_rows, num_cols), dtype=np.float64)
        square_inv(ATA, ATAinv, detJ)
        Ainv[:] = ATAinv @ AT
        detJ[0] = np.sqrt(detJ[0])


def estimate_max_polynomial_degree(form: ufl.form.Form) -> int:
    """
    Estimate the maximum polynomial degree in a ufl form (including variations in the determinant)
    """
    form_data = ufl.algorithms.compute_form_data(
        form, do_apply_function_pullbacks=True, do_apply_integral_scaling=True, do_apply_geometry_lowering=True)
    pol_degrees = []
    for i in range(len(form_data.integral_data)):
        for j in range(len(form_data.integral_data[i].integrals)):
            pol_degrees.append(form_data.integral_data[0].integrals[0].metadata()['estimated_polynomial_degree'])
    return np.max(pol_degrees)
