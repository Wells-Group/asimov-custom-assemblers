# Copyright (C) 2021 Jørgen S. Dokken, Igor Baratta
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np
import numba
import ufl
import dolfinx
"""
Utilities for assembly
"""

__all__ = ["estimate_max_polynomial_degree"]


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
    return facet_info


@numba.njit(fastmath=True, cache=True)
def pack_facet_info_numba(active_facets, c_to_f, f_to_c):
    """
    Given a list of external facets get the owning cell and local facet
    index
    """
    facet_info = np.zeros((len(active_facets), 3), dtype=np.int64)
    c_to_f_pos, c_to_f_offs = c_to_f
    f_to_c_pos, f_to_c_offs = f_to_c

    for j, facet in enumerate(active_facets):
        cells = f_to_c_pos[f_to_c_offs[facet]:f_to_c_offs[facet + 1]]
        assert(len(cells) == 1)
        local_facets = c_to_f_pos[c_to_f_offs[cells[0]]: c_to_f_offs[cells[0] + 1]]
        # Should be wrapped in convenience numba function
        local_index = np.flatnonzero(facet == local_facets)[0]
        facet_info[j, :] = [facet, cells[0], local_index]
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
