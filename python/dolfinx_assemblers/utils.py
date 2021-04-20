# Copyright (C) 2021 Jørgen S. Dokken, Igor Baratta
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np
import numba

"""
Utilities for assembly
"""


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

# TODO: Finalize this to use instead of numpy.linalg.inv
def compute_inverse(A: np.ndarray, Ainv: np.ndarray, detJ: np.ndarray):
    """
    Compute the inverse of A matrix with max dimension 3 on any axis
    """
    num_rows = A.shape[0]
    num_cols = A.shape[1]
    if num_rows == num_cols:
        if num_rows == 1:
            detJ = A[0]
            Ainv[0] = 1./A[0]
        elif num_rows == 2:
            detJ[0] = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
            Ainv[0,0] = A[1,1]/detJ[0]
            Ainv[0,1] = -A[0,1]/detJ[0]
            Ainv[1,0] = -A[1,0]/detJ[0]
            Ainv[1,1] = A[0,0]/detJ[0]
        elif num_rows == 3:
            detJ[0] = A[0, 0] * A[1, 1] * A[2, 2] + A[0, 1] * A[1, 2] * A[2, 0]\
                + A[0, 2] * A[1, 0] * A[2, 1] - A[2, 0] * A[1, 1] * A[0, 2]\
                - A[2, 1] * A[1, 2] * A[0, 0] - A[2, 2] * A[1, 0] * A[0, 1]
        else:
            # print(f"Matrix has invalid size {num_rows}x{num_cols}")
            assert(False)
    else:
        # print(f"Matrix has invalid size {num_rows}x{num_cols}")
        assert(False)
