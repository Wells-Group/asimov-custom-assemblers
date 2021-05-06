# Copyright (C) 2021 Jørgen S. Dokken, Igor Baratta
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numba
import numpy as np
from basix.numba_helpers import (apply_dof_transformation_hexahedron,
                                 apply_dof_transformation_quadrilateral,
                                 apply_dof_transformation_tetrahedron,
                                 apply_dof_transformation_triangle)
from numba.typed import Dict

from .utils import compute_determinant, compute_inverse


@numba.njit(fastmath=True)
def mass_kernel(data: np.ndarray, num_cells: int, num_dofs_per_cell: int, num_dofs_x: int, x_dofs: np.ndarray,
                x: np.ndarray, gdim: int, tdim: int, c_tab: np.ndarray, q_p: np.ndarray, q_w: np.ndarray,
                phi: np.ndarray, is_affine: bool, e_transformations: Dict, e_dofs: Dict, ct: str, cell_perm: int,
                needs_transformations: bool, block_size: int):
    """
    Assemble mass matrix into CSR array "data"
    """

    # Declaration of local structures
    geometry = np.zeros((num_dofs_x, gdim), dtype=np.float64)
    num_q_points = q_w.size
    if ct == "triangle":
        apply_dof_trans = apply_dof_transformation_triangle
    elif ct == "quadrilateral":
        apply_dof_trans = apply_dof_transformation_quadrilateral
    elif ct == "tetrahedron":
        apply_dof_trans = apply_dof_transformation_tetrahedron
    elif ct == "hexahedron":
        apply_dof_trans = apply_dof_transformation_hexahedron
    else:
        assert(False)
    J_q = np.zeros((num_q_points, gdim, tdim), dtype=np.float64)
    detJ_q = np.zeros((num_q_points, 1), dtype=np.float64)
    dphi_c = c_tab[1:gdim + 1, 0, :, 0].copy()
    detJ = np.zeros(1, dtype=np.float64)
    entries_per_cell = (block_size * num_dofs_per_cell)**2
    # Assemble matrix
    Ae = np.zeros((block_size * num_dofs_per_cell, block_size * num_dofs_per_cell))
    blocks = [np.arange(b, block_size * num_dofs_per_cell + b, block_size) for b in range(block_size)]

    for cell in range(num_cells):
        for j in range(num_dofs_x):
            geometry[j] = x[x_dofs[cell, j], : gdim]

        # Compute Jacobian at each quadrature point
        if is_affine:
            J_q[0] = np.dot(geometry.T, dphi_c.T)
            compute_determinant(J_q[0], detJ)
            detJ_q[:] = detJ[0]
        else:
            for i, q in enumerate(q_p):
                dphi_c[:] = c_tab[1:gdim + 1, i, :, 0]
                J_q[i] = geometry.T @ dphi_c.T
                compute_determinant(J_q[i], detJ)
                detJ_q[i] = detJ[0]

        if needs_transformations:
            # Transpose phi before applying dof transformations (ndofs, nquadpoints)
            phi_ = phi.T.copy()
            apply_dof_trans(e_transformations, e_dofs, phi_, num_q_points, cell_perm[cell])
            # Reshape output as the transpose of the phi, i.e. (basis_function, quadrature_point)
            phi_T = phi_.copy()
        else:
            phi_T = phi.T.copy()
        phi_s = (phi_T.T * q_w) * np.abs(detJ_q)
        # Compute weighted basis functions at quadrature points
        # Compute Ae_(i,j) = sum_(s=1)^len(q_w) w_s phi_j(q_s) phi_i(q_s) |det(J(q_s))|
        kernel = phi_T @ phi_s
        # Insert per block size
        for i in range(num_dofs_per_cell):
            for b in range(block_size):
                Ai = Ae[i * block_size + b]
                Ai[blocks[b]] = kernel[i]
        # Add to csr matrix
        data[cell * entries_per_cell: (cell + 1) * entries_per_cell] = np.ravel(Ae)


@numba.njit(fastmath=True)
def stiffness_kernel(data: np.ndarray, num_cells: int, num_dofs_per_cell: int, num_dofs_x: int, x_dofs: np.ndarray,
                     x: np.ndarray, gdim: int, tdim: int, c_tab: np.ndarray, q_p: np.ndarray, q_w: np.ndarray,
                     dphi: np.ndarray, is_affine: bool, e_transformations: Dict, e_dofs: Dict, ct: str,
                     cell_perm: np.ndarray, needs_transformations: bool, block_size: int):
    """
    Assemble stiffness matrix into CSR array "data"
    """

    # Declaration of local structures
    geometry = np.zeros((num_dofs_x, gdim), dtype=np.float64)
    num_q_points = q_w.size
    if ct == "triangle":
        apply_dof_trans = apply_dof_transformation_triangle
    elif ct == "quadrilateral":
        apply_dof_trans = apply_dof_transformation_quadrilateral
    elif ct == "tetrahedron":
        apply_dof_trans = apply_dof_transformation_tetrahedron
    elif ct == "hexahedron":
        apply_dof_trans = apply_dof_transformation_hexahedron
    else:
        assert(False)

    J_q = np.zeros((q_w.size, gdim, tdim), dtype=np.float64)
    invJ = np.zeros((tdim, gdim), dtype=np.float64)
    detJ_q = np.zeros((q_w.size, 1), dtype=np.float64)
    dphi_c = c_tab[1:gdim + 1, 0, :, 0].copy()
    detJ = np.zeros(1, dtype=np.float64)
    entries_per_cell = (block_size * num_dofs_per_cell)**2
    dphi_p = np.zeros((gdim, dphi.shape[2], num_q_points), dtype=np.float64)
    dphi_i = np.zeros((tdim, dphi.shape[2], num_q_points), dtype=np.float64)
    kernel = np.zeros((dphi.shape[2], dphi.shape[2]), dtype=np.float64)

    # Assemble element matrix
    Ae = np.zeros((block_size * num_dofs_per_cell, block_size * num_dofs_per_cell))
    blocks = [np.arange(b, block_size * num_dofs_per_cell + b, block_size) for b in range(block_size)]
    for cell in range(num_cells):
        if needs_transformations:
            # FIXME: Can apply_dof_trans be applied to all dphidxi simultaneously?
            for i in range(tdim):
                # Reshape input as the transpose of the phi, i.e. (basis_function, quadrature_point)
                dphidxi = dphi[i].T.copy()
                apply_dof_trans(e_transformations, e_dofs, dphidxi, num_q_points, cell_perm[cell])
                dphi_i[i, :, :] = dphidxi
        else:
            for i in range(tdim):
                dphi_i[i, :, :] = dphi[i, :, :].T.copy()

        for j in range(num_dofs_x):
            geometry[j] = x[x_dofs[cell, j], : gdim]

        # Compute Jacobian at each quadrature point
        if is_affine:
            dphi_c[:] = c_tab[1:gdim + 1, 0, :, 0]
            J_q[:] = np.dot(geometry.T, dphi_c.T)
            compute_inverse(J_q[0], invJ, detJ)
            detJ_q[:] = detJ[0]
            invJT = invJ.T.copy()
            for p in range(num_q_points):
                dphi_p[:, :, p] = invJT @ dphi_i[:, :, p].copy()
        else:
            for i, q in enumerate(q_p):
                dphi_c[:] = c_tab[1:gdim + 1, i, :, 0]
                J_q[i] = geometry.T @ dphi_c.T
                compute_inverse(J_q[i], invJ, detJ)
                invJT = invJ.T.copy()
                detJ_q[i] = detJ[0]
                dphi_p[:, :, i] = invJT @ dphi_i[:, :, i].copy()

        # Compute weighted basis functions at quadrature points
        scale = q_w * np.abs(detJ_q)
        kernel.fill(0)
        for i in range(gdim):
            dphidxi = dphi_p[i, :, :]
            # Compute Ae_(k,j) += sum_(s=1)^len(q_w) w_s dphi_k/dx_i(q_s) dphi_j/dx_i(q_s) |det(J(q_s))|
            kernel += dphidxi.copy() @ (dphidxi.T * scale)
        # Insert per block size
        for i in range(num_dofs_per_cell):
            for b in range(block_size):
                Ai = Ae[i * block_size + b]
                Ai[blocks[b]] = kernel[i]
        # Add to csr matrix
        data[cell * entries_per_cell: (cell + 1) * entries_per_cell] = np.ravel(Ae)


# @numba.njit


def surface_kernel(data: np.ndarray, num_facets: int, num_dofs_per_cell: int, num_dofs_x: int, x_dofs: np.ndarray,
                   x: np.ndarray, gdim: int, tdim: int, c_tab: np.ndarray, q_p: np.ndarray, q_w: np.ndarray,
                   phi: np.ndarray, is_affine: bool, e_transformations: Dict, e_dofs: Dict, ct: str,
                   cell_perm: np.ndarray, needs_transformations: bool, block_size: int, ref_jacobians: Dict,
                   facet_info: np.ndarray):

    # Declaration of local structures
    geometry = np.zeros((num_dofs_x, gdim), dtype=np.float64)
    num_q_points = q_w.size
    if ct == "triangle":
        apply_dof_trans = apply_dof_transformation_triangle
    elif ct == "quadrilateral":
        apply_dof_trans = apply_dof_transformation_quadrilateral
    elif ct == "tetrahedron":
        apply_dof_trans = apply_dof_transformation_tetrahedron
    elif ct == "hexahedron":
        apply_dof_trans = apply_dof_transformation_hexahedron
    else:
        assert(False)
    J_q = np.zeros((num_q_points, gdim, tdim), dtype=np.float64)
    detJ_q = np.zeros((num_q_points, 1), dtype=np.float64)
    dphi_c = np.zeros(c_tab[0][1:gdim + 1, 0, :, 0].shape, dtype=np.float64)
    detJ = np.zeros(1, dtype=np.float64)
    entries_per_cell = (block_size * num_dofs_per_cell)**2
    # Assemble matrix
    Ae = np.zeros((block_size * num_dofs_per_cell, block_size * num_dofs_per_cell))
    blocks = [np.arange(b, block_size * num_dofs_per_cell + b, block_size) for b in range(block_size)]
    ref_detJ = np.zeros(len(ref_jacobians.keys()), dtype=np.float64)
    rdetJ = np.zeros(1)
    for i, jac in ref_jacobians.items():
        compute_determinant(jac, rdetJ)
        ref_detJ[i] = rdetJ[0]
    for facet in range(num_facets):
        cell = facet_info[facet, 1]
        local_facet = facet_info[facet, 2]
        for j in range(num_dofs_x):
            geometry[j] = x[x_dofs[cell, j], : gdim]

        # Compute Jacobian at each quadrature point
        if is_affine:
            dphi_c[:] = c_tab[local_facet][1:gdim + 1, 0, :, 0]
            J_q[0] = np.dot(geometry.T, dphi_c.T)
            compute_determinant(J_q[0], detJ)
            detJ_q[:] = detJ[0]
        else:
            for i, q in enumerate(q_p[local_facet]):

                dphi_c[:] = c_tab[local_facet][1:gdim + 1, i, :, 0]
                J_q[i] = geometry.T @ dphi_c.T
                compute_determinant(J_q[i], detJ)
                detJ_q[i] = detJ[0]

        if needs_transformations:
            # Transpose phi before applying dof transformations (ndofs, nquadpoints)
            phi_ = phi[local_facet].T.copy()
            apply_dof_trans(e_transformations, e_dofs, phi_, num_q_points, cell_perm[cell])
            # Reshape output as the transpose of the phi, i.e. (basis_function, quadrature_point)
            phi_T = phi_.copy()
        else:
            phi_T = phi[local_facet].T.copy()
        phi_s = (phi_T.T * q_w) * np.abs(detJ_q) * np.abs(ref_detJ[local_facet])
        # Compute weighted basis functions at quadrature points
        # Compute Ae_(i,j) = sum_(s=1)^len(q_w) w_s phi_j(q_s) phi_i(q_s) |det(J(q_s))|
        kernel = phi_T @ phi_s
        # Insert per block size
        for i in range(num_dofs_per_cell):
            for b in range(block_size):
                Ai = Ae[i * block_size + b]
                Ai[blocks[b]] = kernel[i]
        # Add to csr matrix
        data[cell * entries_per_cell: (cell + 1) * entries_per_cell] += np.ravel(Ae)
        print(Ae)
    return
