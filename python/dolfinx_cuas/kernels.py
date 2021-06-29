# Copyright (C) 2021 Jørgen S. Dokken, Igor Baratta
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numba
import numpy as np
from numba.typed import Dict

from .utils import compute_determinant, compute_inverse


@numba.njit(fastmath=True)
def mass_kernel(data: np.ndarray, ct: str, num_cells: int, is_affine: bool, block_size: int, num_dofs_per_cell: int,
                num_dofs_x: int, x_dofs: np.ndarray, x: np.ndarray, gdim: int, tdim: int,
                q_p: np.ndarray, q_w: np.ndarray, c_tab: np.ndarray, phi: np.ndarray,
                e_transformations: Dict, e_dofs: Dict, perm_info: int, needs_transformations: bool):
    """
    Assemble the mass matrix inner(u, v)*dx
    Parameters
    data
        Flattened structure of the matrix we would like to insert data into
    ct
        String indicating which cell type the mesh consists of.
        Valid options: 'triangle', 'quadrilateral', 'tetrahedron', 'hexahedron'.
    num_cells
        Number of cells we are integrating over
    is_affine
        Boolean indicating if we are integrating over an affine cell type (first order simplices).
    block_size
        Block size of the problem (number of repeating dimensions in the function space)
    num_dofs_per_cell
        Number of degrees of freedom on each cell (not multiplied by block size)
    num_dofs_x
        Number of degrees of freedom for the coordinate element
    x_dofs
        Two dimensional array (cell, degree of freedom) containing the mesh geometry dofs corrensponding to
        the ith cells's jth degree of freedom
    x
        Two dimensional array containing the coordinates of the mesh geometry
    gdim
        Geometrical dimension of the mesh
    tdim
        Topological dimension of the mesh
    q_p
        Two-dimensional array containing the quadrature points on the reference cell
    q_w
        Corresponding quadratue weights
    c_tab
        Four-dimensional array containing the coordinate basis functions and first derivatives tabulated at the
        quadrature points. Shape of the array is (derivatives, quadrature points, basis functions, value size).
    phi
        Two-dimensional array containing the basis functions tabulated at quadrature points.
        phi[i, j] is the jth basis function tabulated at the ith quadrature point
    e_transformations
        Dictionary containing the dof transformations for each set of entities
    e_dofs
        Dictionary containing the number of degrees of freedom of each entity. The key is the dimension of the entity.
        e_dofs[i][j] is the number of dofs on the jth entity of dimension i.
    perm_info
        Array containing bit information for each cell that is interpreted by basix to permute the dofs
    needs_transformations
        Boolean indicating if dof transformations are required for the given space
    """

    # Declaration of local structures
    geometry = np.zeros((num_dofs_x, gdim), dtype=np.float64)
    num_q_points = q_w.size
    J_q = np.zeros((num_q_points, gdim, tdim), dtype=np.float64)
    detJ_q = np.zeros((num_q_points, 1), dtype=np.float64)
    dphi_c = c_tab[1:gdim + 1, 0, :, 0].copy()
    detJ = np.zeros(1, dtype=np.float64)
    entries_per_cell = (block_size * num_dofs_per_cell)**2
    # Assemble matrix
    Ae = np.zeros((block_size * num_dofs_per_cell, block_size * num_dofs_per_cell))
    blocks = [np.arange(b, block_size * num_dofs_per_cell + b, block_size) for b in range(block_size)]

    for cell in range(num_cells):
        geometry[:] = x[x_dofs[cell], :gdim]

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
def stiffness_kernel(data: np.ndarray, ct: str, num_cells: int, is_affine: bool, block_size: int,
                     num_dofs_per_cell: int, num_dofs_x: int, x_dofs: np.ndarray,
                     x: np.ndarray, gdim: int, tdim: int, q_p: np.ndarray, q_w: np.ndarray,
                     c_tab: np.ndarray, dphi: np.ndarray, e_transformations: Dict, e_dofs: Dict,
                     perm_info: np.ndarray, needs_transformations: bool):
    """
    Assemble the stiffness matrix inner(grad(u), grad(v))*dx
    Parameters
    data
        Flattened structure of the matrix we would like to insert data into
    ct
        String indicating which cell type the mesh consists of.
        Valid options: 'triangle', 'quadrilateral', 'tetrahedron', 'hexahedron'.
    num_cells
        Number of cells we are integrating over
    is_affine
        Boolean indicating if we are integrating over an affine cell type (first order simplices).
    block_size
        Block size of the problem (number of repeating dimensions in the function space)
    num_dofs_per_cell
        Number of degrees of freedom on each cell (not multiplied by block size)
    num_dofs_x
        Number of degrees of freedom for the coordinate element
    x_dofs
        Two dimensional array (cell, degree of freedom) containing the mesh geometry dofs corrensponding to
        the ith cells's jth degree of freedom
    x
        Two dimensional array containing the coordinates of the mesh geometry
    gdim
        Geometrical dimension of the mesh
    tdim
        Topological dimension of the mesh
    q_p
        Two-dimensional array containing the quadrature points on the reference cell
    q_w
        Corresponding quadratue weights
    c_tab
        Four-dimensional array containing the coordinate basis functions and first derivatives tabulated at the
        quadrature points. Shape of the array is (derivatives, quadrature points, basis functions, value size).
    dphi
        The first derivatives of the basis functions tabulated at quadrature points.
        dphi[i,j,k] is dphi_k/dx_i evaluated at the jth quadrature point.
    e_transformations
        Dictionary containing the dof transformations for each set of entities
    e_dofs
        Dictionary containing the number of degrees of freedom of each entity. The key is the dimension of the entity.
        e_dofs[i][j] is the number of dofs on the jth entity of dimension i.
    perm_info
        Array containing bit information for each cell that is interpreted by basix to permute the dofs
    needs_transformations
        Boolean indicating if dof transformations are required for the given space
    """

    # Declaration of local structures
    geometry = np.zeros((num_dofs_x, gdim), dtype=np.float64)
    num_q_points = q_w.size
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
        for i in range(tdim):
            dphi_i[i, :, :] = dphi[i, :, :].T.copy()

        geometry[:] = x[x_dofs[cell], :gdim]

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


@numba.njit
def surface_kernel(data: np.array, ct: str, is_affine: bool, block_size: int, num_dofs_per_cell: int, num_dofs_x: int,
                   x_dofs: np.ndarray, x: np.ndarray, gdim: int, tdim: int, q_p: np.ndarray, q_w: np.ndarray,
                   c_tab: np.ndarray, phi: Dict, ref_jacobians: Dict, e_transformations: Dict, e_dofs: Dict,
                   perm_info: np.array, needs_transformations: bool, facet_info: np.ndarray):
    """
    Assemble the surface integral inner(u, v)*ds on a set of facets
    Parameters
    data
        Flattened structure of the matrix we would like to insert data into
    ct
        String indicating which cell type the mesh consists of.
        Valid options: 'triangle', 'quadrilateral', 'tetrahedron', 'hexahedron'.
    is_affine
        Boolean indicating if we are integrating over an affine cell type (first order simplices).
    block_size
        Block size of the problem (number of repeating dimensions in the function space)
    num_dofs_per_cell
        Number of degrees of freedom on each cell (not multiplied by block size)
    num_dofs_x
        Number of degrees of freedom for the surface coordinate element
    x_dofs
        Two dimensional array (facet, degree of freedom) containing the mesh geometry dofs corrensponding to
        the ith facet's jth degree of freedom
    x
        Two dimensional array containing the coordinates of the mesh geometry
    gdim
        Geometrical dimension of the mesh
    tdim
        Topological dimension of the mesh
    q_p
        Two-dimensional array containing the quadrature points on the reference cell
    q_w
        Corresponding quadratue weights
    c_tab
        Four-dimensional array containing the coordinate basis functions and first derivatives tabulated at the
        quadrature points. Shape of the array is (derivatives, quadrature points, basis functions, value size).
    phi
        Dictionary of the basis functions tabulated at quadrature points. The key is the local facet number on the
        reference cell. phi[i][j,k] is the kth basis function evaluated on the
        ith local facet at the jth quadrature point.
    ref_jacobians
        Dictionary mapping a local facet of a reference cell to the Jacobian of the reference facet
    e_transformations
        Dictionary containing the dof transformations for each set of entities
    e_dofs
        Dictionary containing the number of degrees of freedom of each entity. The key is the dimension of the entity.
        e_dofs[i][j] is the number of dofs on the jth entity of dimension i.
    perm_info
        Array containing bit information for each cell that is interpreted by basix to permute the dofs
    needs_transformations
        Boolean indicating if dof transformations are required for the given space
    facet_info
        Two dimensional array of the form (num_facets, 2) where the first column contains the cell owning the facet,
        the second colum being the local index of this facet in the cell.
    """
    # Declaration of local structures
    geometry = np.zeros((num_dofs_x, gdim), dtype=np.float64)
    num_q_points = q_w.size
    J_q = np.zeros((num_q_points, gdim, tdim - 1), dtype=np.float64)
    detJ_q = np.zeros((num_q_points, 1), dtype=np.float64)
    dphi_c = np.zeros(c_tab[1:gdim + 1, 0, :, 0].shape, dtype=np.float64)
    detJ = np.zeros(1, dtype=np.float64)
    entries_per_cell = (block_size * num_dofs_per_cell)**2

    # Assemble matrix
    Ae = np.zeros((block_size * num_dofs_per_cell, block_size * num_dofs_per_cell))
    blocks = [np.arange(b, block_size * num_dofs_per_cell + b, block_size) for b in range(block_size)]
    num_facets_per_cell = len(e_dofs[tdim - 1])
    ref_detJ = np.zeros(num_facets_per_cell, dtype=np.float64)
    rdetJ = np.zeros(1)
    num_facets = facet_info.shape[0]
    for i, jac in ref_jacobians.items():
        compute_determinant(jac, rdetJ)
        ref_detJ[i] = rdetJ[0]
    for facet in range(num_facets):
        cell = facet_info[facet, 0]
        local_facet = facet_info[facet, 1]
        # Compute facet geoemtry
        geometry[:] = x[x_dofs[facet], : gdim]

        # Compute Jacobian at each quadrature point
        if is_affine:
            dphi_c[:] = c_tab[1:gdim + 1, 0, :, 0]
            J = geometry.T @ dphi_c.T
            J_q[:] = J
            compute_determinant(J_q[0], detJ)
            detJ_q[:] = detJ[0]
        else:
            for i, q in enumerate(q_p[local_facet]):

                dphi_c[:] = c_tab[1:gdim + 1, i, :, 0]
                J_q[i] = geometry.T @ dphi_c.T
                compute_determinant(J_q[i], detJ)
                detJ_q[i] = detJ[0]
        phi_T = phi[local_facet].T.copy()
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
        data[cell * entries_per_cell: (cell + 1) * entries_per_cell] += np.ravel(Ae)
    return
