# Copyright (C) 2021 Jørgen S. Dokken, Igor Baratta
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


import basix
from basix.numba_helpers import (apply_dof_transformation_triangle,
                                 apply_dof_transformation_quadrilateral,
                                 apply_dof_transformation_tetrahedron,
                                 apply_dof_transformation_hexahedron)
import dolfinx
import numba
from numba import types
from numba.typed import Dict

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from .utils import compute_determinant, create_csr_sparsity_pattern, expand_dofmap
from petsc4py import PETSc
float_type = PETSc.ScalarType

__all__ = ["assemble_mass_matrix"]


@numba.njit
def mass_kernel(data: np.ndarray, num_cells: int, num_dofs_per_cell: int, num_dofs_x: int, x_dofs: np.ndarray,
                x: np.ndarray, gdim: int, tdim: int, c_tab: np.ndarray, q_p: np.ndarray, q_w: np.ndarray,
                phi: np.ndarray, is_affine: bool, e_transformations: Dict, e_dofs: Dict, ct: str, cell_info: int,
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

        # Reshaping phi to "blocked" data and flatten it to a 1D array for input to dof transformations
        if needs_transformations:
            phi_T = phi.T.flatten()
            apply_dof_trans(e_transformations, e_dofs, phi_T, num_q_points, cell_info[cell])
            # Reshape output as the transpose of the phi, i.e. (basis_function, quadrature_point)
            phi_T = phi_T.reshape(phi.shape[1], phi.shape[0])
        else:
            phi_T = phi.T.copy()
        phi_s = (phi_T.T * q_w) * detJ_q
        # Compute weighted basis functions at quadrature points
        # Compute Ae_(i,j) = sum_(s=1)^len(q_w) w_s phi_j(q_s) phi_i(q_s) |det(J(q_s))|
        kernel = phi_T @ phi_s
        # Insert per block size
        for i in range(num_dofs_per_cell):
            for b in range(block_size):
                block = np.arange(b, block_size * num_dofs_per_cell + b, block_size)
                Ai = Ae[i * block_size + b]
                Ai[block] = kernel[i]
        # Add to csr matrix
        data[cell * entries_per_cell: (cell + 1) * entries_per_cell] = np.ravel(Ae)


def assemble_mass_matrix(V: dolfinx.FunctionSpace, quadrature_degree: int):
    """
    Assemble a mass matrix using custom assembler
    """

    # Extract mesh data
    mesh = V.mesh
    num_dofs_x = mesh.geometry.dofmap.links(0).size  # NOTE: Assumes same cell geometry in whole mesh

    gdim = mesh.geometry.dim
    tdim = mesh.topology.dim
    t_imap = mesh.topology.index_map(tdim)
    num_cells = t_imap.size_local + t_imap.num_ghosts
    del t_imap

    x = mesh.geometry.x
    x_dofs = mesh.geometry.dofmap.array.reshape(num_cells, num_dofs_x)

    # Extract function space data
    num_dofs_per_cell = V.dofmap.cell_dofs(0).size
    # Create basix element based on function space
    family = V.ufl_element().family()
    if family == "Q":
        family = "Lagrange"
    ct = str(V.ufl_cell())
    element = basix.create_element(family, ct, V.ufl_element().degree())

    # Get quadrature points and weights
    q_p, q_w = basix.make_quadrature("default", element.cell_type, quadrature_degree)
    q_w = q_w.reshape(q_w.size, 1)

    # Data from coordinate element
    ufl_c_el = mesh.ufl_domain().ufl_coordinate_element()
    ufc_family = ufl_c_el.family()
    if ufc_family == "Q":
        ufc_family = "Lagrange"

    c_element = basix.create_element(ufc_family, str(ufl_c_el.cell()), ufl_c_el.degree())
    c_tab = c_element.tabulate_x(1, q_p)

    # NOTE: Tabulate basis functions at quadrature points
    num_derivatives = 0
    tabulated_data = element.tabulate_x(num_derivatives, q_p)
    phi = tabulated_data[0, :, :, 0]

    # NOTE: This should probably be two flags, one "dof_transformations_are_permutations"
    # and "dof_transformations_are_indentity"
    needs_transformations = not element.dof_transformations_are_identity
    entity_transformations = Dict.empty(key_type=types.int64, value_type=types.float64[:, :])
    for i, transformation in enumerate(element.entity_transformations()):
        entity_transformations[i] = transformation

    entity_dofs = Dict.empty(key_type=types.int64, value_type=types.int32[:])
    for i, e_dofs in enumerate(element.entity_dofs):
        entity_dofs[i] = np.asarray(e_dofs, dtype=np.int32)

    mesh.topology.create_entity_permutations()
    cell_info = mesh.topology.get_cell_permutation_info()

    is_affine = (dolfinx.cpp.mesh.is_simplex(mesh.topology.cell_type) and ufl_c_el.degree() == 1)

    # Create sparsity pattern
    dofmap = V.dofmap.list.array.reshape(num_cells, num_dofs_per_cell)
    block_size = V.dofmap.index_map_bs
    expanded_dofmap = np.zeros((num_cells, num_dofs_per_cell * block_size))
    expand_dofmap(dofmap, block_size, expanded_dofmap)

    rows, cols = create_csr_sparsity_pattern(num_cells, num_dofs_per_cell * block_size, expanded_dofmap)
    data = np.zeros(len(rows), dtype=float_type)

    mass_kernel(data, num_cells, num_dofs_per_cell, num_dofs_x, x_dofs,
                x, gdim, tdim, c_tab, q_p, q_w, phi, is_affine, entity_transformations,
                entity_dofs, ct, cell_info, needs_transformations, block_size)

    num_dofs_glob = V.dofmap.index_map.size_local * block_size

    # Faster than CSR
    out_matrix = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(num_dofs_glob, num_dofs_glob))
    return out_matrix
