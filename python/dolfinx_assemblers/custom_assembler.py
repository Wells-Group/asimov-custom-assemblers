# Copyright (C) 2021 Jørgen S. Dokken, Igor Baratta, Sarah Roggendorf
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


import basix
import dolfinx
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from numba import types
from numba.typed import Dict
from petsc4py import PETSc

from .utils import create_csr_sparsity_pattern, expand_dofmap
from .kernels import mass_kernel, stiffness_kernel

float_type = PETSc.ScalarType

__all__ = ["assemble_matrix", "assemble_stiffness_matrix"]


def assemble_matrix(V: dolfinx.FunctionSpace, quadrature_degree: int, int_type: str = "mass"):
    """
    Assemble a matrix using custom assembler
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
    if int_type == "mass":
        mass_kernel(data, num_cells, num_dofs_per_cell, num_dofs_x, x_dofs,
                    x, gdim, tdim, c_tab, q_p, q_w, phi, is_affine, entity_transformations,
                    entity_dofs, ct, cell_info, needs_transformations, block_size)
    else:
        raise NotImplementedError(f"Integration kernel for {int_type} has not been implemeted.")

    # Coo is faster than CSR
    num_dofs_loc = V.dofmap.index_map.size_local * block_size
    out_matrix = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(num_dofs_loc, num_dofs_loc))
    return out_matrix




def assemble_stiffness_matrix(V: dolfinx.FunctionSpace, quadrature_degree: int):
    """
    Assemble a stiffness matrix using custom assembler
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
    dofmap = V.dofmap.list.array.reshape(num_cells, num_dofs_per_cell)
    #quad = True if mesh.topology.cell_type == dolfinx.cpp.mesh.CellType.quadrilateral else False
    
    # Create basix element based on function space
    family = V.ufl_element().family()
    if family == "Q":
        family = "Lagrange"
    ct = str(V.ufl_cell())
    element = basix.create_element(family, ct, V.ufl_element().degree())

    if not element.dof_transformations_are_identity:
        raise RuntimeError("Dof permutations not supported")

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
    num_derivatives = 1
    tabulated_data = element.tabulate_x(num_derivatives, q_p)
    d_phi = tabulated_data[1:, :, :, 0]

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
    rows, cols = create_csr_sparsity_pattern(num_cells, num_dofs_per_cell, dofmap)
    data = np.zeros(len(rows), dtype=float_type)

    stiffness_kernel(data, num_cells, num_dofs_per_cell, num_dofs_x, x_dofs,
                     x, gdim, tdim, c_tab, q_p, q_w, d_phi, is_affine, entity_transformations,
                entity_dofs, ct, cell_info, needs_transformations)

    num_dofs_glob = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

    # Faster than CSR
    out_matrix = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(num_dofs_glob, num_dofs_glob))
    return out_matrix
