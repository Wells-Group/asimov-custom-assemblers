# Copyright (C) 2021 Jørgen S. Dokken, Igor Baratta
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

from .utils import compute_determinant, create_csr_sparsity_pattern, expand_dofmap, pack_facet_info
from .kernels import mass_kernel, surface_kernel

float_type = PETSc.ScalarType

__all__ = ["assemble_matrix"]

_dolfinx_to_basix_celltype = {dolfinx.cpp.mesh.CellType.interval: basix.CellType.interval,
                              dolfinx.cpp.mesh.CellType.triangle: basix.CellType.triangle,
                              dolfinx.cpp.mesh.CellType.quadrilateral: basix.CellType.quadrilateral,
                              dolfinx.cpp.mesh.CellType.hexahedron: basix.CellType.hexahedron,
                              dolfinx.cpp.mesh.CellType.tetrahedron: basix.CellType.tetrahedron}


def assemble_matrix(V: dolfinx.FunctionSpace, quadrature_degree: int, int_type: str = "mass",
                    mt: dolfinx.MeshTags = None, index: int = None):
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

    # Data from coordinate element
    ufl_c_el = mesh.ufl_domain().ufl_coordinate_element()
    ufc_family = ufl_c_el.family()
    if ufc_family == "Q":
        ufc_family = "Lagrange"
    is_affine = (dolfinx.cpp.mesh.is_simplex(mesh.topology.cell_type) and ufl_c_el.degree() == 1)

    # Create sparsity pattern
    dofmap = V.dofmap.list.array.reshape(num_cells, num_dofs_per_cell)
    block_size = V.dofmap.index_map_bs
    expanded_dofmap = np.zeros((num_cells, num_dofs_per_cell * block_size))
    expand_dofmap(dofmap, block_size, expanded_dofmap)
    rows, cols = create_csr_sparsity_pattern(num_cells, num_dofs_per_cell * block_size, expanded_dofmap)
    data = np.zeros(len(rows), dtype=float_type)

    mesh.topology.create_entity_permutations()
    if int_type == "mass":
        # Get quadrature points and weights
        q_p, q_w = basix.make_quadrature("default", element.cell_type, quadrature_degree)
        q_w = q_w.reshape(q_w.size, 1)

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

        cell_info = mesh.topology.get_cell_permutation_info()
        mass_kernel(data, num_cells, num_dofs_per_cell, num_dofs_x, x_dofs,
                    x, gdim, tdim, c_tab, q_p, q_w, phi, is_affine, entity_transformations,
                    entity_dofs, ct, cell_info, needs_transformations, block_size)
    elif int_type == "surface":
        facet_info = pack_facet_info(mesh, mt, 1)
        num_facets = facet_info.shape[0]
        facet_info = V.mesh.topology.get_facet_permutations()

        # Create quadrature points of reference facet
        surface_cell_type = dolfinx.cpp.mesh.cell_entity_type(mesh.topology.cell_type, mesh.topology.dim - 1)
        surface_str = dolfinx.cpp.mesh.to_string(surface_cell_type)
        surface_element = basix.create_element(family, surface_str, V.ufl_element().degree())

        # Basis functions of reference interval at quadrature points.
        # Shape is (derivatives, num_quadrature_point, num_basis_functions)
        q_p, q_w = basix.make_quadrature("default", surface_element.cell_type, quadrature_degree)
        q_w = q_w.reshape(q_w.size, 1)
        tabulated_data = surface_element.tabulate_x(1, q_p)
        phi_s = tabulated_data[0, :, :, 0]
        dphi_s = tabulated_data[1:, :, :, 0]
        # basix_surface_el = _dolfinx_to_basix_celltype[surface_cell_type]

        # Get the coordinates for the facets of the reference cell
        _cell = _dolfinx_to_basix_celltype[dolfinx.cpp.mesh.to_type(ct)]
        facet_topology = basix.topology(_cell)[mesh.topology.dim - 1]
        ref_geometry = basix.geometry(_cell)
        facet_coords = {}
        for i, facet in enumerate(facet_topology):
            facet_coords[i] = ref_geometry[facet]

        # NOTE: This can be greatly simplified if one uses the assumption that the mapping between the reference geometries
        # are always linear. Then one can use that for the ith facet J_i = (edge_i[1]-edge_i[0]) for 2D,
        # ((edge_i[1]-edge_i[0]),(edge_i[2]-edge_i[0]))
        # NOTE: We however use the surface element of the same order as the function space, but employ
        # that the reference cell is always affine
        # As the reference cell and reference facet is always affine this is simplified
        dphi_s0 = dphi_s[:, 0, :]
        _ref_jacs = {}
        for i, coord in facet_coords.items():
            _ref_jacs[i] = np.dot(coord.T, dphi_s0.T)
        # Map Jacobians to numba dict
        ref_jacobians = Dict.empty(key_type=types.int64, value_type=types.float64[:, :])
        for key, value in _ref_jacs.items():
            ref_jacobians[key] = value

        # Push quadrature points from reference facet forward to reference element
        q_cell = {}
        for i, coords in facet_coords.items():
            _x = np.zeros((len(q_p), mesh.geometry.dim))
            for j in range(_x.shape[0]):
                for k in range(_x.shape[1]):
                    for l in range(coords.shape[0]):
                        _x[j, k] += phi_s[j, l] * coords[l, k]
                q_cell[i] = _x

        # surface_kernel(data, num_facets, num_dofs_per_cell, num_dofs_x, x_dofs,
        #                x, gdim, tdim, c_tab, q_p, q_w, phi, is_affine, entity_transformations,
        #                entity_dofs, ct, facet_info, needs_transformations, block_size)

    else:
        raise NotImplementedError(f"Integration kernel for {int_type} has not been implemeted.")

    # Coo is faster than CSR
    num_dofs_loc = V.dofmap.index_map.size_local * block_size
    out_matrix = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(num_dofs_loc, num_dofs_loc))
    return out_matrix
