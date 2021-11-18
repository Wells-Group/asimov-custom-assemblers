# Copyright (C) 2021 Jørgen S. Dokken, Igor Baratta, Sarah Roggendorf
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


import typing

import basix
import dolfinx
import dolfinx_cuas.cpp
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from numba import types
from numba.typed import Dict
from petsc4py import PETSc

from .kernels import mass_kernel, stiffness_kernel, surface_kernel
from .utils import create_csr_sparsity_pattern, expand_dofmap, pack_facet_info

float_type = PETSc.ScalarType

__all__ = ["assemble_matrix_numba", "assemble_matrix", "assemble_vector"]

_dolfinx_to_basix_celltype = {dolfinx.cpp.mesh.CellType.interval: basix.CellType.interval,
                              dolfinx.cpp.mesh.CellType.triangle: basix.CellType.triangle,
                              dolfinx.cpp.mesh.CellType.quadrilateral: basix.CellType.quadrilateral,
                              dolfinx.cpp.mesh.CellType.hexahedron: basix.CellType.hexahedron,
                              dolfinx.cpp.mesh.CellType.tetrahedron: basix.CellType.tetrahedron}


def _cpp_dirichletbc(bc):
    """Unwrap Dirichlet BC objects as cpp objects"""
    if isinstance(bc, dolfinx.DirichletBC):
        return bc._cpp_object
    elif isinstance(bc, (tuple, list)):
        return list(map(lambda sub_bc: _cpp_dirichletbc(sub_bc), bc))
    return bc


def assemble_matrix(A: PETSc.Mat, V: dolfinx.FunctionSpace,
                    active_cells: np.ndarray,
                    kernel: dolfinx_cuas.cpp.KernelWrapper,
                    coeffs: np.ndarray,
                    consts: np.ndarray,
                    type: dolfinx.cpp.fem.IntegralType,
                    bcs: typing.List[dolfinx.DirichletBC] = [],
                    diagonal: float = 1.0):
    """Assemble bilinear form into a matrix. The returned matrix is not
    finalised, i.e. ghost values are not accumulated.
    """
    cpp_bcs = _cpp_dirichletbc(bcs)
    dolfinx_cuas.cpp.assemble_matrix(A, V._cpp_object, cpp_bcs,
                                     active_cells, kernel, coeffs, consts, type)
    A.assemblyBegin(PETSc.Mat.AssemblyType.FLUSH)
    A.assemblyEnd(PETSc.Mat.AssemblyType.FLUSH)
    dolfinx.cpp.fem.insert_diagonal(A, V._cpp_object, cpp_bcs, diagonal)


def assemble_vector(b: np.ndarray, V: dolfinx.FunctionSpace,
                    active_cells: np.ndarray,
                    kernel: dolfinx_cuas.cpp.KernelWrapper,
                    coeffs: np.ndarray,
                    consts: np.ndarray,
                    type: dolfinx.cpp.fem.IntegralType):
    """Assemble linear form into a vector. """
    dolfinx_cuas.cpp.assemble_vector(b, V._cpp_object, active_cells, kernel, coeffs, consts, type)


def assemble_matrix_numba(V: dolfinx.FunctionSpace, quadrature_degree: int, int_type: str = "mass",
                          mt: dolfinx.MeshTags = None, index: int = None):
    """
    Assemble a matrix using custom assembler.
    Parameters
    ----------
    V
        FunctionSpace used to define trial and test functions
    quadrature_degree
        Degree of quadrature scheme integrating polynomials of this degree exactly
    int_type
        Type of integral: 'mass', 'stiffness' or 'surface' currently supported
    mt
        MeshTag with markers of the entity one would like to integrate (optional)
    index
        Which tags in the MeshTag we would like to integrate over (optional)
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

    # Data from coordinate element
    ufl_c_el = mesh.ufl_domain().ufl_coordinate_element()
    ufc_family = ufl_c_el.family()
    if ufc_family == "Q":
        ufc_family = "Lagrange"
    is_affine = (dolfinx.cpp.mesh.is_simplex(mesh.topology.cell_type)
                 and ufl_c_el.degree() == 1)

    # Create basix element based on function space
    family = V.ufl_element().family()
    if family == "Q":
        family = "Lagrange"
    ct = basix.cell.string_to_type(V.mesh.topology.cell_type.name)
    element = basix.create_element(basix.finite_element.string_to_family(
        family, V.mesh.topology.cell_type.name), ct, V.ufl_element().degree(), basix.LagrangeVariant.gll_warped)

    # Extract data required for dof transformations
    # NOTE: this is untested since now only relevant for non-Lagrange spaces.
    entity_transformations = Dict.empty(key_type=types.string, value_type=types.float64[:, :, :])
    for key, value in element.entity_transformations().items():
        entity_transformations[key] = value
    entity_dofs = Dict.empty(key_type=types.int64, value_type=types.int32[:])
    for i, e_dofs in enumerate(element.num_entity_dofs):
        entity_dofs[i] = np.asarray(e_dofs, dtype=np.int32)

    mesh.topology.create_entity_permutations()
    perm_info = mesh.topology.get_cell_permutation_info()
    # NOTE: This should probably be two flags, one "dof_transformations_are_permutations"
    # and "dof_transformations_are_indentity"
    needs_transformations = V.element.needs_dof_transformations

    # Create sparsity pattern and "matrix"
    num_dofs_per_cell = V.dofmap.cell_dofs(0).size
    block_size = V.dofmap.index_map_bs
    expanded_dofmap = np.zeros((num_cells, num_dofs_per_cell * block_size))
    expand_dofmap(V.dofmap.list.array.reshape(num_cells, num_dofs_per_cell), block_size, expanded_dofmap)
    rows, cols = create_csr_sparsity_pattern(num_cells, num_dofs_per_cell * block_size, expanded_dofmap)
    data = np.zeros(len(rows), dtype=float_type)

    if int_type in ["mass", "stiffness"]:
        # Get quadrature points and weights
        rule = "default"
        q_p, q_w = basix.make_quadrature(basix.quadrature.string_to_type(rule), element.cell_type, quadrature_degree)
        q_w = q_w.reshape(q_w.size, 1)  # Reshape as nd array to use efficiently in kernels

        # Create coordinate element and tabulate basis functions for pullback
        c_element = basix.create_element(basix.finite_element.string_to_family(
            ufc_family, V.mesh.topology.cell_type.name), ct, ufl_c_el.degree(), basix.LagrangeVariant.gll_warped)
        c_tab = c_element.tabulate(1, q_p)

        if int_type == "mass":
            kernel = mass_kernel
            num_derivatives = 0
        elif int_type == "stiffness":
            kernel = stiffness_kernel
            num_derivatives = 1

        # Tabulate basis functions at quadrature points
        tabulated_data = element.tabulate(num_derivatives, q_p)
        if int_type == "mass":
            basis_functions = tabulated_data[0, :, :, 0]
        elif int_type == "stiffness":
            basis_functions = tabulated_data[1:, :, :, 0]

        # Assemble kernel into data
        kernel(data, V.mesh.topology.cell_type.name, num_cells, is_affine, block_size, num_dofs_per_cell,
               num_dofs_x, x_dofs, x, gdim, tdim, q_p, q_w, c_tab, basis_functions, entity_transformations,
               entity_dofs, perm_info, needs_transformations)

    elif int_type == "surface":
        # Extract facets from mesh tag, return ndarray with the (cell_index, local_facet_index) in each row
        facet_info, facet_geom = pack_facet_info(mesh, mt, index)
        num_dofs_x = facet_geom.shape[1]

        # Create coordinate element for facet
        # FIXME: Does not work for prism meshes
        surface_str = dolfinx.cpp.mesh.cell_entity_type(
            mesh.topology.cell_type, mesh.topology.dim - 1, 0).name
        surface_cell_type = basix.cell.string_to_type(surface_str)
        surface_element = basix.create_element(basix.finite_element.string_to_family(
            family, surface_str), surface_cell_type, ufl_c_el.degree(), basix.LagrangeVariant.gll_warped)

        # Tabulate reference coordinate element basis functions for facets at quadrature points.
        # Shape is (derivatives, num_quadrature_point, num_basis_functions, value_size)
        # NOTE: Current assumption is that value size is 1
        rule = "default"
        q_p, q_w = basix.make_quadrature(basix.quadrature.string_to_type(rule),
                                         surface_element.cell_type, quadrature_degree)
        q_w = q_w.reshape(q_w.size, 1)
        c_tab = surface_element.tabulate(1, q_p)
        # Get the coordinates for the facets of the reference cell
        _cell = _dolfinx_to_basix_celltype[dolfinx.cpp.mesh.to_type(V.mesh.topology.cell_type.name)]
        facet_topology = basix.topology(_cell)[mesh.topology.dim - 1]
        ref_geometry = basix.geometry(_cell)

        # Compute Jacobian of reference geometry and push quadrature points forward to reference element
        # for each facet
        # NOTE: The reference Jacobian for each facet can be greatly simplified if one uses the assumption
        # that the mapping between the reference geometries are always linear.
        # Then one can use that for the ith facet J_i = (edge_i[1]-edge_i[0]) for 2D,
        # ((edge_i[1]-edge_i[0]),(edge_i[2]-edge_i[0]))
        phi_s = c_tab[0, :, :, 0]
        # NOTE: We exploit that the reference cell is always affine and the mapping to the reference facet
        # is affine, thus there is only one Jacobian per facet
        dphi_s_q0 = c_tab[1:, 0, :, 0]
        ref_jacobians = Dict.empty(key_type=types.int64, value_type=types.float64[:, :])
        q_cell = Dict.empty(key_type=types.int64, value_type=types.float64[:, :])
        phi = Dict.empty(key_type=types.int64, value_type=types.float64[:, :])
        for i, facet in enumerate(facet_topology):
            coords = ref_geometry[facet]
            ref_jacobians[i] = np.dot(coords.T, dphi_s_q0.T)
            q_cell[i] = phi_s @ coords
            phi[i] = element.tabulate(0, q_cell[i])[0, :, :, 0]
        # Assemble surface integral
        surface_kernel(data, V.mesh.topology.cell_type.name, is_affine, block_size, num_dofs_per_cell, num_dofs_x, facet_geom,
                       x, gdim, tdim, q_cell, q_w, c_tab, phi, ref_jacobians, entity_transformations,
                       entity_dofs, perm_info, needs_transformations, facet_info)
    else:
        raise NotImplementedError(f"Integration kernel for {int_type} has not been implemeted.")

    # Create coo_matrix (as it is faster than CSR)
    num_dofs_loc = V.dofmap.index_map.size_local * block_size
    out_matrix = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(num_dofs_loc, num_dofs_loc))
    return out_matrix
