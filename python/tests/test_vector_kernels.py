# Copyright (C) 2021 Sarah Roggendorf
#
# SPDX-License-Identifier:   MIT

from dolfinx.generation import UnitSquareMesh, UnitCubeMesh
from dolfinx.fem import Form, FunctionSpace, create_vector, assemble_vector, IntegralType
from dolfinx.mesh import MeshTags, locate_entities_boundary
import basix
import dolfinx_cuas
import dolfinx_cuas.cpp
import numpy as np
import pytest
import ufl
from mpi4py import MPI
from petsc4py import PETSc

compare_matrices = dolfinx_cuas.utils.compare_matrices


@pytest.mark.parametrize("kernel_type", [dolfinx_cuas.Kernel.Rhs])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("P", [1, 2, 3, 4, 5])
def test_vector_kernels(dim, kernel_type, P):
    N = 30 if dim == 2 else 10
    mesh = UnitSquareMesh(MPI.COMM_WORLD, N, N) if dim == 2 else UnitCubeMesh(MPI.COMM_WORLD, N, N, N)

    # Define variational form
    V = FunctionSpace(mesh, ("CG", P))

    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh)
    L = v * dx
    kernel_type = dolfinx_cuas.Kernel.Rhs

    # Compile UFL form
    cffi_options = ["-Ofast", "-march=native"]
    L = Form(L, jit_parameters={"cffi_extra_compile_args": cffi_options, "cffi_libraries": ["m"]})
    b = create_vector(L)

    # Normal assembly
    b.zeroEntries()
    assemble_vector(b, L)
    b.assemble()

    # Custom assembly
    num_local_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    active_cells = np.arange(num_local_cells, dtype=np.int32)
    b2 = create_vector(L)

    q_rule = dolfinx_cuas.QuadratureRule(mesh.topology.cell_type, P + 1,
                                         mesh.topology.dim, basix.quadrature.string_to_type("default"))
    kernel = dolfinx_cuas.cpp.generate_vector_kernel(V._cpp_object, kernel_type, q_rule)
    b2.zeroEntries()
    consts = np.zeros(0)
    coeffs = np.zeros((num_local_cells, 0), dtype=PETSc.ScalarType)
    dolfinx_cuas.assemble_vector(b2, V, active_cells, kernel, coeffs, consts, IntegralType.cell)
    b2.assemble()

    assert np.allclose(b.array, b2.array)


@pytest.mark.parametrize("kernel_type", [dolfinx_cuas.Kernel.Rhs])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("P", [1, 2, 3, 4, 5])
def test_vector_surface_kernel(dim, kernel_type, P):
    N = 30 if dim == 2 else 10
    mesh = UnitSquareMesh(MPI.COMM_WORLD, N, N) if dim == 2 else UnitCubeMesh(MPI.COMM_WORLD, N, N, N)

    # Find facets on boundary to integrate over
    facets = locate_entities_boundary(mesh, mesh.topology.dim - 1,
                                      lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                                              np.isclose(x[0], 1.0)))
    values = np.ones(len(facets), dtype=np.int32)
    ft = MeshTags(mesh, mesh.topology.dim - 1, facets, values)

    # Define variational form
    V = FunctionSpace(mesh, ("CG", P))
    v = ufl.TestFunction(V)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)
    L = v * ds(1)
    kernel_type = dolfinx_cuas.Kernel.Rhs
    # Compile UFL form
    # cffi_options = []  # ["-Ofast", "-march=native"]
    L = Form(L)  # , jit_parameters={"cffi_extra_compile_args": cffi_options, "cffi_libraries": ["m"]})
    b = create_vector(L)

    # Normal assembly
    b.zeroEntries()
    assemble_vector(b, L)
    b.assemble()

    # Custom assembly
    num_local_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    consts = np.zeros(0)
    coeffs = np.zeros((num_local_cells, 0), dtype=PETSc.ScalarType)

    b2 = create_vector(L)
    q_rule = dolfinx_cuas.QuadratureRule(mesh.topology.cell_type, P + 1,
                                         mesh.topology.dim - 1, basix.quadrature.string_to_type("default"))
    kernel = dolfinx_cuas.cpp.generate_surface_vector_kernel(V._cpp_object, kernel_type, q_rule)
    b2.zeroEntries()
    dolfinx_cuas.assemble_vector(b2, V, ft.indices, kernel, coeffs, consts, IntegralType.exterior_facet)
    b2.assemble()

    assert np.allclose(b.array, b2.array)
