# Copyright (C) 2021 Sarah Roggendorf
#
# SPDX-License-Identifier:   LGPL-3.0-or-later

import dolfinx
import dolfinx_cuas
import dolfinx_cuas.cpp
import numpy as np
import pytest
import ufl
from mpi4py import MPI
from petsc4py import PETSc

kt = dolfinx_cuas.cpp.contact.Kernel
it = dolfinx.cpp.fem.IntegralType
compare_matrices = dolfinx_cuas.utils.compare_matrices


@pytest.mark.parametrize("kernel_type", [kt.NitscheRigidSurface])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("P", [1, 2, 3, 4, 5])
def test_vector_surface_kernel(dim, kernel_type, P):
    N = 30 if dim == 2 else 10
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, N, N) if dim == 2 else dolfinx.UnitCubeMesh(MPI.COMM_WORLD, N, N, N)

    # Find facets on boundary to integrate over
    facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1,
                                                   lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                                                           np.isclose(x[0], 1.0)))
    values = np.ones(len(facets), dtype=np.int32)
    ft = dolfinx.MeshTags(mesh, mesh.topology.dim - 1, facets, values)

    # Define variational form
    V = dolfinx.FunctionSpace(mesh, ("CG", P))
    v = ufl.TestFunction(V)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)
    L = v * ds(1)
    # Compile UFL form
    cffi_options = ["-Ofast", "-march=native"]
    L = dolfinx.fem.Form(L, jit_parameters={"cffi_extra_compile_args": cffi_options, "cffi_libraries": ["m"]})
    b = dolfinx.fem.create_vector(L)

    # Normal assembly
    b.zeroEntries()
    dolfinx.fem.assemble_vector(b, L)
    b.assemble()

    # Custom assembly
    num_local_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    consts = np.zeros(0)
    coeffs = np.zeros((num_local_cells, 0), dtype=PETSc.ScalarType)

    b2 = dolfinx.fem.create_vector(L)
    kernel = dolfinx_cuas.cpp.contact.generate_rhs_kernel(V._cpp_object, kernel_type, P + 1)
    b2.zeroEntries()
    dolfinx_cuas.assemble_vector(b2, V, ft.indices, kernel, coeffs, consts, it.exterior_facet)
    b2.assemble()

    assert np.allclose(b.array, b2.array)
