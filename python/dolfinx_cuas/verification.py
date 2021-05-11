# Copyright (C) 2021 Jørgen S. Dokken, Igor Baratta, Sarah Roggendorf
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import dolfinx
import ufl

"""
Verification of assembly using dolfin-x
"""

__all__ = ["compute_reference_mass_matrix", "compute_reference_stiffness_matrix"]


def compute_reference_mass_matrix(V: dolfinx.FunctionSpace, quadrature_degree: int = -1, jit_parameters={}):
    """
    Compute mass matrix with given quadrature degree
    """
    mesh = V.mesh
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.dx(domain=mesh, metadata={"quadrature_degree": quadrature_degree})
    a = ufl.inner(u, v) * dx
    a = dolfinx.fem.Form(a, jit_parameters=jit_parameters)
    Aref = dolfinx.fem.assemble_matrix(a)
    Aref.assemble()
    return Aref


def compute_reference_stiffness_matrix(V: dolfinx.FunctionSpace, quadrature_degree: int = -1, jit_parameters={}):
    """
    Compute stiffness matrix with given quadrature degree
    """
    mesh = V.mesh
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.dx(domain=mesh, metadata={"quadrature_degree": quadrature_degree})
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    a = dolfinx.fem.Form(a, jit_parameters=jit_parameters)
    Aref = dolfinx.fem.assemble_matrix(a)
    Aref.assemble()
    return Aref


def compute_reference_surface_matrix(V: dolfinx.FunctionSpace, quadrature_degree: int = -1, mt: dolfinx.MeshTags = None,
                                     index: int = None, jit_parameters={}):
    """
    Compute mass matrix with given quadrature degree
    """
    mesh = V.mesh
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    if mt is not None and index is not None:
        ds = ufl.ds(domain=mesh, metadata={"quadrature_degree": quadrature_degree},
                    subdomain_data=mt, subdomain_id=index)
    else:
        ds = ufl.ds(domain=mesh, metadata={"quadrature_degree": quadrature_degree})
    a = ufl.inner(u, v) * ds
    a = dolfinx.fem.Form(a)
    Aref = dolfinx.fem.assemble_matrix(a)
    Aref.assemble()
    return Aref