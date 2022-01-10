# Copyright (C) 2021 Jørgen S. Dokken, Igor Baratta, Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

from dolfinx import fem as _fem
from dolfinx import mesh as _mesh
import ufl

"""
Verification of assembly using DOLFINx
"""

__all__ = ["compute_reference_mass_matrix", "compute_reference_stiffness_matrix", "compute_reference_surface_matrix"]


def compute_reference_mass_matrix(V: _fem.FunctionSpace, quadrature_degree: int = -1, jit_parameters={}):
    """
    Compute mass matrix with given quadrature degree
    """
    mesh = V.mesh
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.dx(domain=mesh, metadata={"quadrature_degree": quadrature_degree})
    a = ufl.inner(u, v) * dx
    a = _fem.form(a, jit_parameters=jit_parameters)
    Aref = _fem.assemble_matrix(a)
    Aref.assemble()
    return Aref


def compute_reference_stiffness_matrix(V: _fem.FunctionSpace, quadrature_degree: int = -1, jit_parameters={}):
    """
    Compute stiffness matrix with given quadrature degree
    """
    mesh = V.mesh
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.dx(domain=mesh, metadata={"quadrature_degree": quadrature_degree})
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    a = _fem.form(a, jit_parameters=jit_parameters)
    Aref = _fem.assemble_matrix(a)
    Aref.assemble()
    return Aref


def compute_reference_surface_matrix(V: _fem.FunctionSpace, quadrature_degree: int = -1, mt: _mesh.MeshTags = None,
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
    a = _fem.form(a)
    Aref = _fem.assemble_matrix(a)
    Aref.assemble()
    return Aref
