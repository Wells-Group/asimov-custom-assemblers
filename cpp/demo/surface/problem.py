from ufl import (Coefficient, Constant, FunctionSpace, Mesh, TestFunction,
                 TrialFunction, VectorElement, ds, grad, inner, sym,
                 tetrahedron)

cell_type = tetrahedron
degree = 1

element = VectorElement("Lagrange", cell_type, degree)
mesh = Mesh(VectorElement("Lagrange", cell_type, 1))

V = FunctionSpace(mesh, element)

u = TrialFunction(V)
v = TestFunction(V)
f = Coefficient(V)
kappa = Constant(mesh)


def epsilon(v):
    return sym(grad(v))


a = kappa * inner(epsilon(u), epsilon(v)) * ds(1)
