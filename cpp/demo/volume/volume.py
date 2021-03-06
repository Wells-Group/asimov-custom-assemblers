from ufl import (FiniteElement, FunctionSpace, Mesh, TestFunction,
                 TrialFunction, VectorElement, dx, grad, inner, tetrahedron)

# Load namespace
ns = vars()
forms = []

for p in range(1, 6):
    cell_type = tetrahedron
    element = FiniteElement("Lagrange", cell_type, p)
    mesh = Mesh(VectorElement("Lagrange", cell_type, 1))

    V = FunctionSpace(mesh, element)
    u = TrialFunction(V)
    v = TestFunction(V)
    a_mass_name = 'a_mass' + str(p)
    a_stiffness_name = 'a_stiffness' + str(p)

    # Insert into namespace so that the forms will be named a1, a2, a3 etc.
    ns[a_mass_name] = inner(u, v) * dx
    ns[a_stiffness_name] = inner(grad(u), grad(v)) * dx
    # Delete, so that the forms will get unnamed args and coefficients
    # and default to v_0, v_1, w0, w1 etc.
    del u, v
    forms += [ns[a_mass_name], ns[a_stiffness_name]]
