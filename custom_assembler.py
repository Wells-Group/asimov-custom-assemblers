import basix
import dolfinx
import numpy as np
import numpy.linalg as linalg
import scipy.sparse
import ufl
from mpi4py import MPI
from petsc4py import PETSc

float_type = PETSc.ScalarType


def compute_reference_mass_matrix(V: dolfinx.FunctionSpace, quadrature_degree: int):
    """
    Compute mass matrix with given quadrature degree
    """
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.dx(domain=mesh, metadata={"quadrature_degree": quadrature_degree})
    a = ufl.inner(u, v) * dx
    Aref = dolfinx.fem.assemble_matrix(a)
    Aref.assemble()
    return Aref[:, :]


def assemble_mass_matrix(V: dolfinx.FunctionSpace, quadrature_degree: int):
    """
    Assemble a mass matrix using custom assembler
    """
    # NOTE: Assumes same cell geometry in whole mesh
    mesh = V.mesh
    num_dofs_x = mesh.geometry.dofmap.links(0).size
    gdim = mesh.geometry.dim
    tdim = mesh.topology.dim
    t_imap = mesh.topology.index_map(tdim)
    num_cells = t_imap.size_local + t_imap.num_ghosts
    x = mesh.geometry.x
    x_dofs = mesh.geometry.dofmap.array.reshape(num_cells, num_dofs_x)

    num_dofs_per_cell = V.dofmap.cell_dofs(0).size
    dofmap = V.dofmap.list.array.reshape(num_cells, num_dofs_per_cell)
    geometry = np.zeros((num_dofs_x, gdim))

    family = V.ufl_element().family() if not quad else "Lagrange"
    element = basix.create_element(family, str(
        V.ufl_cell()), V.ufl_element().degree())

    # NOTE: NEED to add base permutation support
    base_transformations = element.base_transformations()
    needs_dof_permutations = False
    for transformation in base_transformations:
        if not np.allclose(transformation, np.eye(transformation.shape[0])):
            needs_dof_permutations = True
            break
    if needs_dof_permutations:
        raise RuntimeError("Dof permutations not supported")

    q_p, q_w = basix.make_quadrature("default", element.cell_type, quadrature_degree)
    q_w = q_w.reshape(q_w.size, 1)
    # Shape (Derivative, points, num_basis_functions, value_shape)
    # NOTE: For some cases we could get num derivatives from UFL
    num_derivatives = 0
    tabulated_data = element.tabulate_x(num_derivatives, q_p)
    phi = tabulated_data[0, :, :, 0]

    # Data from coordinate element
    ufl_c_el = mesh.ufl_domain().ufl_coordinate_element()
    ufc_family = ufl_c_el.family() if not quad else "Lagrange"
    c_element = basix.create_element(ufc_family, str(ufl_c_el.cell()), ufl_c_el.degree())
    c_tab = c_element.tabulate_x(1, q_p)

    J_q = np.zeros((q_w.size, gdim, tdim))
    detJ_q = np.zeros((q_w.size, 1))

    # Create sparsity pattern
    entries_per_cell = num_dofs_per_cell**2
    num_data = num_cells * entries_per_cell
    rows, cols, data = np.zeros(num_data, dtype=np.int32), np.zeros(
        num_data, dtype=np.int32), np.zeros(num_data, dtype=float_type)
    offset = 0
    for cell in range(num_cells):
        cell_dofs = dofmap[cell]
        for i in range(num_dofs_per_cell):
            rows[offset:offset + num_dofs_per_cell] = cell_dofs
            cols[offset:offset + num_dofs_per_cell] = np.full(num_dofs_per_cell, cell_dofs[i])
            offset += num_dofs_per_cell

    # Assemble matrix
    num_dofs_glob = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    for cell in range(num_cells):
        # FIXME: This assumes a particular geometry dof layout
        for j in range(num_dofs_x):
            geometry[j] = x[x_dofs[cell, j], : gdim]
        # Compute Jacobian at each quadrature point

        for i, q in enumerate(q_p):
            dphi_c = c_tab[1: 3, i, :, 0]
            J_q[i] = geometry.T @ dphi_c.T
            detJ_q[i] = np.abs(linalg.det(J_q[i]))
        phi_w = phi * q_w * detJ_q
        mass_kernel = (phi.T @ phi_w)

        # Add to global
        cell_dofs = dofmap[cell]
        data[cell * entries_per_cell: (cell + 1) * entries_per_cell] = np.ravel(mass_kernel)

    csr = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(num_dofs_glob, num_dofs_glob))
    csr.eliminate_zeros()
    csr.prune()
    return csr


def create_mesh(quad):
    """
    Create simplistic first order mesh
    """
    if quad:
        x = np.array([[0, 0], [1, 0], [0, 1.3], [1.2, 1]])
        cells = np.array([[0, 1, 2, 3]], dtype=np.int32)
        ufl_mesh = ufl.Mesh(ufl.VectorElement("Lagrange", "quadrilateral", 1))
    else:
        x = np.array([[0, 0], [1.1, 0], [0.3, 1.0], [2, 1.5]])
        cells = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
        ufl_mesh = ufl.Mesh(ufl.VectorElement("Lagrange", "triangle", 1))
    mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, x, ufl_mesh)
    return mesh


if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    for quad in [True, False]:
        mesh = create_mesh(quad)
        cell_str = "quadrilateral" if quad else "triangle"
        el = ufl.FiniteElement("CG", cell_str, 2)
        quadrature_degree = 2 * el.degree() + 1
        V = dolfinx.FunctionSpace(mesh, el)
        print(f"{25*'-'}{cell_str}{25*'-'}")
        Aref = compute_reference_mass_matrix(V, quadrature_degree)
        print(f"Reference matrix:\n{Aref}\n")

        A = assemble_mass_matrix(V, quadrature_degree)
        print(f"Solution:\n {A.toarray()}")
        assert(np.allclose(A.toarray(), Aref[:, :]))
