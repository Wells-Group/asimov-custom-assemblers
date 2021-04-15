import basix
import dolfinx
import numba
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
    return Aref


@numba.njit(cache=True)
def create_csr_sparsity_pattern(num_cells: int, num_dofs_per_cell: int, dofmap: np.ndarray):
    """
    Create a csr matrix given a flattened dofmap and the number of cells and dofs per cell
    """
    entries_per_cell = num_dofs_per_cell**2
    num_data = num_cells * entries_per_cell
    rows, cols = np.zeros(num_data, dtype=np.int32), np.zeros(num_data, dtype=np.int32)
    offset = 0
    for cell in range(num_cells):
        cell_dofs = dofmap[cell]
        for i in range(num_dofs_per_cell):
            rows[offset:offset + num_dofs_per_cell] = cell_dofs
            cols[offset:offset + num_dofs_per_cell] = np.full(num_dofs_per_cell, cell_dofs[i])
            offset += num_dofs_per_cell
    return (rows, cols)


@numba.njit(cache=True, fastmath=True)
def mass_kernel(data: np.ndarray, num_cells: int, num_dofs_per_cell: int, num_dofs_x: int, x_dofs: np.ndarray,
                x: np.ndarray, gdim: int, tdim: int, c_tab: np.ndarray, q_p: np.ndarray, q_w: np.ndarray,
                phi: np.ndarray, is_affine: bool):
    """
    Assemble mass matrix into CSR array "data"
    """
    # Declaration of local structures
    geometry = np.zeros((num_dofs_x, gdim), dtype=np.float64)
    J_q = np.zeros((q_w.size, gdim, tdim), dtype=np.float64)
    detJ_q = np.zeros((q_w.size, 1), dtype=np.float64)

    entries_per_cell = num_dofs_per_cell**2

    dphi_c = np.empty(c_tab[1:3, 0, :, 0].shape, dtype=np.float64)
    # Assemble matrix
    for cell in range(num_cells):
        for j in range(num_dofs_x):
            geometry[j] = x[x_dofs[cell, j], : gdim]

        # Compute Jacobian at each quadrature point
        if is_affine:
            dphi_c[:] = c_tab[1:3, 0, :, 0]
            J_q[0] = np.dot(geometry.T, dphi_c.T)
            detJ = np.abs(linalg.det(J_q[0]))
            detJ_q[:] = detJ
        else:
            for i, q in enumerate(q_p):
                dphi_c[:] = c_tab[1: 3, i, :, 0]
                J_q[i] = geometry.T @ dphi_c.T
                # This is the slow part of the assembly
                detJ_q[i] = np.abs(linalg.det(J_q[i]))
            pass
        # Compute Ae_(i,j) = sum_(s=1)^len(q_w) w_s phi_j(q_s) phi_i(q_s) |det(J(q_s))|
        phi_scaled = phi * q_w * detJ_q
        kernel = phi.T @ phi_scaled
        # Add to csr matrix
        data[cell * entries_per_cell: (cell + 1) * entries_per_cell] = np.ravel(kernel)


def assemble_mass_matrix(V: dolfinx.FunctionSpace, quadrature_degree: int):
    """
    Assemble a mass matrix using custom assembler
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

    # Create basix element based on function space
    family = V.ufl_element().family() if not quad else "Lagrange"
    element = basix.create_element(family, str(
        V.ufl_cell()), V.ufl_element().degree())

    if not element.dof_transformations_are_identity:
        raise RuntimeError("Dof permutations not supported")

    # Get quadrature points and weights
    q_p, q_w = basix.make_quadrature("default", element.cell_type, quadrature_degree)
    q_w = q_w.reshape(q_w.size, 1)

    # Data from coordinate element
    ufl_c_el = mesh.ufl_domain().ufl_coordinate_element()
    ufc_family = ufl_c_el.family() if not quad else "Lagrange"
    c_element = basix.create_element(ufc_family, str(ufl_c_el.cell()), ufl_c_el.degree())
    c_tab = c_element.tabulate_x(1, q_p)

    # NOTE: Tabulate basis functions at quadrature points
    num_derivatives = 0
    tabulated_data = element.tabulate_x(num_derivatives, q_p)
    phi = tabulated_data[0, :, :, 0]
    is_affine = (dolfinx.cpp.mesh.is_simplex(mesh.topology.cell_type) and ufl_c_el.degree() == 1)

    # Create sparsity pattern
    rows, cols = create_csr_sparsity_pattern(num_cells, num_dofs_per_cell, dofmap)
    data = np.zeros(len(rows), dtype=float_type)

    mass_kernel(data, num_cells, num_dofs_per_cell, num_dofs_x, x_dofs,
                x, gdim, tdim, c_tab, q_p, q_w, phi, is_affine)

    num_dofs_glob = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    csr = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(num_dofs_glob, num_dofs_glob))
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
    ct = dolfinx.cpp.mesh.CellType.quadrilateral if quad else dolfinx.cpp.mesh.CellType.triangle
    N = 100
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, N, N, cell_type=ct)
    return mesh


if __name__ == "__main__":
    import argparse
    import time
    parser = argparse.ArgumentParser(description="Custom assembler of mass matrix using numba and Basix",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num-runs", default=2, type=np.int32, dest="runs",
                        help="Number of times to run the assembler")
    parser.add_argument("--degree", default=1, type=int, dest="degree",
                        help="Degree of Lagrange finite element space")
    _ct = parser.add_mutually_exclusive_group(required=False)
    _ct.add_argument('--quad', dest='quad', action='store_true',
                     help="Use quadrilateral mesh", default=False)
    _verbose = parser.add_mutually_exclusive_group(required=False)
    _verbose.add_argument('--verbose', dest='verbose', action='store_true',
                          help="Print matrices", default=False)

    args = parser.parse_args()
    quad = args.quad
    runs = args.runs
    verbose = args.verbose
    degree = args.degree

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    mesh = create_mesh(quad)

    cell_str = "quadrilateral" if quad else "triangle"
    el = ufl.FiniteElement("CG", cell_str, degree)
    quadrature_degree = 2 * el.degree() + 1
    V = dolfinx.FunctionSpace(mesh, el)

    for i in range(runs):
        start = time.time()
        Aref = compute_reference_mass_matrix(V, quadrature_degree)
        end = time.time()
        print(f"{i}: DOLFINx {end-start:.2e}")

        start = time.time()
        A = assemble_mass_matrix(V, quadrature_degree)
        end = time.time()
        print(f"{i}: Numba {end-start:.2e}")

    if verbose:
        print(f"Reference:\n {Aref[:,:]}")
        print(f"Solution:\n {A.toarray()}")

    print(f"Norm of matrix error {np.linalg.norm(Aref[:, :] - A.toarray())}")
    assert(np.allclose(A.toarray(), Aref[:, :]))
