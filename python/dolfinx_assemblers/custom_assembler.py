import basix
import dolfinx
import numba
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from .utils import compute_determinant, create_csr_sparsity_pattern
from petsc4py import PETSc

float_type = PETSc.ScalarType

__all__ = ["assemble_mass_matrix"]


@numba.njit(cache=True)
def mass_kernel(data: np.ndarray, num_cells: int, num_dofs_per_cell: int, num_dofs_x: int, x_dofs: np.ndarray,
                x: np.ndarray, gdim: int, tdim: int, c_tab: np.ndarray, q_p: np.ndarray, q_w: np.ndarray,
                phi: np.ndarray, is_affine: bool):
    """
    Assemble mass matrix into CSR array "data"
    """
    # Compute weighted basis functions at quadrature points
    phi_w = phi * q_w

    # Declaration of local structures
    geometry = np.zeros((num_dofs_x, gdim), dtype=np.float64)
    J_q = np.zeros((q_w.size, gdim, tdim), dtype=np.float64)
    detJ_q = np.zeros((q_w.size, 1), dtype=np.float64)
    dphi_c = np.empty(c_tab[1:3, 0, :, 0].shape, dtype=np.float64)
    detJ = np.zeros(1, dtype=np.float64)
    entries_per_cell = num_dofs_per_cell**2

    # Assemble matrix
    for cell in range(num_cells):
        for j in range(num_dofs_x):
            geometry[j] = x[x_dofs[cell, j], : gdim]

        # Compute Jacobian at each quadrature point
        if is_affine:
            dphi_c[:] = c_tab[1:3, 0, :, 0]
            J_q[0] = np.dot(geometry.T, dphi_c.T)
            compute_determinant(J_q[0], detJ)
            detJ_q[:] = detJ[0]
        else:
            for i, q in enumerate(q_p):
                dphi_c[:] = c_tab[1: 3, i, :, 0]
                J_q[i] = geometry.T @ dphi_c.T
                compute_determinant(J_q[i], detJ)
                detJ_q[i] = detJ[0]

        # Compute Ae_(i,j) = sum_(s=1)^len(q_w) w_s phi_j(q_s) phi_i(q_s) |det(J(q_s))|
        phi_scaled = phi_w * np.abs(detJ_q)
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
    quad = True if mesh.topology.cell_type == dolfinx.cpp.mesh.CellType.quadrilateral else False

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

    # Faster than CSR
    out_matrix = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(num_dofs_glob, num_dofs_glob))
    return out_matrix
