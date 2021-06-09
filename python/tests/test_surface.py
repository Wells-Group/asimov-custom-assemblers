import dolfinx
from mpi4py import MPI
from dolfinx_cuas import assemble_matrix
import numpy as np
np.set_printoptions(formatter={'float': '{:0.3f}'.format})
mesh = dolfinx.BoxMesh(MPI.COMM_WORLD, [np.array([0, 0, 0]), np.array([1, 1, 1])], [1, 1, 1])

mesh.topology.create_entity_permutations()


def left_boundary(x):
    return np.isclose(x[0], 0.0)


left_facets = dolfinx.mesh.locate_entities_boundary(mesh, 2, left_boundary)
left_values = np.ones(len(left_facets), dtype=np.int32)
mt = dolfinx.MeshTags(mesh, 2, left_facets, left_values)
V = dolfinx.FunctionSpace(mesh, ("CG", 1))

A = assemble_matrix(V, 3, int_type="surface", mt=mt, index=1)
# print(A.todense())
