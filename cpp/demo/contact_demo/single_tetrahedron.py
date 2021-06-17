import dolfinx
import dolfinx.io
import ufl
from mpi4py import MPI
import numpy as np

gdim = 3
shape = "tetrahedron"
degree = 1


cell = ufl.Cell(shape, geometric_dimension=gdim)
domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, degree))

x = np.array([[0.6, 0.2, 0.2], [5., 0.3, 0.2], [0.6, 7., 0.2], [0.6, 0.3, 9]])
cells = np.array([[0, 1, 2, 3]], dtype=np.int32)
mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, x, domain)
mesh.topology.create_connectivity_all()
facets = np.arange(mesh.topology.index_map(mesh.topology.dim - 1).size_local, dtype=np.int32)
mt = dolfinx.MeshTags(mesh, mesh.topology.dim - 1, facets, facets)

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "tetrahedron.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(mt)
print(mt.values, mt.indices)
