import dolfinx
import basix
import numpy as np

__all__ = ["facet_master_puppet_relation"]

_dolfinx_to_basix_celltype = {dolfinx.cpp.mesh.CellType.interval: basix.CellType.interval,
                              dolfinx.cpp.mesh.CellType.triangle: basix.CellType.triangle,
                              dolfinx.cpp.mesh.CellType.quadrilateral: basix.CellType.quadrilateral,
                              dolfinx.cpp.mesh.CellType.hexahedron: basix.CellType.hexahedron,
                              dolfinx.cpp.mesh.CellType.tetrahedron: basix.CellType.tetrahedron}


def facet_master_puppet_relation(mesh, puppet_facets, master_facets, quadrature_degree=2):
    """
    Create a map from a set of facets to the closest ones in a bounding box tree
    """

    tdim = mesh.topology.dim
    fdim = tdim - 1
    # Create midpoint tree as compute_closest_entity will be called many times
    master_bbox = dolfinx.cpp.geometry.BoundingBoxTree(mesh, fdim, master_facets)
    master_midpoint_tree = dolfinx.cpp.geometry.create_midpoint_tree(mesh, fdim, master_facets)

    # Connectivity to evaluate at vertices
    mesh.topology.create_connectivity(fdim, 0)
    f_to_v = mesh.topology.connectivity(fdim, 0)
    # Connectivity to evaluate at quadrature points
    mesh.topology.create_connectivity(fdim, tdim)
    f_to_c = mesh.topology.connectivity(fdim, tdim)
    mesh.topology.create_connectivity(tdim, fdim)
    c_to_f = mesh.topology.connectivity(tdim, fdim)
    x_dofmap = mesh.geometry.dofmap
    cell_type = mesh.topology.cell_type
    basix_cell = _dolfinx_to_basix_celltype[cell_type]
    basix_facet = _dolfinx_to_basix_celltype[dolfinx.cpp.mesh.cell_entity_type(cell_type, fdim)]
    # FIXME: Add mapping from dolfin cell to basix cell type
    quadrature_points, _ = basix.make_quadrature("default", basix_facet, quadrature_degree)
    basix_top = basix.topology(basix_cell)
    basix_geom = basix.geometry(basix_cell)
    cmap = mesh.geometry.cmap
    gdim = mesh.geometry.dim
    x = np.zeros((quadrature_points.shape[0], gdim), dtype=np.float64)

    puppet_to_master = {}
    mesh_geometry = mesh.geometry.x
    for facet in puppet_facets:
        # For each vertex on facet, find closest entity on the other interface
        vertices = f_to_v.links(facet)
        vertex_x = dolfinx.cpp.mesh.entities_to_geometry(mesh, 0, vertices, False)
        m_facets = []
        for geometry_index in vertex_x:
            point = mesh_geometry[geometry_index].reshape(3,)
            # Find initial search radius
            potential_facet, R_init = dolfinx.geometry.compute_closest_entity(master_midpoint_tree, point, mesh)
            # Find mesh entity
            master_facet, R = dolfinx.geometry.compute_closest_entity(master_bbox, point, mesh, R=R_init)
            m_facets.append(master_facet)
        m_facets = np.unique(m_facets)
        puppet_to_master[facet] = m_facets

        o_facets = []
        # Find physical coordinates for facet integrals quadrature points
        # First, find local index of facet in cell
        cells = f_to_c.links(facet)
        assert(len(cells) == 1)
        cell = cells[0]
        x_dofs = x_dofmap.links(cell)
        facets = c_to_f.links(cell)
        local_index = np.argwhere(facets == facet)[0, 0]

        # Map quadrature point from facet to reference cell
        reference_vertices = basix_top[tdim - 1][local_index]
        X = basix_geom[reference_vertices]
        X_quad = np.zeros((quadrature_points.shape[0], X.shape[1]), dtype=np.float64)
        if basix_geom.shape[1] == 2:
            # 2D geometry, quadrature is interval
            for (i, quad) in enumerate(quadrature_points):
                X_quad[i] = X[0] + quad[0] * X[1]
        elif basix_geom.shape[1] == 3:
            # 3D geometry, quadrature over triangle/quadrilateral
            for (i, quad) in enumerate(quadrature_points):
                X_quad[i] = X[0] + quad[0] * (X[1] - X[0]) + quad[1] * (X[2] - X[0])
        else:
            raise NotImplementedError("Collision-detection for 1D meshes with 0D boundarires not implemented.")
        # Second, get physical coordinate of cell geometry
        num_nodes = len(x_dofs)
        coordinate_dofs = np.zeros((num_nodes, gdim), dtype=np.float64)
        for i in range(num_nodes):
            coordinate_dofs[i] = mesh_geometry[x_dofs[i], :gdim]
        # Third, compute physical coordinates for quadrature points
        x = cmap.push_forward(X_quad, coordinate_dofs)
        for point in x:
            point_ = point
            if mesh.geometry.dim != 3:
                point_ = np.zeros(3)
                point_[:gdim] = point

            # Find initial search radius
            potential_facet, R_init = dolfinx.geometry.compute_closest_entity(master_midpoint_tree, point_, mesh)
            # Find mesh entity
            master_facet, R = dolfinx.geometry.compute_closest_entity(master_bbox, point_, mesh, R=R_init)
            # Compute displacement from point to entity
            if facet == 5395:
                circ_facet_x = dolfinx.cpp.mesh.entities_to_geometry(mesh, fdim, [master_facet], False)
                x_circ = mesh_geometry[circ_facet_x][0]
                dist_vec = dolfinx.geometry.compute_distance_gjk(x_circ, point_)
                print(point_, point_ + dist_vec)
            o_facets.append(master_facet)
        o_facets = np.unique(o_facets)
        # puppet_to_master[facet] = o_facets
    return puppet_to_master
