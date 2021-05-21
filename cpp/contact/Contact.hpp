#include <basix/cell.h>
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx.h>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <iostream>
#include <xtensor-blas/xlinalg.hpp>
#include <xtl/xspan.hpp>

namespace dolfinx_cuas
{
namespace contact
{
class Contact
{
public:
  Contact(std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>> marker, int surface_0,
          int surface_1)
      : _marker(marker), _surface_0(surface_0), _surface_1(surface_1)
  {
  }

  // Return map from surface 0 to 1
  const std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> map_0_to_1() const
  {
    return _map_0_to_1;
  }
  // Return map from surface 0 to 1
  const std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> map_1_to_0() const
  {
    return _map_1_to_0;
  }

  const int surface_0() const { return _surface_0; }
  const int surface_1() const { return _surface_1; }
  const int quadrature_degree() const { return _quadrature_degree; }

  std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>> meshtags() const { return _marker; }

  void create_reference_facet_qp()
  { /*This function computes the push forward of the quadrature points on the facet to the
     reference cell
     saved in _qp_ref_facet*/

    auto mesh = _marker->mesh();
    // Mesh info
    const int gdim = mesh->geometry().dim();
    const int tdim = mesh->topology().dim();
    const int fdim = tdim - 1;
    // FIXME: Need basix element public in mesh
    // auto degree = mesh->geometry().cmap()._element->degree;
    int degree = 1;

    auto dolfinx_cell = mesh->topology().cell_type();
    auto basix_cell = basix::cell::str_to_type(dolfinx::mesh::to_string(dolfinx_cell));
    auto dolfinx_facet = dolfinx::mesh::cell_entity_type(dolfinx_cell, fdim);
    auto dolfinx_facet_str = dolfinx::mesh::to_string(dolfinx_facet);
    auto basix_facet = basix::cell::str_to_type(dolfinx_facet_str);

    // Connectivity to evaluate at vertices
    mesh->topology_mutable().create_connectivity(fdim, 0);
    auto f_to_v = mesh->topology().connectivity(fdim, 0);

    // Create reference topology and geometry
    auto facet_topology = basix::cell::topology(basix_cell)[fdim];
    auto ref_geom = basix::cell::geometry(basix_cell);

    // Create facet quadrature points
    auto quadrature_points
        = basix::quadrature::make_quadrature("default", basix_facet, _quadrature_degree).first;

    // Push forward quadrature points on reference facet to reference cell
    auto surface_element = basix::create_element("Lagrange", dolfinx_facet_str, degree);
    auto c_tab = surface_element.tabulate(0, quadrature_points);
    xt::xtensor<double, 2> phi_s = xt::view(c_tab, 0, xt::all(), xt::all(), 0);

    const std::uint32_t num_facets = facet_topology.size();
    const std::uint32_t num_quadrature_pts = quadrature_points.shape(0);
    _qp_ref_facet = xt::xtensor<double, 3>({num_facets, num_quadrature_pts, ref_geom.shape(1)});
    for (int i = 0; i < num_facets; ++i)
    {
      auto facet = facet_topology[i];
      auto coords = xt::view(ref_geom, xt::keep(facet), xt::all());
      auto q_facet = xt::view(_qp_ref_facet, i, xt::all(), xt::all());
      q_facet = xt::linalg::dot(phi_s, coords);
    }
  }

  void tabulate_on_ref_cell()
  {
    /*This function tabulates the basis function at the quadrature point
     _qp_ref_facet
     The values are saved in _phi_ref_facets*/
    std::uint32_t num_facets = _qp_ref_facet.shape(0);
    std::uint32_t num_quadrature_pts = _qp_ref_facet.shape(1);
    // FIXME: For higher order geometry need basix element public in mesh
    // auto degree = mesh->geometry().cmap()._element->degree;
    int degree = 1;
    auto dolfinx_cell = _marker->mesh()->topology().cell_type();
    auto coordinate_element
        = basix::create_element("Lagrange", dolfinx::mesh::to_string(dolfinx_cell), degree);
    _phi_ref_facets
        = xt::xtensor<double, 3>({num_facets, num_quadrature_pts, coordinate_element.dim()});
    for (int i = 0; i < num_facets; ++i)
    {
      auto phi_i = xt::view(_phi_ref_facets, i, xt::all(), xt::all());
      auto q_facet = xt::view(_qp_ref_facet, i, xt::all(), xt::all());
      auto cell_tab = coordinate_element.tabulate(0, q_facet);
      phi_i = xt::view(cell_tab, 0, xt::all(), xt::all(), 0);
    }
  }

  void create_q_phys(int origin_meshtag)
  {
    /*computes the push forward of the quadrature points _qp_ref_facet to the physical
     facet for each facet in puppet_facets
     depending on origin_meshtag: saved in
     _qp_phys_0 or _qp_phys_1*/
    // Mesh info
    auto mesh = _marker->mesh();
    auto mesh_geometry = mesh->geometry().x();
    auto cmap = mesh->geometry().cmap();
    auto x_dofmap = mesh->geometry().dofmap();
    const int gdim = mesh->geometry().dim();
    const int tdim = mesh->topology().dim();
    const int fdim = tdim - 1;
    // Connectivity to evaluate at quadrature points
    mesh->topology_mutable().create_connectivity(fdim, tdim);
    auto f_to_c = mesh->topology().connectivity(fdim, tdim);
    mesh->topology_mutable().create_connectivity(tdim, fdim);
    auto c_to_f = mesh->topology().connectivity(tdim, fdim);

    std::vector<int32_t> puppet_facets;
    std::int32_t puppet_value;

    if (origin_meshtag == 0)
    {
      puppet_value = _surface_0;
    }
    else
    {
      puppet_value = _surface_1;
    }
    for (int i = 0; i < _marker->indices().size(); ++i)
    {
      const std::int32_t tag = _marker->values()[i];
      const std::int32_t facet = _marker->indices()[i];
      if (tag == puppet_value)
      {
        puppet_facets.push_back(facet);
      }
    }
    xt::xtensor<double, 3>* q_phys_pt;
    if (origin_meshtag == 0)
    {
      _qp_phys_0 = xt::xtensor<double, 3>(
          {puppet_facets.size(), _qp_ref_facet.shape(1), _qp_ref_facet.shape(2)});
      q_phys_pt = &_qp_phys_0;
    }
    else
    {
      _qp_phys_1 = xt::xtensor<double, 3>(
          {puppet_facets.size(), _qp_ref_facet.shape(1), _qp_ref_facet.shape(2)});
      q_phys_pt = &_qp_phys_1;
    }
    xt::xtensor<double, 2> q_phys({_qp_ref_facet.shape(1), _qp_ref_facet.shape(2)});
    for (int i = 0; i < puppet_facets.size(); ++i)
    {
      int facet = puppet_facets[i];
      auto cells = f_to_c->links(facet);
      assert(cells.size() == 1);
      auto cell = cells[0];
      auto x_dofs = x_dofmap.links(cell);
      auto facets = c_to_f->links(cell);

      auto local_facet = std::find(facets.begin(), facets.end(), facet);
      const std::int32_t local_index = std::distance(facets.data(), local_facet);
      auto coordinate_dofs = xt::view(mesh_geometry, xt::keep(x_dofs), xt::range(0, gdim));
      auto phi_facet = xt::view(_phi_ref_facets, local_index, xt::all(), xt::all());
      xt::xtensor<double, 2> q_phys({_qp_ref_facet.shape(1), _qp_ref_facet.shape(2)});
      cmap.push_forward(q_phys, coordinate_dofs, phi_facet);
      xt::view(*q_phys_pt, i, xt::all(), xt::all()) = q_phys;
    }
  }
  void create_distance_map(int origin_meshtag)
  {
    /*This function finally computes the closest candidate facet for each
    puppet facet at each quadrature point
    saved in _map_0_to_1 or _map_1_to_0*/
    create_reference_facet_qp();
    tabulate_on_ref_cell();
    create_q_phys(origin_meshtag);
    auto mesh = _marker->mesh();

    // Mesh info
    const int gdim = mesh->geometry().dim();
    const int tdim = mesh->topology().dim();
    const int fdim = tdim - 1;

    std::array<double, 3> point;
    point[2] = 0;

    std::vector<int32_t> candidate_facets;
    std::vector<int32_t> puppet_facets;
    std::int32_t candidate_value;
    std::int32_t puppet_value;
    xt::xtensor<double, 3>* q_phys_pt;
    if (origin_meshtag == 0)
    {
      candidate_value = _surface_1;
      puppet_value = _surface_0;
      q_phys_pt = &_qp_phys_0;
    }
    else
    {
      candidate_value = _surface_0;
      puppet_value = _surface_1;
      q_phys_pt = &_qp_phys_1;
    }
    for (int i = 0; i < _marker->indices().size(); ++i)
    {
      const std::int32_t tag = _marker->values()[i];
      const std::int32_t facet = _marker->indices()[i];
      if (tag == candidate_value)
      {
        candidate_facets.push_back(facet);
      }
      else if (tag == puppet_value)
      {
        puppet_facets.push_back(facet); // Would it make sense to save these as they are also needed
                                        // in void create_q_phys(int origin_meshtag)
      }
    }
    // Create midpoint tree as compute_closest_entity will be called many times
    const std::vector<std::int32_t> candidate_facets_copy(candidate_facets.begin(),
                                                          candidate_facets.end());
    dolfinx::geometry::BoundingBoxTree master_bbox(*mesh, fdim, candidate_facets_copy);
    auto master_midpoint_tree
        = dolfinx::geometry::create_midpoint_tree(*mesh, fdim, candidate_facets_copy);

    std::vector<std::int32_t> data;
    std::vector<std::int32_t> offset(1);
    offset[0] = 0;
    for (int i = 0; i < puppet_facets.size(); ++i)
    {
      for (int j = 0; j < (*q_phys_pt).shape(1); ++j)
      {
        for (int k = 0; k < gdim; ++k)
          point[k] = (*q_phys_pt)(i, j, k);
        // Find initial search radius
        std::pair<int, double> intermediate_result
            = dolfinx::geometry::compute_closest_entity(master_midpoint_tree, point, *mesh);
        std::pair<int, double> search_result = dolfinx::geometry::compute_closest_entity(
            master_bbox, point, *mesh, intermediate_result.second);
        data.push_back(search_result.first);
      }
      offset.push_back(data.size());
    }

    if (origin_meshtag == 0)
      _map_0_to_1 = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(data, offset);
    else
      _map_1_to_0 = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(data, offset);

    std::cout << "new class" << std::endl;
    for (int i = 0; i < offset.size() - 1; ++i)
    {
      std::cout << offset[i] << std::endl;
      for (int j = offset[i]; j < offset[i + 1]; ++j)
      {
        std::cout << data[j] << " ";
      }
      std::cout << "\n";
    }
    return;
  }

private:
  int _quadrature_degree = 3;
  std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>> _marker;
  int _surface_0;
  int _surface_1;
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> _map_0_to_1;
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> _map_1_to_0;
  xt::xtensor<double, 3> _qp_phys_0;
  xt::xtensor<double, 3> _qp_phys_1;
  xt::xtensor<double, 3> _qp_ref_facet;
  xt::xtensor<double, 3> _phi_ref_facets;
};
} // namespace contact
} // namespace dolfinx_cuas
