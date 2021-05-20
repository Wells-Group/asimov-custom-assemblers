#include <basix/cell.h>
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx.h>
#include <dolfinx/geometry/utils.h>
#include <iostream>
#include <xtensor-blas/xlinalg.hpp>
#include <xtl/xspan.hpp>

namespace dolfinx_cuas
{
namespace contact
{

void facet_master_puppet_relation(const std::shared_ptr<dolfinx::mesh::Mesh>& mesh,
                                  const xtl::span<const std::int32_t>& puppet_facets,
                                  const xtl::span<const std::int32_t>& candidate_facets,
                                  int quadrature_degree)
{

  // Mesh info
  const std::int32_t gdim = mesh->geometry().dim();
  const std::int32_t tdim = mesh->topology().dim();
  const std::int32_t fdim = tdim - 1;
  // FIXME: Need basix element public in mesh
  // auto degree = mesh->geometry().cmap()._element->degree;
  int degree = 1;
  auto x_dofmap = mesh->geometry().dofmap();
  auto dolfinx_cell = mesh->topology().cell_type();
  auto basix_cell = basix::cell::str_to_type(dolfinx::mesh::to_string(dolfinx_cell));
  auto dolfinx_facet = dolfinx::mesh::cell_entity_type(dolfinx_cell, fdim);
  auto dolfinx_facet_str = dolfinx::mesh::to_string(dolfinx_facet);
  auto basix_facet = basix::cell::str_to_type(dolfinx_facet_str);

  // Connectivity to evaluate at vertices
  mesh->topology().create_connectivity(fdim, 0);
  auto f_to_v = mesh->topology().connectivity(fdim, 0);

  // Connectivity to evaluate at quadrature points
  mesh->topology().create_connectivity(fdim, tdim);
  auto f_to_c = mesh->topology().connectivity(fdim, tdim);
  mesh->topology().create_connectivity(tdim, fdim);
  auto c_to_f = mesh->topology().connectivity(tdim, fdim);

  // Create reference topology and geometry
  auto facet_topology = basix::cell::topology(basix_cell)[fdim];
  auto ref_geom = basix::cell::geometry(basix_cell);

  // Create facet quadrature points
  auto quadrature_points
      = basix::quadrature::make_quadrature("default", basix_facet, quadrature_degree).first;

  // Push forward quadrature points on reference facet to reference cell
  auto surface_element = basix::create_element("Lagrange", dolfinx_facet_str, degree);
  auto c_tab = surface_element.tabulate(0, quadrature_points);
  xt::xtensor<double, 2> phi_s = xt::view(c_tab, 0, xt::all(), xt::all(), 0);

  const std::uint32_t num_facets = facet_topology.size();
  const std::uint32_t num_quadrature_pts = quadrature_points.shape(0);
  xt::xtensor<double, 3> q_cell({num_facets, num_quadrature_pts, ref_geom.shape(1)});
  for (int i = 0; i < num_facets; ++i)
  {
    auto facet = facet_topology[i];
    auto coords = xt::view(ref_geom, xt::keep(facet), xt::all());
    auto q_facet = xt::view(q_cell, i, xt::all(), xt::all());
    q_facet = xt::linalg::dot(phi_s, coords);
  }

  // Push forward quadrature points on reference facet to reference cell
  auto coordinate_element
      = basix::create_element("Lagrange", dolfinx::mesh::to_string(dolfinx_cell), degree);
  xt::xtensor<double, 3> phi_facets({num_facets, num_quadrature_pts, coordinate_element.dim()});
  for (int i = 0; i < num_facets; ++i)
  {
    auto phi_i = xt::view(phi_facets, i, xt::all(), xt::all());
    auto q_facet = xt::view(q_cell, i, xt::all(), xt::all());
    auto cell_tab = coordinate_element.tabulate(0, q_facet);
    phi_i = xt::view(cell_tab, 0, xt::all(), xt::all(), 0);
  }
  std::cout << phi_facets << "\n";

  auto mesh_geometry = mesh->geometry().x();
  auto cmap = mesh->geometry().cmap();
  std::array<double, 3> point;
  point[2] = 0;

  // Create midpoint tree as compute_closest_entity will be called many times
  const std::vector<std::int32_t> candidate_facets_copy(candidate_facets.begin(),
                                                        candidate_facets.end());
  dolfinx::geometry::BoundingBoxTree master_bbox(*mesh, fdim, candidate_facets);
  auto master_midpoint_tree
      = dolfinx::geometry::create_midpoint_tree(*mesh, fdim, candidate_facets_copy);

  std::vector<std::int32_t> data;
  std::vector<std::int32_t> offset(1);
  offset[0] = 0;
  for (auto facet : puppet_facets)
  {
    auto cells = f_to_c->links(facet);
    assert(cells.size() == 1);
    auto cell = cells[0];
    auto x_dofs = x_dofmap.links(cell);
    auto facets = c_to_f->links(cell);

    auto local_facet = std::find(facets.begin(), facets.end(), facet);
    const std::int32_t local_index = std::distance(facets.data(), local_facet);
    auto coordinate_dofs = xt::view(mesh_geometry, xt::keep(x_dofs), xt::range(0, gdim));
    auto phi_facet = xt::view(phi_facets, local_index, xt::all(), xt::all());
    xt::xtensor<double, 2> q_phys({q_cell.shape(1), q_cell.shape(2)});
    cmap.push_forward(q_phys, coordinate_dofs, phi_facet);

    for (int i = 0; i < q_phys.shape(0); ++i)
    {
      for (int j = 0; j < gdim; ++j)
      {
        point[j] = q_phys(i, j);
      }
      // Find initial search radius
      std::pair<int, double> intermediate_result
          = dolfinx::geometry::compute_closest_entity(master_midpoint_tree, point, *mesh);
      std::pair<int, double> search_result = dolfinx::geometry::compute_closest_entity(
          master_bbox, point, *mesh, intermediate_result.second);
      data.push_back(search_result.first);
    }
    offset.push_back(data.size());
  }
  std::cout << "function" << std::endl;
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
} // namespace contact
} // namespace dolfinx_cuas
