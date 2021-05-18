#include <dolfinx.h>
#include <dolfinx/geometry/utils.h>
#include <basix/finite-element.h>
#include <basix/cell.h>
#include <basix/quadrature.h>
#include <iostream>
#include <xtl/xspan.hpp>

namespace dolfinx_cuas {
namespace contact {

void facet_master_puppet_relation(const std::shared_ptr<dolfinx::mesh::Mesh>& mesh, const xtl::span<const std::int32_t>& puppet_facets,
                                  const xtl::span<const std::int32_t>& candidate_facets, int quadrature_degree){
    
    // Mesh info
    const std::int32_t gdim = mesh->geometry().dim();
    const std::int32_t tdim = mesh->topology().dim();
    const std::int32_t fdim = tdim - 1;
    // FIXME: Need basix element public in mesh
    // auto degree = mesh->geometry().cmap()._element->degree;
    int degree = 1;
    auto x_dofmap = mesh->geometry().dofmap();
    auto dolfinx_cell = mesh->topology().cell_type();
    auto basix_cell= basix::cell::str_to_type(dolfinx::mesh::to_string(dolfinx_cell));
    auto dolfinx_facet = dolfinx::mesh::cell_entity_type(dolfinx_cell, fdim);
    auto dolfinx_facet_str = dolfinx::mesh::to_string(dolfinx_facet);
    auto basix_facet = basix::cell::str_to_type(dolfinx_facet_str);

    // Create midpoint tree as compute_closest_entity will be called many times
    const std::vector<std::int32_t> candidate_facets_copy(candidate_facets.begin(), candidate_facets.end());
    dolfinx::geometry::BoundingBoxTree master_bbox(*mesh, fdim, candidate_facets);
    auto master_midpoint_tree = dolfinx::geometry::create_midpoint_tree(*mesh, fdim, candidate_facets_copy);

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
    auto quadrature_points = basix::quadrature::make_quadrature("default", basix_facet, quadrature_degree).first;

    //Push forward quadrature points on reference facet to reference cell
    auto surface_element = basix::create_element("Lagrange", dolfinx_facet_str, degree);
    auto c_tab = surface_element.tabulate(0, quadrature_points);
    xt::xtensor<double, 2> phi_s = xt::view(c_tab, 0, xt::all(), xt::all(), 0);

    for(int i =0; i < facet_topology.size(); ++i){
        auto facet = facet_topology[i];
        auto coords = ref_geom[facet];
        // for (int j = 0; j < gdim; ++j){
        //     for (int k = 0; k < tdim; ++k){
        //         q_cell[i][j] += phi[]coords[]
        //     }
        // }
    }

    for(xtl::span<const std::int32_t>::iterator facet = puppet_facets.begin(); facet != puppet_facets.end(); ++facet)
    {   
        auto cells = f_to_c->links(*facet);
        assert(cells.size == 1);
        auto cell = cells[0];
        auto x_dofs = x_dofmap.links(cell);
        auto facets = c_to_f->links(cell);
        
    }
    std::cout << "master_bbox" << std::endl;
    return;

}
} //namespace contact
}// namespace dolfinx_cuas

