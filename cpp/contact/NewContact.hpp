// Copyright (C) 2021 Jørgen S. Dokken and Sarah Roggendorf
//
// This file is part of DOLFINx_CUAS
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx_cuas/QuadratureRule.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
namespace dolfinx_cuas
{
namespace contact
{
class ContactInterface
{
public:
  /// Constructor
  /// @param[in] marker The meshtags defining the contact surfaces
  /// @param[in] surface_0 Value of the meshtag marking the first surface
  /// @param[in] surface_1 Value of the meshtag marking the second surface
  ContactInterface(std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>> marker, int surface_0,
                   int surface_1)
  {
    std::shared_ptr<const dolfinx::mesh::Mesh> mesh = marker->mesh();
    const int tdim = mesh->topology().dim();
    mesh->topology_mutable().create_connectivity(tdim - 1, tdim);
    mesh->topology_mutable().create_connectivity(tdim, tdim - 1);
    const dolfinx::mesh::Topology& topology = mesh->topology();
    auto f_to_c = mesh->topology().connectivity(tdim - 1, tdim);
    assert(f_to_c);
    auto c_to_f = mesh->topology().connectivity(tdim, tdim - 1);
    assert(c_to_f);
    std::vector<std::int32_t> facets_0 = marker->find(surface_0);
    std::vector<std::int32_t> facets_1 = marker->find(surface_1);

    // Helper function to convert facets to (cell index, local index)
    auto get_cell_indices = [c_to_f, f_to_c](std::vector<std::int32_t> facets)
    {
      std::vector<std::pair<std::int32_t, int>> indices;
      indices.reserve(facets.size());
      for (auto facet : facets)
      {
        auto cells = f_to_c->links(facet);
        assert(cells.size() == 1);
        auto local_facets = c_to_f->links(cells[0]);
        const auto it = std::find(local_facets.begin(), local_facets.end(), facet);
        assert(it != local_facets.end());
        const int facet_index = std::distance(local_facets.begin(), it);
        indices.push_back({cells[0], facet_index});
      }
      return indices;
    };

    _facets_0 = get_cell_indices(facets_0);
    _facets_1 = get_cell_indices(facets_1);
  }

private:
  std::vector<std::pair<std::int32_t, int>>
      _facets_0; // List of pairs of (cell_index, local_index) for the first surface to identify the
                 // facets
  std::vector<std::pair<std::int32_t, int>>
      _facets_1; // List of pairs of (cell_index, local_index) for the first surface to identify the
                 // facets
};
} // namespace contact

} // namespace dolfinx_cuas