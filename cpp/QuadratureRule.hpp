
// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINx_CUAS
//
// SPDX-License-Identifier:    MIT

#pragma once

#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/utils.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

namespace dolfinx_cuas
{

class QuadratureRule
{
  // Contains quadrature points and weights on a cell on a set of entities

public:
  /// Constructor
  /// @param[in] ct The cell type
  /// @param[in] degree Degree of quadrature rule
  /// @param[in] Dimension of entity
  /// @param[in] type Type of quadrature rule
  QuadratureRule(dolfinx::mesh::CellType ct, int degree, int dim,
                 basix::quadrature::type type = basix::quadrature::type::Default)
      : _cell_type(ct), _degree(degree), _type(type), _dim(dim)
  {

    basix::cell::type b_ct = dolfinx::mesh::cell_type_to_basix_type(ct);
    const int num_entities = basix::cell::num_sub_entities(b_ct, dim);
    _points.reserve(num_entities);
    _weights.reserve(num_entities);

    const int tdim = basix::cell::topological_dimension(b_ct);

    // If cell dimension no pushing forward
    if (tdim == dim)
    {
      std::array<std::vector<double>, 2> quadrature
          = basix::quadrature::make_quadrature(type, b_ct, degree);
      // NOTE: Conversion could be easier if return-type had been nicer from Basix
      // Currently we need to determine the dimension of the quadrature rule and reshape data
      // accordingly
      std::size_t num_pts = quadrature[1].size();
      std::size_t pt_shape = quadrature[0].size() / quadrature[1].size();
      xt::xtensor<double, 2> points({num_pts, pt_shape});
      std::copy(quadrature[0].cbegin(), quadrature[0].cend(), points.begin());
      for (std::int32_t i = 0; i < num_entities; i++)
      {
        _points.push_back(points);
        _weights.push_back(quadrature[1]);
      }
    }
    else
    {
      // Create reference topology and geometry
      auto entity_topology = basix::cell::topology(b_ct)[dim];

      // Create map for each facet type to the local index
      for (std::int32_t i = 0; i < num_entities; i++)
      {
        // FIXME: Support higher order cmap
        // NOTE: Not sure we need a higher order coordinate element, as reference facets
        // and cell is affine
        const int e_degree = 1;
        basix::cell::type et = basix::cell::sub_entity_type(b_ct, dim, i);
        basix::FiniteElement entity_element = basix::create_element(
            basix::element::family::P, et, e_degree, basix::element::lagrange_variant::gll_warped);

        // Create quadrature and tabulate on entity
        std::array<std::vector<double>, 2> quadrature
            = basix::quadrature::make_quadrature(et, degree);
        const std::size_t num_pts = quadrature[1].size();
        const std::size_t pt_shape = quadrature[0].size() / quadrature[1].size();

        std::array<std::size_t, 4> shape = entity_element.tabulate_shape(0, num_pts);
        xt::xtensor<double, 4> c_tab(shape);
        std::array<std::size_t, 2> pts_shape = {num_pts, pt_shape};
        entity_element.tabulate(0, basix::impl::cmdspan2_t(quadrature[0].data(), pts_shape),
                                basix::impl::mdspan4_t(c_tab.data(), shape));
        xt::xtensor<double, 2> phi_s = xt::view(c_tab, 0, xt::all(), xt::all(), 0);

        std::pair<std::vector<double>, std::array<std::size_t, 2>> sub_geom
            = basix::cell::sub_entity_geometry(b_ct, dim, i);
        xt::xtensor<double, 2> coords(sub_geom.second);

        std::copy(sub_geom.first.cbegin(), sub_geom.first.cend(), coords.begin());

        // Push forward quadrature point from reference entity to reference entity on cell
        _weights.push_back(quadrature[1]);
        xt::xtensor<double, 2> entity_qp
            = xt::zeros<double>({num_pts, static_cast<std::size_t>(coords.shape(1))});
        dolfinx::math::dot(phi_s, coords, entity_qp);
        _points.push_back(entity_qp);
      }
    }
  }
  /// Return a list of quadrature points for each entity in the cell (using local entity index as in
  /// DOLFINx/Basix)
  const std::vector<xt::xtensor<double, 2>>& points_ref() const { return _points; }

  /// Return a list of quadrature weights for each entity in the cell (using local entity index as
  /// in DOLFINx/Basix)
  const std::vector<std::vector<double>>& weights_ref() const { return _weights; }

  /// Return a list of quadrature points for each entity in the cell (using local entity index as in
  /// DOLFINx/Basix)
  std::vector<xt::xtensor<double, 2>> points() { return _points; }

  /// Return a list of quadrature weights for each entity in the cell (using local entity index as
  /// in DOLFINx/Basix)
  std::vector<std::vector<double>> weights() { return _weights; }

  /// Return dimension of entity in the quadrature rule
  int dim() const { return _dim; }

  /// Return the cell type for the ith quadrature rule
  /// @param[in] Local entity number
  dolfinx::mesh::CellType cell_type(int i) const
  {
    basix::cell::type b_ct = dolfinx::mesh::cell_type_to_basix_type(_cell_type);
    assert(i < basix::cell::num_sub_entities(b_ct, _dim));
    basix::cell::type et = basix::cell::sub_entity_type(b_ct, _dim, i);
    return dolfinx::mesh::cell_type_from_basix_type(et);
  }

  /// Return the number of quadrature points per entity
  std::size_t num_points(int i) { return _points[i].shape(0); }

private:
  dolfinx::mesh::CellType _cell_type;
  int _degree;
  basix::quadrature::type _type;
  int _dim;
  std::vector<xt::xtensor<double, 2>> _points; // Quadrature points for each entity on the cell
  std::vector<std::vector<double>> _weights;   // Quadrature weights for each entity on the cell
};

} // namespace dolfinx_cuas