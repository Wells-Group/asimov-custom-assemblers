
// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINx_CUAS
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <basix/quadrature.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/utils.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
namespace dolfinx_cuas
{

class QuadratureRule
{
  // Contains quadrature points and weights.

public:
  /// Constructor
  /// @param[in] ct The cell type
  /// @param[in] degree Degree of quadrature rule
  /// @param[in] type Type of quadrature rule
  QuadratureRule(dolfinx::mesh::CellType ct, int degree, std::string type = "default")
      : _cell_type(ct), _degree(degree), _type(type)
  {
    std::pair<xt::xarray<double>, std::vector<double>> quadrature
        = basix::quadrature::make_quadrature(type, dolfinx::mesh::cell_type_to_basix_type(ct),
                                             degree);
    // NOTE: Conversion could be easier if return-type had been nicer from Basix
    // Currently we need to determine the dimension of the quadrature rule and reshape data
    // accordingly
    if (quadrature.first.dimension() == 1)
      _points = xt::empty<double>({quadrature.first.shape(0)});
    else
      _points = xt::empty<double>({quadrature.first.shape(0), quadrature.first.shape(1)});
    for (std::size_t i = 0; i < quadrature.first.size(); i++)
      _points[i] = quadrature.first[i];

    _weights = quadrature.second;
  }

  /// Return quadrature points
  xt::xarray<double>& points_ref() { return _points; }

  /// Return quadrature weights
  std::vector<double>& weights_ref() { return _weights; }

  /// Return quadrature points
  xt::xarray<double> points() { return _points; }

  /// Return quadrature weights
  std::vector<double> weights() { return _weights; }

  /// Return number of quadrature points
  std::size_t num_points() { return _points.size(); }

  /// Return the quadrature cell type
  dolfinx::mesh::CellType cell_type() { return _cell_type; }

  /// Create facet quadrature points on cell
  QuadratureRule create_facet_rule()
  {
    basix::cell::type ct = dolfinx::mesh::cell_type_to_basix_type(_cell_type);
    const int tdim = basix::cell::topological_dimension(ct);
    const int num_facets = basix::cell::num_sub_entities(ct, tdim - 1);
    // Create map for each facet type to the local index
    std::map<basix::cell::type, std::vector<int>> type_to_facet_index;
    for (std::int32_t i = 0; i < num_facets; i++)
    {
      basix::cell::type ft = basix::cell::sub_entity_type(ct, tdim - 1, i);
      type_to_facet_index[ft].push_back(i);
    }
    for (auto type : type_to_facet_index)
      std::cout << int(type.first) << ":" << xt::adapt(type.second) << "HERE!\n";
  };

private:
  xt::xarray<double> _points;   // 2D array of quadrature points
  std::vector<double> _weights; // 1D array of quadrature weights
  dolfinx::mesh::CellType _cell_type;
  int _degree;
  std::string _type;
};

} // namespace dolfinx_cuas