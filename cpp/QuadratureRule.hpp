
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
  {
    std::pair<xt::xarray<double>, std::vector<double>> quadrature
        = basix::quadrature::make_quadrature(
            type, basix::cell::str_to_type(dolfinx::mesh::to_string(ct)), degree);
    // NOTE: Conversion could be easier if return-type had been nicer from Basix
    // Currently we need to determine the dimension of the quadrature rule and reshape data
    // accordingly
    if (quadrature.first.dimension() == 1)
      _points = xt::empty<double>({quadrature.first.shape(0)});
    else
      _points = xt::empty<double>({quadrature.first.shape(0), quadrature.first.shape(1)});
    for (std::size_t i = 0; i < quadrature.first.size(); i++)
      _points[i] = quadrature.first[i];

    _weights = xt::adapt(quadrature.second);
  }

  /// Return quadrature points
  xt::xarray<double>& points_ref() { return _points; }

  /// Return quadrature weights
  xt::xarray<double>& weights_ref() { return _weights; }

  /// Return quadrature points
  xt::xarray<double> points() { return _points; }

  /// Return quadrature weights
  xt::xarray<double> weights() { return _weights; }

  /// Return number of quadrature points
  std::size_t num_points() { return _points.size(); }

private:
  xt::xarray<double> _points;  // 2D array of quadrature points
  xt::xarray<double> _weights; // 1D array of quadrature weights
};

} // namespace dolfinx_cuas