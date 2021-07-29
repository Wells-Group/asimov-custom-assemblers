
// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINx_CUAS
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <basix/quadrature.h>
#include <dolfinx/mesh/Mesh.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
namespace dolfinx_cuas
{

class QuadratureRule
{
  // Contains quadrature points and weights.

public:
  /// Constructor
  /// @param[in] mesh The mesh
  /// @param[in] degree Degree of quadrature rul
  /// @param[in] type Type of quadrature rule
  QuadratureRule(std::shared_ptr<const dolfinx::mesh::Mesh> mesh, int degree,
                 std::string type = "default")
  {

    dolfinx::mesh::CellType ct = mesh->topology().cell_type();
    std::pair<xt::xarray<double>, std::vector<double>> quadrature
        = basix::quadrature::make_quadrature(
            type, basix::cell::str_to_type(dolfinx::mesh::to_string(ct)), degree);
    // NOTE: Conversion could be easier if return-type had been nicer from Basix
    _points = xt::empty<double>({quadrature.first.size()});
    for (std::size_t i = 0; i < quadrature.first.size(); i++)
      _points[i] = quadrature.first[i];
    _weights = xt::empty<double>({quadrature.second.size()});
    for (std::size_t i = 0; i < quadrature.second.size(); i++)
      _weights[i] = quadrature.second[i];
  }

  /// Return pointer to quadrature points
  xt::xtensor<double, 1> points() { return _points; }

  /// Return pointer to quadrature weights
  xt::xtensor<double, 1> weights() { return _weights; }

  /// Return number of quadrature points
  std::size_t num_points() { return _points.size(); }

private:
  xt::xtensor<double, 1> _points;  // 2D array of quadrature points
  xt::xtensor<double, 1> _weights; // 1D array of quadrature weights
};

} // namespace dolfinx_cuas