
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
  // Contains quadrature points and weights on a cell on a set of entities

public:
  /// Constructor
  /// @param[in] ct The cell type
  /// @param[in] degree Degree of quadrature rule
  /// @param[in] Dimension of entity
  /// @param[in] type Type of quadrature rule
  QuadratureRule(dolfinx::mesh::CellType ct, int degree, int dim, std::string type = "default")
      : _cell_type(ct), _degree(degree), _type(type), _dim(dim)
  {

    basix::cell::type b_ct = dolfinx::mesh::cell_type_to_basix_type(ct);
    const int tdim = basix::cell::topological_dimension(b_ct);
    if (tdim == dim)
    {
      std::pair<xt::xarray<double>, std::vector<double>> quadrature
          = basix::quadrature::make_quadrature(type, b_ct, degree);
      // NOTE: Conversion could be easier if return-type had been nicer from Basix
      // Currently we need to determine the dimension of the quadrature rule and reshape data
      // accordingly
      xt::xarray<double> points;
      if (quadrature.first.dimension() == 1)
        auto points = xt::empty<double>({quadrature.first.shape(0)});
      else
        auto points = xt::empty<double>({quadrature.first.shape(0), quadrature.first.shape(1)});
      for (std::size_t i = 0; i < quadrature.first.size(); i++)
        points[i] = quadrature.first[i];
      std::vector<basix::cell::type> entity_types = basix::cell::subentity_types(b_ct)[dim];
      for (std::int32_t i = 0; i < entity_types.size(); i++)
      {
        _type_to_local_index[dolfinx::mesh::cell_type_from_basix_type(entity_types[i])].push_back(
            i);
      }
      for (auto j : _type_to_local_index)
      {
        std::cout << dolfinx::mesh::to_string(j.first) << "\n";
        std::cout << xt::adapt(j.second);
      }
    }
    else
    {
      const int num_entities = basix::cell::num_sub_entities(b_ct, dim);
      // Create map for each facet type to the local index
      std::map<dolfinx::mesh::CellType, std::vector<int>> type_to_facet_index;
      for (std::int32_t i = 0; i < num_entities; i++)
      {
        basix::cell::type ft = basix::cell::sub_entity_type(b_ct, dim, i);
        type_to_facet_index[dolfinx::mesh::cell_type_from_basix_type(ft)].push_back(i);
      }
      for (auto type : type_to_facet_index)
      {
        std::cout << dolfinx::mesh::to_string(type.first);
        std::cout << xt::adapt(type.second) << "HERE!\n";
      }
    }
  }

  /// Return quadrature points
  std::map<dolfinx::mesh::CellType, xt::xarray<double>>& points_ref() { return _points; }

  /// Return quadrature weights
  std::map<dolfinx::mesh::CellType, std::vector<double>>& weights_ref() { return _weights; }

  /// Return quadrature points
  std::map<dolfinx::mesh::CellType, xt::xarray<double>> points() { return _points; }

  /// Return quadrature weights
  std::map<dolfinx::mesh::CellType, std::vector<double>> weights() { return _weights; }

  /// Return the cell type for the ith quadrature rule
  /// @param[in] Local entity number
  dolfinx::mesh::CellType cell_type(int i)
  {
    basix::cell::type b_ct = dolfinx::mesh::cell_type_to_basix_type(_cell_type);
    assert(i < basix::cell::num_sub_entities(b_ct, _dim));
    basix::cell::type et = basix::cell::sub_entity_type(b_ct, _dim, i);
    return dolfinx::mesh::cell_type_from_basix_type(et);
  }

private:
  std::map<dolfinx::mesh::CellType, xt::xarray<double>>
      _points; // Quadrature points for each cell type
  std::map<dolfinx::mesh::CellType, std::vector<double>>
      _weights; // Quadrature weights for each cell type
  std::map<dolfinx::mesh::CellType, std::vector<int>>
      _type_to_local_index; // Map from cell type to local indices
  dolfinx::mesh::CellType _cell_type;
  int _degree;
  std::string _type;
  int _dim;
};

} // namespace dolfinx_cuas