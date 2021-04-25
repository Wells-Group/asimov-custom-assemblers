// Copyright (C) 2021 Igor Baratta
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
#pragma once

#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

namespace custom::la
{
template <typename ValueType, typename IndexType>
class CooMatrix
{
public:
  using value_type = ValueType;
  using index_type = IndexType;

  CooMatrix(index_type num_cells, index_type num_dofs_cell)
  {
    _data = xt::zeros<ValueType>({num_cells, num_dofs_cell, num_dofs_cell});
  }

  void add_values(xt::xtensor<value_type, 2> Ae, index_type cell)
  {
    xt::view(_data, cell, xt::all(), xt::all()) = Ae;
  }

  void add_values(xt::xtensor<value_type, 2> Ae, xtl::span<const index_type> dofs, index_type cell)
  {
    auto A_local = xt::view(_data, cell, xt::all(), xt::all());
    for (std::size_t i = 0; i < dofs.size(); i++)
      for (std::size_t j = 0; j < dofs.size(); j++)
        A_local(dofs[i], dofs[j]) = Ae(i, j);
  }

  auto& array() { return _data; }

private:
  xt::xtensor<value_type, 3> _data;
};
} // namespace custom::la