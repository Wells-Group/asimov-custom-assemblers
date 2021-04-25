// Copyright (C) 2021 Igor Baratta
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
#pragma once

#include <xtensor/xadapt.hpp>
#include <xtensor/xexpression.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

namespace custom::la
{
template <typename ValueType, typename IndexType>
class CsrMatrix
{
public:
  using value_type = ValueType;
  using index_type = IndexType;

  CsrMatrix(xt::xtensor<value_type, 1>& data, xt::xtensor<index_type, 1>& indptr,
            xt::xtensor<index_type, 1>& indices)
      : _data(std::move(data)), _indptr(std::move(indptr)), _indices(std::move(indices))
  {
  }
  const auto row_indices(index_type i) const
  {
    return xt::view(_indices, xt::range(_indptr[i], _indptr[i + 1]));
  }
  auto row(index_type i) { return xt::view(_data, xt::range(_indptr[i], _indptr[i + 1])); }

  void add_values(xt::xtensor<value_type, 2> Ae, xtl::span<const index_type> dofs)
  {
    assert(dofs.size() == Ae.shape(0));
    for (std::size_t i = 0; i < Ae.shape(0); i++)
    {
      auto indices = row_indices(dofs[i]);
      auto data = row(dofs[i]);
      for (std::size_t j = 0; j < Ae.shape(2); j++)
      {
        auto it = std::lower_bound(indices.begin(), indices.end(), dofs[j]);
        auto pos = std::distance(indices.begin(), it);
        data[pos] += Ae(i, j);
      }
    }
  }

  const index_type rows() { return _indptr.size() - 1; }
  const index_type nnz() { return _data.size(); }

  auto& array() { return _data; }

private:
  xt::xtensor<value_type, 1> _data;
  xt::xtensor<index_type, 1> _indptr;
  xt::xtensor<index_type, 1> _indices;
};
} // namespace custom::la