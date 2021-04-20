// Copyright (C) 2021 Igor Baratta
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
#include <dolfinx.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xexpression.hpp>
#include <xtensor/xtensor.hpp>

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
      auto current = indices.begin();
      for (std::size_t j = 0; j < Ae.shape(2); j++)
      {
        auto it = std::find(indices.begin(), indices.end(), dofs[j]);
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

// ----------------------------------------------------------------------------------------------
// Create CSR Matrix using dolfinx create_sparsity_pattern.
template <typename ValueType, typename IndexType>
auto create_csr_matrix(const std::shared_ptr<dolfinx::fem::FunctionSpace>& V)
{
  const auto& mesh = V->mesh();
  std::array<const std::reference_wrapper<const dolfinx::fem::DofMap>, 2> dofmaps{*V->dofmap(),
                                                                                  *V->dofmap()};
  dolfinx::la::SparsityPattern pattern = dolfinx::fem::create_sparsity_pattern(
      mesh->topology(), dofmaps, {dolfinx::fem::IntegralType::cell});
  pattern.assemble();

  const auto& offsets = pattern.diagonal_pattern().offsets();
  xt::xtensor<IndexType, 1> indptr = xt::adapt(offsets);

  const auto& array = pattern.diagonal_pattern().array();
  xt::xtensor<IndexType, 1> indices = xt::adapt(array);
  xt::xtensor<ValueType, 1> data = xt::zeros<ValueType>({array.size()});

  return CsrMatrix<ValueType, IndexType>(data, indptr, indices);
}
} // namespace custom::la