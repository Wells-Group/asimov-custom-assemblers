#include "CsrMatrix.hpp"
#include <dolfinx.h>

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

  return custom::la::CsrMatrix<ValueType, IndexType>(data, indptr, indices);
}