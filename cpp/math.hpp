// Copyright (C) 2021 Igor Baratta
//
// This file is part of DOLFINx_CUAS
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cmath>
#include <dolfinx/common/math.h>

namespace dolfinx_cuas::math
{

/// Compute C += B^T * A^T
/// @param[in] A Input matrix
/// @param[in] B Input matrix
/// @param[in, out] C Filled to be C += B^T * A^T
template <typename U, typename V, typename P>
void dotT(const U& A, const V& B, P& C)
{
  assert(A.shape(1) == B.shape(0));
  assert(C.shape(0) == B.shape(1));
  assert(C.shape(1) == A.shape(0));
  for (std::size_t k = 0; k < B.shape(1); k++)
    for (std::size_t i = 0; i < A.shape(0); i++)
      for (std::size_t l = 0; l < B.shape(0); l++)
        C(k, i) += B(l, k) * A(i, l);
}

template <typename U, typename V>
void compute_inv(const U& A, V& B)
{
  using value_type = typename U::value_type;
  const int nrows = A.shape(0);
  const int ncols = A.shape(1);
  if (nrows == ncols)
  {
    dolfinx::math::inv(A, B);
  }
  else
  {
    dolfinx::math::pinv(A, B);
  }
}

// Computes the determinant of rectangular matrices
// det(A^T * A) = det(A) * det(A)
template <typename Matrix>
double compute_determinant(Matrix& A)
{
  if (A.shape(0) == A.shape(1))
    return dolfinx::math::det(A);
  else
  {
    using T = typename Matrix::value_type;
    xt::xtensor<T, 2> B = xt::transpose(A);
    xt::xtensor<T, 2> BA = xt::zeros<T>({B.shape(0), A.shape(1)});
    dolfinx::math::dot(B, A, BA);
    return std::sqrt(dolfinx::math::det(BA));
  }
}

template <typename U, typename V, typename P>
void compute_jacobian(const U& dphi, const V& coords, P& J)
{
  // J [gdim, tdim]
  // coords [d, gdim]
  // dphi [tdim, d]
  assert(J.shape(0) == coords.shape(1));
  assert(J.shape(1) == dphi.shape(0));
  assert(dphi.shape(1) == coords.shape(0));

  dotT(dphi, coords, J);
}

} // namespace dolfinx_cuas::math
