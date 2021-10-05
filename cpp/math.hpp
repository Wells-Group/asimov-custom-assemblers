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

  dolfinx::math::dot(coords, dphi, J, true);
}

} // namespace dolfinx_cuas::math
