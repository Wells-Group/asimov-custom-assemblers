#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <string>
#include <xtensor-blas/xlinalg.hpp>

#include "math.hpp"

constexpr std::int32_t gdim = 3;
constexpr std::int32_t tdim = 3;
constexpr std::int32_t d = 4;

enum Kernel
{
  Mass,
  Stiffness
};

using kernel_fn = std::function<void(double*, const double*, const double*, const double*,
                                     const int*, const std::uint8_t*, const std::uint32_t)>;

namespace dolfinx_cuas
{

kernel_fn generate_kernel(std::string family, std::string cell, Kernel type, int P)
{

  int quad_degree = 0;
  if (type == Kernel::Stiffness)
    quad_degree = (P - 1) + (P - 1);
  else if (type == Kernel::Mass)
    quad_degree = 2 * P;

  auto [points, weight]
      = basix::quadrature::make_quadrature("default", basix::cell::str_to_type(cell), quad_degree);
  std::vector<double> weights(weight);

  // Create Finite element for test and trial functions and tabulate shape functions
  basix::FiniteElement element = basix::create_element(family, cell, P);
  xt::xtensor<double, 4> basis = element.tabulate(1, points);
  xt::xtensor<double, 2> phi = xt::view(basis, 0, xt::all(), xt::all(), 0);
  xt::xtensor<double, 3> dphi = xt::view(basis, xt::range(1, tdim + 1), xt::all(), xt::all(), 0);

  // Get coordinate element from dolfinx
  basix::FiniteElement coordinate_element = basix::create_element("Lagrange", cell, 1);
  xt::xtensor<double, 4> coordinate_basis = coordinate_element.tabulate(1, points);

  xt::xtensor<double, 2> dphi0_c
      = xt::round(xt::view(coordinate_basis, xt::range(1, tdim + 1), 0, xt::all(), 0));

  std::int32_t ndofs_cell = phi.shape(2);
  xt::xtensor<double, 2> J = xt::zeros<double>({gdim, tdim});
  xt::xtensor<double, 2> K = xt::zeros<double>({tdim, gdim});

  // Stiffness Matrix using quadrature formulation
  // =====================================================================================

  xt::xtensor<double, 3> _dphi({dphi.shape(1), dphi.shape(2), dphi.shape(0)});
  for (std::int32_t k = 0; k < tdim; k++)
    for (std::size_t q = 0; q < weights.size(); q++)
      for (std::int32_t i = 0; i < ndofs_cell; i++)
        _dphi(q, i, k) = dphi(k, q, i);

  kernel_fn stiffness
      = [dphi0_c, _dphi, weights,
         ndofs_cell](double* A, const double* c, const double* w, const double* coordinate_dofs,
                     const int* entity_local_index, const std::uint8_t* quadrature_permutation,
                     const std::uint32_t cell_permutation)
  {
    // Compute local matrix
    xt::xtensor_fixed<double, xt::fixed_shape<gdim, tdim>> J;
    xt::xtensor_fixed<double, xt::fixed_shape<tdim, gdim>> K;

    J(0, 0) = dphi0_c(0, 0) * coordinate_dofs[0] + dphi0_c(0, 1) * coordinate_dofs[3];
    J(0, 1) = dphi0_c(0, 0) * coordinate_dofs[1] + dphi0_c(0, 1) * coordinate_dofs[4];
    J(0, 2) = dphi0_c(0, 0) * coordinate_dofs[2] + dphi0_c(0, 1) * coordinate_dofs[5];
    J(1, 0) = dphi0_c(1, 0) * coordinate_dofs[0] + dphi0_c(1, 2) * coordinate_dofs[6];
    J(1, 1) = dphi0_c(1, 0) * coordinate_dofs[1] + dphi0_c(1, 2) * coordinate_dofs[7];
    J(1, 2) = dphi0_c(1, 0) * coordinate_dofs[2] + dphi0_c(1, 2) * coordinate_dofs[8];
    J(2, 0) = dphi0_c(2, 0) * coordinate_dofs[0] + dphi0_c(2, 3) * coordinate_dofs[9];
    J(2, 1) = dphi0_c(2, 0) * coordinate_dofs[1] + dphi0_c(2, 3) * coordinate_dofs[10];
    J(2, 2) = dphi0_c(2, 0) * coordinate_dofs[2] + dphi0_c(2, 3) * coordinate_dofs[11];

    dolfinx_cuas::math::inv(J, K);
    double detJ = std::fabs(dolfinx_cuas::math::det(J));

    // // Main loop
    for (std::size_t q = 0; q < weights.size(); q++)
    {
      double w0 = weights[q] * detJ;
      double d0[ndofs_cell];
      double d1[ndofs_cell];
      double d2[ndofs_cell];

      for (int i = 0; i < ndofs_cell; i++)
      {
        d0[i] = K(0, 0) * _dphi(q, i, 0) + K(0, 1) * _dphi(q, i, 1) + K(0, 2) * _dphi(q, i, 2);
        d1[i] = K(1, 0) * _dphi(q, i, 0) + K(1, 1) * _dphi(q, i, 1) + K(1, 2) * _dphi(q, i, 2);
        d2[i] = K(2, 0) * _dphi(q, i, 0) + K(2, 1) * _dphi(q, i, 1) + K(2, 2) * _dphi(q, i, 2);
      }

      for (int i = 0; i < ndofs_cell; i++)
      {
        for (int j = 0; j < ndofs_cell; j++)
        {
          A[i * ndofs_cell + j] += (d0[i] * d0[j] + d1[i] * d1[j] + d2[i] * d2[j]) * w0;
        }
      }
    }
  };

  kernel_fn mass = [=](double* A, const double* c, const double* w, const double* coordinate_dofs,
                       const int* entity_local_index, const std::uint8_t* quadrature_permutation,
                       const std::uint32_t cell_permutation)
  {
    // Compute local matrix
    xt::xtensor_fixed<double, xt::fixed_shape<gdim, tdim>> J;
    xt::xtensor_fixed<double, xt::fixed_shape<tdim, gdim>> K;

    J(0, 0) = dphi0_c(0, 0) * coordinate_dofs[0] + dphi0_c(0, 1) * coordinate_dofs[3];
    J(0, 1) = dphi0_c(0, 0) * coordinate_dofs[1] + dphi0_c(0, 1) * coordinate_dofs[4];
    J(0, 2) = dphi0_c(0, 0) * coordinate_dofs[2] + dphi0_c(0, 1) * coordinate_dofs[5];
    J(1, 0) = dphi0_c(1, 0) * coordinate_dofs[0] + dphi0_c(1, 2) * coordinate_dofs[6];
    J(1, 1) = dphi0_c(1, 0) * coordinate_dofs[1] + dphi0_c(1, 2) * coordinate_dofs[7];
    J(1, 2) = dphi0_c(1, 0) * coordinate_dofs[2] + dphi0_c(1, 2) * coordinate_dofs[8];
    J(2, 0) = dphi0_c(2, 0) * coordinate_dofs[0] + dphi0_c(2, 3) * coordinate_dofs[9];
    J(2, 1) = dphi0_c(2, 0) * coordinate_dofs[1] + dphi0_c(2, 3) * coordinate_dofs[10];
    J(2, 2) = dphi0_c(2, 0) * coordinate_dofs[2] + dphi0_c(2, 3) * coordinate_dofs[11];

    dolfinx_cuas::math::inv(J, K);
    double detJ = std::fabs(dolfinx_cuas::math::det(J));

    // // Main loop
    for (std::size_t q = 0; q < weights.size(); q++)
    {
      double w0 = weights[q] * detJ;
      for (int i = 0; i < ndofs_cell; i++)
      {
        double w1 = w0 * phi.unchecked(q, i);
        for (int j = 0; j < ndofs_cell; j++)
          A[i * ndofs_cell + j] += w1 * phi.unchecked(q, j);
      }
    }
  };

  switch (type)
  {
  case Kernel::Mass:
    return mass;
  case Kernel::Stiffness:
    return stiffness;
  default:
    throw std::runtime_error("unrecognized kernel");
  }
}
} // namespace dolfinx_cuas
