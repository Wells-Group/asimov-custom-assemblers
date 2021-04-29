#include "math.hpp"
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx/common/math.h>
#include <string>
#include <xtensor-blas/xlinalg.hpp>

constexpr std::int32_t gdim = 3;
constexpr std::int32_t tdim = 3;
constexpr std::int32_t d = 4;

enum Kernel
{
  Mass,
  Stiffness
};

enum Representation
{
  Quadrature,
  Tensor
};

template <std::int32_t N>
using kernel_fn = std::function<void(xt::xtensor_fixed<double, xt::fixed_shape<d, gdim>>&,
                                     xt::xtensor_fixed<double, xt::fixed_shape<N, N>>&)>;

template <std::int32_t P>
auto generate_kernel(std::string family, std::string cell, Kernel type, Representation repr)
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

  constexpr std::int32_t ndofs_cell = (P + 1) * (P + 2) * (P + 3) / 6;
  xt::xtensor<double, 2> J = xt::zeros<double>({gdim, tdim});
  xt::xtensor<double, 2> K = xt::zeros<double>({tdim, gdim});

  // Mass Matrix using quadrature formulation
  // =====================================================================================
  kernel_fn<ndofs_cell> mass
      = [=](xt::xtensor_fixed<double, xt::fixed_shape<d, gdim>>& coordinate_dofs,
            xt::xtensor_fixed<double, xt::fixed_shape<ndofs_cell, ndofs_cell>>& Ae) {
          xt::xtensor<double, 2> J = xt::zeros<double>({gdim, tdim});
          dot34(dphi0_c, coordinate_dofs, J);
          double detJ = std::fabs(dolfinx::math::det(J));

          // Compute local matrix
          for (std::size_t q = 0; q < weights.size(); q++)
          {
            double w0 = weights[q] * detJ;
            for (int i = 0; i < ndofs_cell; i++)
            {
              double w1 = w0 * phi.unchecked(q, i);
              for (int j = 0; j < ndofs_cell; j++)
                Ae.unchecked(i, j) += w1 * phi.unchecked(q, j);
            }
          }
        };

  // Mass matrix using tensor representation
  // =====================================================================================
  // Pre-compute local matrix for reference element
  xt::xtensor<double, 2> A0 = xt::zeros<double>({ndofs_cell, ndofs_cell});
  for (std::size_t q = 0; q < weights.size(); q++)
    for (int i = 0; i < ndofs_cell; i++)
      for (int j = 0; j < ndofs_cell; j++)
        A0(i, j) += weights[q] * phi(q, i) * phi(q, j);

  kernel_fn<ndofs_cell> mass_tensor
      = [=](xt::xtensor_fixed<double, xt::fixed_shape<d, gdim>>& coordinate_dofs,
            xt::xtensor_fixed<double, xt::fixed_shape<ndofs_cell, ndofs_cell>>& Ae) {
          xt::xtensor<double, 2> J = xt::linalg::dot(coordinate_dofs, dphi0_c);
          double detJ = std::fabs(dolfinx::math::det(J));
          // Compute local matrix
          for (int i = 0; i < ndofs_cell; i++)
            for (int j = 0; j < ndofs_cell; j++)
              Ae(i, j) = A0(i, j) * detJ;
        };

  // Stiffness Matrix using quadrature formulation
  // =====================================================================================

  xt::xtensor<double, 3> _dphi({dphi.shape(1), dphi.shape(2), dphi.shape(0)});
  for (std::int32_t k = 0; k < tdim; k++)
    for (std::size_t q = 0; q < weights.size(); q++)
      for (std::int32_t i = 0; i < ndofs_cell; i++)
        _dphi(q, i, k) = dphi(k, q, i);

  kernel_fn<ndofs_cell> stiffness =
      [dphi0_c, _dphi, weights,
       ndofs_cell](xt::xtensor_fixed<double, xt::fixed_shape<d, gdim>>& coordinate_dofs,
                   xt::xtensor_fixed<double, xt::fixed_shape<ndofs_cell, ndofs_cell>>& Ae) {
        // Compute local matrix
        xt::xtensor_fixed<double, xt::fixed_shape<gdim, tdim>> J;
        xt::xtensor_fixed<double, xt::fixed_shape<tdim, gdim>> K;

        dot34(dphi0_c, coordinate_dofs, J);
        dolfinx::math::inv(J, K);
        double detJ = std::fabs(dolfinx::math::det(J));

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
              Ae(i, j) += (d0[i] * d0[j] + d1[i] * d1[j] + d2[i] * d2[j]) * w0;
            }
          }
        }
      };

  // Stiffness Matrix using tensor representation
  // =====================================================================================

  kernel_fn<ndofs_cell> stiffness_tensor
      = [=](xt::xtensor_fixed<double, xt::fixed_shape<d, gdim>>& coordinate_dofs,
            xt::xtensor_fixed<double, xt::fixed_shape<ndofs_cell, ndofs_cell>>& Ae) {
          // form_cell_integral_otherwise(coordinate_dofs, Ae);
        };

  if (type == Kernel::Mass)
  {
    if (repr == Representation::Quadrature)
      return mass;
    else
      return mass_tensor;
  }
  else if (type == Kernel::Stiffness)
  {
    if (repr == Representation::Quadrature)
      return stiffness;
    else
      return stiffness_tensor;
  }
  else
    throw std::runtime_error("Unknown kernel");
}
