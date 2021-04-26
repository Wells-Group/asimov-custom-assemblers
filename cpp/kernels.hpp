#include "math.hpp"
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx/common/math.h>
#include <string>
#include <xtensor-blas/xlinalg.hpp>

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

using kernel_fn = std::function<void(xt::xtensor<double, 2>&, xt::xtensor<double, 2>&)>;

template <std::int32_t P>
std::function<void(xt::xtensor<double, 2>&, xt::xtensor<double, 2>&)>
generate_kernel(std::string family, std::string cell, Kernel type, Representation repr)
{
  constexpr std::int32_t gdim = 3;
  constexpr std::int32_t tdim = 3;

  int quad_degree = 0;
  if (type == Kernel::Stiffness)
    quad_degree = (P - 1) + (P - 1) + 1;
  else if (type == Kernel::Mass)
    quad_degree = 2 * P + 1;

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
      = xt::view(coordinate_basis, xt::range(1, tdim + 1), 0, xt::all(), 0);

  constexpr std::int32_t ndofs_cell = (P + 1) * (P + 2) * (P + 3) / 6;
  xt::xtensor<double, 2> J = xt::zeros<double>({gdim, tdim});
  xt::xtensor<double, 2> K = xt::zeros<double>({tdim, gdim});

  // Mass Matrix using quadrature formulation
  // =====================================================================================
  kernel_fn mass = [=](xt::xtensor<double, 2>& coordinate_dofs, xt::xtensor<double, 2>& Ae) {
    xt::xtensor<double, 2> J = xt::zeros<double>({gdim, tdim});
    dot34(dphi0_c, coordinate_dofs, J);
    double detJ = std::abs(dolfinx::math::det(J));

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

  kernel_fn mass_tensor = [=](xt::xtensor<double, 2>& coordinate_dofs, xt::xtensor<double, 2>& Ae) {
    xt::xtensor<double, 2> J = xt::linalg::dot(coordinate_dofs, dphi0_c);
    double detJ = std::abs(dolfinx::math::det(J));
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

  kernel_fn stiffness = [=](xt::xtensor<double, 2>& coordinate_dofs, xt::xtensor<double, 2>& Ae) {
    // Compute local matrix
    xt::xtensor<double, 2> J = xt::empty<double>({gdim, tdim});
    xt::xtensor<double, 2> K = xt::empty<double>({tdim, gdim});

    dot34(dphi0_c, coordinate_dofs, J);
    J = xt::transpose(J);
    dolfinx::math::inv(J, K);
    double detJ = std::abs(dolfinx::math::det(J));

    // K = xt::transpose(K);

    // Main loop
    for (std::size_t q = 0; q < weights.size(); q++)
    {
      for (std::int32_t i = 0; i < ndofs_cell; i++)
      {
        for (std::int32_t j = 0; j < ndofs_cell; j++)
        {
          double acc = 0;
          for (std::int32_t k = 0; k < gdim; k++)
            for (std::int32_t l = 0; l < tdim; l++)
              acc += K(k, l) * _dphi(q, i, k) * K(k, l) * _dphi(q, j, k);
          Ae(i, j) += weights[q] * acc * detJ;
        }
      }
    }
  };

  // Stiffness Matrix using tensor representation
  // =====================================================================================
  kernel_fn stiffness_tensor
      = [=](xt::xtensor<double, 2>& coordinate_dofs, xt::xtensor<double, 2>& Ae) {
          // Compute local matrix
          xt::xtensor<double, 2> J = xt::linalg::dot(coordinate_dofs, dphi0_c);
          xt::xtensor<double, 2> K = xt::empty<double>({tdim, gdim});
          dolfinx::math::inv(J, K);
          double detJ = std::abs(dolfinx::math::det(J));

          // Main loop
          for (std::size_t q = 0; q < weights.size(); q++)
          {
            for (std::int32_t i = 0; i < ndofs_cell; i++)
            {
              for (std::int32_t j = 0; j < ndofs_cell; j++)
              {
                double acc = 0;
                for (std::int32_t k = 0; k < gdim; k++)
                  for (std::int32_t l = 0; l < tdim; l++)
                    acc += K(k, l) * _dphi(q, i, k) * K(k, l) * _dphi(q, j, k);
                Ae(i, j) += weights[q] * acc * detJ;
              }
            }
          }
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