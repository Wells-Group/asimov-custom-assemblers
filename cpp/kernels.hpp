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

using kernel_ = std::function<void(xt::xtensor<double, 2>&, xt::xtensor<double, 2>&)>;

template <std::int32_t P>
std::function<void(xt::xtensor<double, 2>&, xt::xtensor<double, 2>&)>
generate_kernel(std::string family, std::string cell, Kernel type)
{
  constexpr std::int32_t gdim = 3;
  constexpr std::int32_t tdim = 3;

  int quad_degree = 0;
  if (type == Kernel::Stiffness)
    quad_degree = (P - 1) + (P - 1) + 1;
  else if (type == Kernel::Mass)
    quad_degree = 2 * P + 1;

  auto [points, weights]
      = basix::quadrature::make_quadrature("default", basix::cell::str_to_type(cell), quad_degree);

  // Create Finite element for test and trial functions and tabulate shape functions
  basix::FiniteElement element = basix::create_element(family, cell, P);
  xt::xtensor<double, 4> basis = element.tabulate(1, points);
  xt::xtensor<double, 2> phi = xt::view(basis, 0, xt::all(), xt::all(), 0);
  xt::xtensor<double, 3> dphi = xt::view(basis, xt::range(1, tdim + 1), xt::all(), xt::all(), 0);

  // Get coordinate element from dolfinx
  basix::FiniteElement coordinate_element = basix::create_element("Lagrange", cell, 1);
  xt::xtensor<double, 4> c_basis = coordinate_element.tabulate(1, points);

  // FIXME: Add documentation about the dimensions of dphi0, transpose in relation to basix
  // definition for performance purposes
  xt::xtensor<double, 2> dphi0_c
      = xt::transpose(xt::view(c_basis, xt::range(1, tdim + 1), 0, xt::all(), 0));

  constexpr std::int32_t ndofs_cell = (P + 1) * (P + 2) * (P + 3) / 6;
  //   constexpr std::int32_t d = 4;

  // NOTE: transposed in relation to dolfinx, so we avoid tranposing
  // it for multiplifcation
  //  N0  N1, ..., Nd
  // [x0, x1, ..., xd]
  // [y0, y1, ..., yd]
  // [z0, z2, ..., zd]
  xt::xtensor<double, 2> J = xt::empty<double>({gdim, tdim});
  xt::xtensor<double, 2> K = xt::empty<double>({tdim, gdim});

  // Mass Matrix using quadrature formulation
  // =====================================================================================
  kernel_ mass = [=](xt::xtensor<double, 2>& coordinate_dofs, xt::xtensor<double, 2>& Ae) {
    xt::xtensor<double, 2> J = xt::linalg::dot(coordinate_dofs, dphi0_c);
    double detJ = std::abs(dolfinx::math::det(J));

    // Compute local matrix
    for (std::size_t q = 0; q < weights.size(); q++)
      for (int i = 0; i < ndofs_cell; i++)
        for (int j = 0; j < ndofs_cell; j++)
          Ae(i, j) += weights[q] * phi(q, i) * phi(q, j) * detJ;
  };

  // Mass matrix using tensor representation
  // =====================================================================================
  xt::xtensor<double, 2> A0 = xt::zeros<double>({ndofs_cell, ndofs_cell});

  // Compute local matrix for reference element
  for (std::size_t q = 0; q < weights.size(); q++)
    for (int i = 0; i < ndofs_cell; i++)
      for (int j = 0; j < ndofs_cell; j++)
        A0(i, j) += weights[q] * phi(q, i) * phi(q, j);

  kernel_ mass_tensor = [=](xt::xtensor<double, 2>& coordinate_dofs, xt::xtensor<double, 2>& Ae) {
    xt::xtensor<double, 2> J = xt::linalg::dot(coordinate_dofs, dphi0_c);
    double detJ = std::abs(dolfinx::math::det(J));
    // Compute local matrix
    for (int i = 0; i < ndofs_cell; i++)
      for (int j = 0; j < ndofs_cell; j++)
        Ae(i, j) = A0(i, j) * detJ;
  };

  // Mass Matrix using quadrature formulation
  // =====================================================================================
  kernel_ stiffness = [=](xt::xtensor<double, 2>& coordinate_dofs, xt::xtensor<double, 2>& Ae) {
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
              acc += K(k, l) * dphi(k, q, i) * K(k, l) * dphi(k, q, j);
          Ae(i, j) += weights[q] * acc * detJ;
        }
      }
    }
  };

  // Mass Matrix using quadrature formulation
  // =====================================================================================
  kernel_ stiffness_tensor
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
                    acc += K(k, l) * dphi(k, q, i) * K(k, l) * dphi(k, q, j);
                Ae(i, j) += weights[q] * acc * detJ;
              }
            }
          }
        };

  if (type == Kernel::Mass)
    return mass_tensor;
  else
    return stiffness;
}
