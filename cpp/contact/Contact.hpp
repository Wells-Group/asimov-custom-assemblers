#include <basix/cell.h>
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/MeshTags.h>
#include <iostream>
#include <xtensor-blas/xlinalg.hpp>
#include <xtl/xspan.hpp>

namespace dolfinx_cuas
{
namespace contact
{
class Contact
{
public:
  Contact(std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>> marker, int surface_0,
          int surface_1)
      : _marker(marker), _surface_0(surface_0), _surface_1(surface_1)
  {
  }

private:
  int _quadrature_degree = 3;
  std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>> _marker;
  int _surface_0;
  int _surface_1;
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> _map_0_to_1;
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> _map_1_to_0;
  xt::xtensor<double, 3> _qp_phys;
};
} // namespace contact
} // namespace dolfinx_cuas