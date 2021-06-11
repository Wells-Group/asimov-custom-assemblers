#include "../math.hpp"
#include <basix/cell.h>
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx.h>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <iostream>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtl/xspan.hpp>

using kernel_fn = std::function<void(double*, const double*, const double*, const double*,
                                     const int*, const std::uint8_t*, const std::uint32_t)>;

namespace dolfinx_cuas
{
namespace contact
{
class Contact
{
public:
  /// Constructor
  /// @param[in] marker The meshtags defining the contact surfaces
  /// @param[in] surface_0 Value of the meshtag marking the first surface
  /// @param[in] surface_1 Value of the meshtag marking the first surface
  Contact(std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>> marker, int surface_0,
          int surface_1, std::shared_ptr<dolfinx::fem::FunctionSpace> V)
      : _marker(marker), _surface_0(surface_0), _surface_1(surface_1), _V(V)
  {
    // Extract facets in _surface_0 and _surface_1. Store in _facet_1 and _facet_2
    for (int i = 0; i < _marker->indices().size(); ++i)
    {
      const std::int32_t tag = _marker->values()[i];
      const std::int32_t facet = _marker->indices()[i];
      if (tag == _surface_0)
        _facet_0.push_back(facet);
      else if (tag == _surface_1)
        _facet_1.push_back(facet);
    }
  }

  // Return Adjacency list of closest facet on surface_1 for every quadrature point in _qp_phys_0
  // (quadrature points on every facet of surface_0)
  const std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> map_0_to_1() const
  {
    return _map_0_to_1;
  }
  // Return Adjacency list of closest facet on surface_0 for every quadrature point in _qp_phys_1
  // (quadrature points on every facet of surface_1)
  const std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> map_1_to_0() const
  {
    return _map_1_to_0;
  }
  const std::vector<int32_t> facet_0() const { return _facet_0; }
  const std::vector<int32_t> facet_1() const { return _facet_1; }
  // Return meshtag value for surface_0
  const int surface_0() const { return _surface_0; }
  // Return mestag value for suface_1
  const int surface_1() const { return _surface_1; }
  // return quadrature degree
  const int quadrature_degree() const { return _quadrature_degree; }

  // Return meshtags
  std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>> meshtags() const { return _marker; }

  /// Compute the push forward of the quadrature points on the reference facet
  /// to the reference cell.
  /// This creates and fills _qp_ref_facet
  void create_reference_facet_qp()
  {
    // Mesh info
    auto mesh = _marker->mesh();             // mesh
    const int gdim = mesh->geometry().dim(); // geometrical dimension
    const int tdim = mesh->topology().dim(); // topological dimension
    const int fdim = tdim - 1;               // topological dimesnion of facet
    // FIXME: Need basix element public in mesh
    // auto degree = mesh->geometry().cmap()._element->degree;
    int degree = 1; // element degree

    auto dolfinx_cell = mesh->topology().cell_type(); // doffinx cell type
    auto basix_cell
        = basix::cell::str_to_type(dolfinx::mesh::to_string(dolfinx_cell)); // basix cell type
    auto dolfinx_facet
        = dolfinx::mesh::cell_entity_type(dolfinx_cell, fdim);        // dolfinx facet cell type
    auto dolfinx_facet_str = dolfinx::mesh::to_string(dolfinx_facet); // facet cell type as string
    auto basix_facet = basix::cell::str_to_type(dolfinx_facet_str);   // basix facet cell type

    // Connectivity to evaluate at vertices
    mesh->topology_mutable().create_connectivity(fdim, 0);
    auto f_to_v = mesh->topology().connectivity(fdim, 0);

    // Create reference topology and geometry
    auto facet_topology = basix::cell::topology(basix_cell)[fdim];
    auto ref_geom = basix::cell::geometry(basix_cell);

    // Create facet quadrature points
    auto [quadrature_points, weights]
        = basix::quadrature::make_quadrature("default", basix_facet, _quadrature_degree);

    // Create basix surface element and tabulate basis functions
    auto surface_element = basix::create_element("Lagrange", dolfinx_facet_str, degree);
    auto c_tab = surface_element.tabulate(0, quadrature_points);
    xt::xtensor<double, 2> phi_s = xt::view(c_tab, 0, xt::all(), xt::all(), 0);

    // Push forward quadrature points on reference facet to reference cell
    const std::uint32_t num_facets = facet_topology.size();
    const std::uint32_t num_quadrature_pts = quadrature_points.shape(0);
    _qp_ref_facet = xt::xtensor<double, 3>({num_facets, num_quadrature_pts, ref_geom.shape(1)});
    _qw_ref_facet = std::vector<double>(weights);
    for (int i = 0; i < num_facets; ++i)
    {
      auto facet = facet_topology[i];
      auto coords = xt::view(ref_geom, xt::keep(facet), xt::all());
      auto q_facet = xt::view(_qp_ref_facet, i, xt::all(), xt::all());
      q_facet = xt::linalg::dot(phi_s, coords);
    }
  }
  /// Tabulatethe basis function at the quadrature points _qp_ref_facet
  /// creates and fills _phi_ref_facets
  xt::xtensor<double, 3> tabulate_on_ref_cell(basix::FiniteElement element)
  {

    // Create _phi_ref_facets
    std::uint32_t num_facets = _qp_ref_facet.shape(0);
    std::uint32_t num_quadrature_pts = _qp_ref_facet.shape(1);
    std::uint32_t num_local_dofs = element.dim();
    xt::xtensor<double, 3> phi({num_facets, num_quadrature_pts, num_local_dofs});

    // Tabulate basis functions at quadrature points _qp_ref_facet for each facet of the referenec
    // cell. Fill _phi_ref_facets
    for (int i = 0; i < num_facets; ++i)
    {
      auto phi_i = xt::view(phi, i, xt::all(), xt::all());
      auto q_facet = xt::view(_qp_ref_facet, i, xt::all(), xt::all());
      auto cell_tab = element.tabulate(0, q_facet);
      phi_i = xt::view(cell_tab, 0, xt::all(), xt::all(), 0);
    }
    return phi;
  }

  /// Compute push forward of quadrature points _qp_ref_facet to the physical facet for
  /// each facet in _facet_"origin_meshtag" Creates and fills _qp_phys_"origin_meshtag"
  /// @param[in] origin_meshtag flag to choose between surface_0 and  surface_1
  void create_q_phys(int origin_meshtag)
  {
    // Mesh info
    auto mesh = _marker->mesh();
    auto mesh_geometry = mesh->geometry().x();
    auto cmap = mesh->geometry().cmap();
    auto x_dofmap = mesh->geometry().dofmap();
    const int gdim = mesh->geometry().dim(); // geometrical dimension
    const int tdim = mesh->topology().dim(); // topological dimension
    const int fdim = tdim - 1;               // topological dimension of facet

    // Connectivity to evaluate at quadrature points
    mesh->topology_mutable().create_connectivity(fdim, tdim);
    auto f_to_c = mesh->topology().connectivity(fdim, tdim);
    mesh->topology_mutable().create_connectivity(tdim, fdim);
    auto c_to_f = mesh->topology().connectivity(tdim, fdim);

    std::vector<int32_t>* puppet_facets; // pointer to point to _facet_"origin_meshtag"
    xt::xtensor<double, 3>* q_phys_pt;   // pointer to point to _qp_phys_"origin_meshtag"
    if (origin_meshtag == 0)
    {
      puppet_facets = &_facet_0;
      // create _qp_phys_0
      _qp_phys_0 = xt::xtensor<double, 3>(
          {(*puppet_facets).size(), _qp_ref_facet.shape(1), _qp_ref_facet.shape(2)});
      q_phys_pt = &_qp_phys_0;
    }
    else
    {
      puppet_facets = &_facet_1;
      // create _qp_phys_1
      _qp_phys_1 = xt::xtensor<double, 3>(
          {(*puppet_facets).size(), _qp_ref_facet.shape(1), _qp_ref_facet.shape(2)});
      q_phys_pt = &_qp_phys_1;
    }

    // push forward of quadrature points _qp_ref_facet to physical facet for each facet in
    // _facet_"origin_meshtag"
    xt::xtensor<double, 2> q_phys({_qp_ref_facet.shape(1), _qp_ref_facet.shape(2)});
    for (int i = 0; i < (*puppet_facets).size(); ++i)
    {
      int facet = (*puppet_facets)[i]; // extract facet
      auto cells = f_to_c->links(facet);
      // since the facet is on the boundary it should only link to one cell
      assert(cells.size() == 1);
      auto cell = cells[0]; // extract cell

      // find local index of facet
      auto facets = c_to_f->links(cell);
      auto local_facet = std::find(facets.begin(), facets.end(), facet);
      const std::int32_t local_index = std::distance(facets.data(), local_facet);

      // extract local dofs
      auto x_dofs = x_dofmap.links(cell);
      auto coordinate_dofs = xt::view(mesh_geometry, xt::keep(x_dofs), xt::range(0, gdim));
      auto phi_facet = xt::view(_phi_ref_facets, local_index, xt::all(), xt::all());
      xt::xtensor<double, 2> q_phys({_qp_ref_facet.shape(1), _qp_ref_facet.shape(2)});

      // push forward of quadrature points _qp_ref_facet to the physical facet
      cmap.push_forward(q_phys, coordinate_dofs, phi_facet);
      xt::view(*q_phys_pt, i, xt::all(), xt::all()) = q_phys;
    }
  }

  /// Compute closest candidate_facet for each quadrature point in _qp_phys_"origin_meshtag"
  /// This is saved as an adjacency list _map_0_to_1 or _map_1_to_0
  void create_distance_map(int origin_meshtag)
  {
    // Create _qp_ref_facet (quadrature points on reference facet)
    create_reference_facet_qp();
    // Tabulate basis function on reference cell (_phi_ref_facets)// Create coordinate element
    // FIXME: For higher order geometry need basix element public in mesh
    // auto degree = mesh->geometry().cmap()._element->degree;
    int degree = 1;
    auto dolfinx_cell = _marker->mesh()->topology().cell_type();
    auto coordinate_element
        = basix::create_element("Lagrange", dolfinx::mesh::to_string(dolfinx_cell), degree);

    _phi_ref_facets = tabulate_on_ref_cell(coordinate_element);
    // Compute quadrature points on physical facet _qp_phys_"origin_meshtag"
    create_q_phys(origin_meshtag);

    // Mesh info
    auto mesh = _marker->mesh();
    const int gdim = mesh->geometry().dim();
    const int tdim = mesh->topology().dim();
    const int fdim = tdim - 1;

    std::array<double, 3> point;
    point[2] = 0;

    // assign puppet_ and candidate_facets
    std::vector<int32_t>* candidate_facets;
    std::vector<int32_t>* puppet_facets;
    xt::xtensor<double, 3>* q_phys_pt;
    if (origin_meshtag == 0)
    {
      puppet_facets = &_facet_0;
      candidate_facets = &_facet_1;
      q_phys_pt = &_qp_phys_0;
    }
    else
    {
      puppet_facets = &_facet_1;
      candidate_facets = &_facet_0;
      q_phys_pt = &_qp_phys_1;
    }
    // Create midpoint tree as compute_closest_entity will be called many times
    dolfinx::geometry::BoundingBoxTree master_bbox(*mesh, fdim, (*candidate_facets));
    auto master_midpoint_tree
        = dolfinx::geometry::create_midpoint_tree(*mesh, fdim, (*candidate_facets));

    std::vector<std::int32_t> data; // will contain closest candidate facet
    std::vector<std::int32_t> offset(1);
    offset[0] = 0;
    for (int i = 0; i < (*puppet_facets).size(); ++i)
    {
      for (int j = 0; j < (*q_phys_pt).shape(1); ++j)
      {
        for (int k = 0; k < gdim; ++k)
          point[k] = (*q_phys_pt)(i, j, k);
        // Find initial search radius R = intermediate_result.second
        std::pair<int, double> intermediate_result
            = dolfinx::geometry::compute_closest_entity(master_midpoint_tree, point, *mesh);
        // Find closest facet to point
        std::pair<int, double> search_result = dolfinx::geometry::compute_closest_entity(
            master_bbox, point, *mesh, intermediate_result.second);
        data.push_back(search_result.first);
      }
      offset.push_back(data.size());
    }

    // save as an adjacency list _map_0_to_1 or _map_1_to_0
    if (origin_meshtag == 0)
      _map_0_to_1 = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(data, offset);
    else
      _map_1_to_0 = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(data, offset);

    evaluate(origin_meshtag);
  }

  void evaluate(int origin_meshtag)
  {
    // Mesh info
    auto mesh = _marker->mesh();             // mesh
    const int gdim = mesh->geometry().dim(); // geometrical dimension
    const int tdim = mesh->topology().dim();
    const int fdim = tdim - 1;
    auto mesh_geometry = mesh->geometry().x();
    std::vector<int32_t>* puppet_facets;
    std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> map;
    xt::xtensor<double, 3>* q_phys_pt;
    if (origin_meshtag == 0)
    {
      puppet_facets = &_facet_0;
      map = _map_0_to_1;
      q_phys_pt = &_qp_phys_0;
    }
    else
    {
      puppet_facets = &_facet_1;
      map = _map_1_to_0;
      q_phys_pt = &_qp_phys_1;
    }
    // std::cout << "dist c++ \n";
    for (int i = 0; i < (*puppet_facets).size(); ++i)
    {
      auto links = map->links(i);
      auto master_facet_geometry = dolfinx::mesh::entities_to_geometry(*mesh, fdim, links, false);

      for (int j = 0; j < map->num_links(i); ++j)
      {
        xt::xtensor<double, 2> point = {{0, 0, 0}};
        for (int k = 0; k < gdim; k++)
          point(0, k) = (*q_phys_pt)(i, j, k);
        auto master_facet = xt::view(master_facet_geometry, j, xt::all());
        auto master_coords = xt::view(mesh_geometry, xt::keep(master_facet), xt::all());

        auto dist_vec = dolfinx::geometry::compute_distance_gjk(master_coords, point);
        // std::cout << dist_vec << "\n";
      }
    }
  }

  kernel_fn generate_surface_kernel(
      int origin_meshtag) //, double gamma, double theta, std::vector<double> n_2)
  {
    // Starting with implementing the following term in Jacobian:
    // u*v*ds
    // Mesh info
    auto mesh = _marker->mesh();             // mesh
    const int gdim = mesh->geometry().dim(); // geometrical dimension
    const int tdim = mesh->topology().dim(); // topological dimension
    const int fdim = tdim - 1;               // topological dimesnion of facet
    // FIXME: Need basix element public in mesh
    // auto degree = mesh->geometry().cmap()._element->degree;
    int degree = 1; // element degree

    auto dolfinx_cell = mesh->topology().cell_type(); // doffinx cell type
    auto basix_cell
        = basix::cell::str_to_type(dolfinx::mesh::to_string(dolfinx_cell)); // basix cell type
    auto dolfinx_facet
        = dolfinx::mesh::cell_entity_type(dolfinx_cell, fdim);        // dolfinx facet cell type
    auto dolfinx_facet_str = dolfinx::mesh::to_string(dolfinx_facet); // facet cell type as string
    auto basix_facet = basix::cell::str_to_type(dolfinx_facet_str);   // basix facet cell type

    // Connectivity to evaluate at quadrature points
    mesh->topology_mutable().create_connectivity(fdim, tdim);
    auto f_to_c = mesh->topology().connectivity(fdim, tdim);
    mesh->topology_mutable().create_connectivity(tdim, fdim);
    auto c_to_f = mesh->topology().connectivity(tdim, fdim);

    // Create facet quadrature points
    auto quadrature_points
        = basix::quadrature::make_quadrature("default", basix_facet, _quadrature_degree).first;

    auto surface_element = basix::create_element("Lagrange", dolfinx_facet_str, degree);
    auto basix_element
        = basix::create_element("Lagrange", dolfinx::mesh::to_string(dolfinx_cell), degree);
    auto c_tab = surface_element.tabulate(1, quadrature_points);
    xt::xtensor<double, 2> dphi0_c
        = xt::round(xt::view(c_tab, xt::range(1, tdim + 1), 0, xt::all(), 0));
    std::uint32_t num_facets = _qp_ref_facet.shape(0);
    std::uint32_t num_quadrature_pts = _qp_ref_facet.shape(1);
    std::shared_ptr<const dolfinx::fem::FiniteElement> element = _V->element();
    std::uint32_t num_local_dofs = element->space_dimension();
    xt::xtensor<double, 3> phi({num_facets, num_quadrature_pts, num_local_dofs});
    xt::xtensor<double, 4> dphi({num_facets, tdim, num_quadrature_pts, num_local_dofs});
    xt::xtensor<double, 4> cell_tab({tdim + 1, num_quadrature_pts, num_local_dofs, 1});
    auto ref_jacobians = basix::cell::facet_jacobians(basix_cell);
    std::cout << "shape of ref jacobians " << ref_jacobians.shape(0) << ", "
              << ref_jacobians.shape(1) << ", " << ref_jacobians.shape(2) << "\n";
    for (int i = 0; i < num_facets; ++i)
    {
      auto phi_i = xt::view(phi, i, xt::all(), xt::all());
      auto dphi_i = xt::view(dphi, i, xt::all(), xt::all(), xt::all());
      auto q_facet = xt::view(_qp_ref_facet, i, xt::all(), xt::all());
      element->tabulate(cell_tab, q_facet, 1);
      phi_i = xt::view(cell_tab, 0, xt::all(), xt::all(), 0);
      dphi_i = xt::view(cell_tab, xt::range(1, tdim + 1), xt::all(), xt::all(), 0);
    }

    kernel_fn surface
        = [dphi0_c, phi, dphi, gdim, tdim, fdim, ref_jacobians, f_to_c, c_to_f,
           this](double* A, const double* c, const double* w, const double* coordinate_dofs,
                 const int* entity_local_index, const std::uint8_t* quadrature_permutation,
                 const std::uint32_t cell_permutation)
    {
      std::cout << "I am here \n";
      // Compute Jacobian at each quadrature point: currently assumed to be constant...
      xt::xtensor<double, 2> J = xt::zeros<double>({gdim, fdim});
      xt::xtensor<double, 2> K = xt::zeros<double>({fdim, gdim});
      // xt::xtensor<double, 2> M = xt::zeros<double>({tdim, gdim});
      // TODO: In kernels hpp 4 is given as a const expr d. What does this mean?
      /// shape = {num dofs on surface element, gdim}
      std::array<std::size_t, 2> shape = {3, gdim};
      std::cout << "this should not break me\n";
      xt::xtensor<double, 2> coord
          = xt::adapt(coordinate_dofs, 3 * gdim, xt::no_ownership(), shape);
      std::cout << " and yet it does \n";
      std::cout << dphi0_c << "\n";
      std::cout << J << "\n";
      std::cout << coord << "\n";
      dolfinx_cuas::math::compute_jacobian(dphi0_c, coord, J);
      std::cout << "jacobian shape " << J.shape(0) << ", " << J.shape(1) << "\n";
      double detJ = std::fabs(dolfinx_cuas::math::compute_determinant(J));
      dolfinx_cuas::math::inv(J, K);

      // Get number of dofs per cell
      std::int32_t ndofs_cell = phi.shape(2);

      // Get ref Jacobian
      auto cells = f_to_c->links(*entity_local_index);
      // since the facet is on the boundary it should only link to one cell
      assert(cells.size() == 1);
      auto cell = cells[0]; // extract cell

      // find local index of facet
      auto facets = c_to_f->links(cell);
      auto local_facet = std::find(facets.begin(), facets.end(), *entity_local_index);
      const std::int32_t local_index = std::distance(facets.data(), local_facet);

      auto ref_jacobian = xt::view(ref_jacobians, local_index, xt::all(), xt::all());
      std::cout << "dimensions of ref Jac: " << ref_jacobian.shape(0) << ", "
                << ref_jacobian.shape(1) << "\n";
      std::cout << "gdim: " << gdim << ", fdim: " << fdim << "\n";
      auto M = xt::linalg::dot(ref_jacobian, K);
      std::cout << M << "\n";
      // for (int i = 0; i < tdim; i++)
      // {
      //   for (int j = 0; j < gdim; j++)
      //   {
      //     for (int k = 0; k < fdim; k++)
      //     {
      //       M(i, j) += ref_jacobian(i, k) * K(k, j);
      //     }
      //   }
      // }

      xt::xtensor<double, 2> temp({tdim, ndofs_cell});
      std::cout << "num quadrature points: " << phi.shape(1) << "\n";
      // Main loop
      for (std::size_t q = 0; q < phi.shape(1); q++)
      {
        double w0 = _qw_ref_facet[q] * detJ;

        // precompute J^-T * dphi in temporary array temp
        for (int i = 0; i < ndofs_cell; i++)
        {
          for (int j = 0; j < tdim; j++)
          {
            temp(j, i) = 0;
            for (int k = 0; k < gdim; k++)
            {
              std::cout << "q " << q << ", i " << i << ", j " << j << ", k " << k << "\n";
              temp(j, i) += M(k, j) * dphi(q, j, i);
            }
          }
          // d0[i] = K(0, 0) * _dphi(q, i, 0) + K(1, 0) * _dphi(q, i, 1) + K(2, 0) * _dphi(q, i, 2);
          // d1[i] = K(0, 1) * _dphi(q, i, 0) + K(1, 1) * _dphi(q, i, 1) + K(2, 1) * _dphi(q, i, 2);
          // d2[i] = K(0, 2) * _dphi(q, i, 0) + K(1, 2) * _dphi(q, i, 1) + K(2, 2) * _dphi(q, i, 2);
        }

        for (int i = 0; i < ndofs_cell; i++)
        {
          // double w1 = w0 * phi(*entity_local_index, q, i);
          for (int j = 0; j < ndofs_cell; j++)
          {
            // A[i * ndofs_cell + j] += w1 * phi(*entity_local_index, q, j);
            for (int k = 0; k < tdim; k++)
            {
              A[i * ndofs_cell + j] += temp(k, i) * temp(k, j) * w0;
            }
          }
        }
      }
      std::cout << "The problem is further down in the code... \n";
    };

    return surface;
  }

private:
  int _quadrature_degree = 3;
  std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>> _marker;
  int _surface_0;                                  // meshtag value for surface 0
  int _surface_1;                                  // meshtag value for surface 1
  std::shared_ptr<dolfinx::fem::FunctionSpace> _V; // Function space

  // Adjacency list of closest facet on surface_1 for every quadrature point in _qp_phys_0
  // (quadrature points on every facet of surface_0)
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> _map_0_to_1;
  // Adjacency list of closest facet on surface_0 for every quadrature point in _qp_phys_1
  // (quadrature points on every facet of surface_1)
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> _map_1_to_0;
  // quadrature points on physical facet for each facet on surface 0
  xt::xtensor<double, 3> _qp_phys_0;
  // quadrature points on physical facet for each facet on surface 1
  xt::xtensor<double, 3> _qp_phys_1;
  // quadrature points on reference facet
  xt::xtensor<double, 3> _qp_ref_facet;
  // quadrature weights
  std::vector<double> _qw_ref_facet;
  // quadrature points on facets of reference cell
  xt::xtensor<double, 3> _phi_ref_facets;
  // facets in surface 0
  std::vector<int32_t> _facet_0;
  // facets in surface 1
  std::vector<int32_t> _facet_1;
};
} // namespace contact
} // namespace dolfinx_cuas
