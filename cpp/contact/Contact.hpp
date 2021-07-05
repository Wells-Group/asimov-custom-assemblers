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
                                     const int*, const std::uint8_t*)>;

namespace dolfinx_cuas
{
namespace contact
{
enum Kernel
{
  Mass,
  Stiffness,
  Contact_Jac
};
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
      }
    }
  }

  kernel_fn
  generate_surface_kernel(int origin_meshtag,
                          Kernel type) //, double gamma, double theta, std::vector<double> n_2)
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
    auto facets = basix::cell::topology(basix_cell)[tdim - 1];

    // Create facet quadrature points
    auto quadrature_points
        = basix::quadrature::make_quadrature("default", basix_facet, _quadrature_degree).first;

    auto surface_element = basix::create_element("Lagrange", dolfinx_facet_str, degree);
    auto basix_element
        = basix::create_element("Lagrange", dolfinx::mesh::to_string(dolfinx_cell), degree);

    // tabulate on reference facet
    auto f_tab = surface_element.tabulate(1, quadrature_points);
    xt::xtensor<double, 2> dphi0_f
        = xt::round(xt::view(f_tab, xt::range(1, tdim + 1), 0, xt::all(), 0));

    // tabulate on reference cell
    // not quite the right quadrature points if jacobian non-constant
    auto qp_cell = xt::view(_qp_ref_facet, 0, xt::all(), xt::all());

    auto c_tab = basix_element.tabulate(1, qp_cell);
    xt::xtensor<double, 2> dphi0_c
        = xt::round(xt::view(c_tab, xt::range(1, tdim + 1), 0, xt::all(), 0));

    std::uint32_t num_quadrature_pts = _qp_ref_facet.shape(1);
    std::uint32_t num_facets = _qp_ref_facet.shape(0);
    std::shared_ptr<const dolfinx::fem::FiniteElement> element = _V->element();
    int bs = element->block_size();
    std::uint32_t num_local_dofs = element->space_dimension() / bs;
    xt::xtensor<double, 3> phi({num_facets, num_quadrature_pts, num_local_dofs});
    xt::xtensor<double, 4> dphi({num_facets, tdim, num_quadrature_pts, num_local_dofs});
    xt::xtensor<double, 4> cell_tab({tdim + 1, num_quadrature_pts, num_local_dofs, bs});
    auto ref_jacobians = basix::cell::facet_jacobians(basix_cell);
    const xt::xtensor<double, 2> x = basix::cell::geometry(basix_cell);
    // tabulate at quadrature points on facet
    for (int i = 0; i < num_facets; ++i)
    {
      auto phi_i = xt::view(phi, i, xt::all(), xt::all());
      auto dphi_i = xt::view(dphi, i, xt::all(), xt::all(), xt::all());
      auto q_facet = xt::view(_qp_ref_facet, i, xt::all(), xt::all());
      element->tabulate(cell_tab, q_facet, 1);
      phi_i = xt::view(cell_tab, 0, xt::all(), xt::all(), 0);
      dphi_i = xt::view(cell_tab, xt::range(1, tdim + 1), xt::all(), xt::all(), 0);
    }

    kernel_fn mass = [facets, dphi0_f, phi, gdim, tdim, fdim, bs, this](
                         double* A, const double* c, const double* w, const double* coordinate_dofs,
                         const int* entity_local_index, const std::uint8_t* quadrature_permutation)
    {
      // Compute Jacobian at each quadrature point
      xt::xtensor<double, 2> J = xt::zeros<double>({gdim, fdim});
      // TODO: In kernels hpp 4 is given as a const expr d. What does this mean?
      /// shape = {num dofs on surface element, gdim}
      std::array<std::size_t, 2> shape = {4, gdim};
      xt::xtensor<double, 2> coord
          = xt::adapt(coordinate_dofs, 4 * gdim, xt::no_ownership(), shape);
      dolfinx_cuas::math::compute_jacobian(
          dphi0_f, xt::view(coord, xt::keep(facets[*entity_local_index])), J);
      double detJ = std::fabs(dolfinx_cuas::math::compute_determinant(J));
      // Get number of dofs per cell
      std::int32_t ndofs_cell = phi.shape(2);
      // Main loop
      for (std::size_t q = 0; q < phi.shape(1); q++)
      {
        double w0 = _qw_ref_facet[q] * detJ;

        for (int i = 0; i < ndofs_cell; i++)
        {
          double w1 = w0 * phi(*entity_local_index, q, i);
          for (int j = 0; j < ndofs_cell; j++)
          {
            double value = w1 * phi(*entity_local_index, q, j);
            for (int k = 0; k < bs; k++)
            {
              A[(k + i * bs) * (ndofs_cell * bs) + k + j * bs] += value;
            }
          }
        }
      }
    };

    kernel_fn stiffness
        = [facets, dphi0_f, dphi, gdim, tdim, fdim, bs, dphi0_c,
           this](double* A, const double* c, const double* w, const double* coordinate_dofs,
                 const int* entity_local_index, const std::uint8_t* quadrature_permutation)
    {
      // Compute Jacobian at each quadrature point: currently assumed to be constant...
      xt::xtensor<double, 2> J_facet = xt::zeros<double>({gdim, fdim});
      xt::xtensor<double, 2> J = xt::zeros<double>({gdim, tdim});
      xt::xtensor<double, 2> K = xt::zeros<double>({tdim, gdim});

      // TODO: In kernels hpp 4 is given as a const expr d. What does this mean?
      /// shape = {num dofs on surface element, gdim}
      std::array<std::size_t, 2> shape = {4, gdim};
      xt::xtensor<double, 2> coord
          = xt::adapt(coordinate_dofs, 4 * gdim, xt::no_ownership(), shape);

      dolfinx_cuas::math::compute_jacobian(
          dphi0_f, xt::view(coord, xt::keep(facets[*entity_local_index])), J_facet);
      dolfinx_cuas::math::compute_jacobian(dphi0_c, coord, J);
      dolfinx_cuas::math::compute_inv(J, K);

      // Get number of dofs per cell
      std::int32_t ndofs_cell = dphi.shape(3);

      double detJ = std::fabs(dolfinx_cuas::math::compute_determinant(J_facet));

      xt::xtensor<double, 2> temp({gdim, ndofs_cell});
      // Main loop

      for (std::size_t q = 0; q < dphi.shape(2); q++)
      {
        double w0 = _qw_ref_facet[q] * detJ; //

        // precompute J^-T * dphi in temporary array temp
        for (int i = 0; i < ndofs_cell; i++)
        {

          for (int j = 0; j < gdim; j++)
          {
            temp(j, i) = 0;
            for (int k = 0; k < tdim; k++)
            {
              temp(j, i) += K(k, j) * dphi(*entity_local_index, k, q, i);
            }
          }
        }

        for (int i = 0; i < ndofs_cell; i++)
        {
          for (int j = 0; j < ndofs_cell; j++)
          {
            double value = 0;
            for (int k = 0; k < gdim; k++)
            {
              value += temp(k, i) * temp(k, j) * w0;
            }
            for (int k = 0; k < bs; k++)
            {
              A[(k + i * bs) * (ndofs_cell * bs) + k + j * bs] += value;
            }
          }
        }
      }
    };

    kernel_fn contact_jac
        = [facets, dphi0_f, dphi, gdim, tdim, fdim, bs, dphi0_c,
           this](double* A, const double* c, const double* w, const double* coordinate_dofs,
                 const int* entity_local_index, const std::uint8_t* quadrature_permutation)
    {
      assert(bs == tdim);
      // Compute Jacobian at each quadrature point: currently assumed to be constant...
      xt::xtensor<double, 2> J_facet = xt::zeros<double>({gdim, fdim});
      xt::xtensor<double, 2> J = xt::zeros<double>({gdim, tdim});
      xt::xtensor<double, 2> K = xt::zeros<double>({tdim, gdim});

      // TODO: In kernels hpp 4 is given as a const expr d. What does this mean?
      /// shape = {num dofs on surface element, gdim}
      std::array<std::size_t, 2> shape = {4, gdim};
      xt::xtensor<double, 2> coord
          = xt::adapt(coordinate_dofs, 4 * gdim, xt::no_ownership(), shape);

      dolfinx_cuas::math::compute_jacobian(
          dphi0_f, xt::view(coord, xt::keep(facets[*entity_local_index])), J_facet);
      dolfinx_cuas::math::compute_jacobian(dphi0_c, coord, J);
      dolfinx_cuas::math::compute_inv(J, K);

      // Get number of dofs per cell
      std::int32_t ndofs_cell = dphi.shape(3);

      double detJ = std::fabs(dolfinx_cuas::math::compute_determinant(J_facet));

      xt::xtensor<double, 2> temp({gdim, ndofs_cell});
      // Main loop

      for (std::size_t q = 0; q < dphi.shape(2); q++)
      {
        double w0 = _qw_ref_facet[q] * detJ; //

        // precompute J^-T * dphi in temporary array temp
        for (int i = 0; i < ndofs_cell; i++)
        {

          for (int j = 0; j < gdim; j++)
          {
            temp(j, i) = 0;
            for (int k = 0; k < tdim; k++)
            {
              temp(j, i) += K(k, j) * dphi(*entity_local_index, k, q, i);
            }
          }
        }
        // This currently corresponds to the term sym(grad(u)):sym(grad(v)) (see
        // https://www.overleaf.com/2212919918tbbqtnmnrynf for details)
        for (int i = 0; i < ndofs_cell; i++)
        {
          for (int j = 0; j < ndofs_cell; j++)
          {
            double value = 0;
            for (int k = 0; k < gdim; k++)
            {
              value += temp(k, i) * temp(k, j) * w0;
            }
            for (int k = 0; k < bs; k++)
            {
              for (int l = 0; l < bs; l++)
              {
                if (k == l)
                {
                  A[(k + i * bs) * (ndofs_cell * bs) + k + j * bs] += 0.5 * value;
                }
                A[(k + i * bs) * (ndofs_cell * bs) + l + j * bs]
                    += 0.5 * temp(k, i) * temp(l, j) * w0;
              }
            }
          }
        }
      }
    };
    switch (type)
    {
    case Kernel::Mass:
      return mass;
    case Kernel::Stiffness:
      return stiffness;
    case Kernel::Contact_Jac:
      return contact_jac;
    default:
      throw std::runtime_error("unrecognized kernel");
    }
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
