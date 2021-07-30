# DOLFINx_CUAS
[![Test Python assemblers](https://github.com/Wells-Group/asimov-custom-assemblers/actions/workflows/python-app.yml/badge.svg)](https://github.com/Wells-Group/asimov-custom-assemblers/actions/workflows/python-app.yml)
[![Test C++ assemblers](https://github.com/Wells-Group/asimov-custom-assemblers/actions/workflows/cpp-app.yml/badge.svg)](https://github.com/Wells-Group/asimov-custom-assemblers/actions/workflows/cpp-app.yml)

Authors: Igor A. Baratta, Jørgen S. Dokken, Sarah Roggendorf

Add-on to DOLFINx for assembly of custom kernels.
This packages provides a set of specialized kernels that can be used along-side DOLFINx for assembly into PETSc matrices.
See: `python/tests/test_cpp_kernels.py` for examples on how to interface with the Python-layer.

See: `cpp/demo/main.cpp` for how to interface with the C++ layer.

# Affine meshes (triangles and tetrahedra)
## Lagrange elements (Scalar and Vector) (degree 1-5)
- `ufl.inner(u, v) * ufl.dx` (3D)
- `ufl.inner(u, v) * ufl.ds` (2D and 3D)
- `ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx` (3D)
- `ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.ds` (2D and 3D)
- `ufl.inner(ufl.tr(ufl.sym(ufl.grad(u))), ufl.sym(ufl.grad(v))) * ufl.dx` (3D)
- `2 * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx` (3D)
- `ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.ds` (2D and 3D)
- `2 * ufl.inner(ufl.sym(ufl.grad(u))) * ufl.FacetNormal(mesh), v) * ufl.ds` (2D and 3D)
- `v*ufl.dx` (3D)
- `v*ufl.ds` (2D and 3D)

# Non-affine meshes (quadrilaterals and hexahedra)
## Lagrange elements (degree 1-5)
- `ufl.inner(u, v) * ufl.ds`

## Dependencies
DOLFINX_CUAS depends on DOLFINx and in turn its dependencies.
For instructions on how to install DOLFINx, see: https://github.com/FEniCS/dolfinx#readme

## Installation
DOLFINX_CUAS is currently a header-only library.


### Clone repository
```bash
    git clone https://github.com/Wells-Group/asimov-custom-assemblers.git
```
### Build C++ layer
```bash
    cd asimov-custom-assemblers
    cmake -G Ninja -B build-dir -DCMAKE_BUILD_TYPE=Release cpp/
    ninja -C build-dir install
```
### Compile C++ demos
Navigate to the folder of a C++ demo, then call
```bash
    ffcx *.ufl
    cmake -G Ninja -B build-dir -DCMAKE_BUILD_TYPE=Release .
    ninja -C build-dir 
```
### Build Python-interface
```bash
    cd python
    pip3 install . -v --upgrade
```