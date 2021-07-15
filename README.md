# DOLFINx_CUAS
Authors: Igor A. Baratta, Jørgen S. Dokken, Sarah Roggendorf

Add-on to DOLFINx for assembly of custom kernels.
This packages provides a set of specialized kernels that can be used along-side DOLFINx for assembly into PETSc matrices.
See: `python/tests/test_cpp_kernels.py` for examples on how to interface with the Python-layer.

See: `cpp/demo/main.cpp` for how to interface with the C++ layer.

Currently implemented kernels for Lagrange elements (affine meshes):
- `ufl.inner(u, v) * ufl.dx`
- `ufl.inner(u, v) * ufl.ds`
- `ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx`
- `ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.ds`
- `ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.ds`

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