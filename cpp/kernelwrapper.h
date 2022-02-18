// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINx_CUAS
//
// SPDX-License-Identifier:    MIT
#include "kernels.hpp"

namespace cuas_wrappers
{

/// This class wraps kernels from C++ for use in pybind11,
/// as pybind automatically wraps std::function of pointers to ints,
/// which in turn cannot be transferred back to C++

template <typename T>
class KernelWrapper
{
public:
  /// Wrap a Kernel
  KernelWrapper(dolfinx_cuas::kernel_fn<T> kernel) : _kernel(kernel) {}

  /// Assignment operator
  KernelWrapper& operator=(dolfinx_cuas::kernel_fn<T> kernel)
  {
    this->_kernel = kernel;
    return *this;
  }

  /// Get the C++ kernel
  dolfinx_cuas::kernel_fn<T> get() { return _kernel; }

private:
  // The underlying communicator
  dolfinx_cuas::kernel_fn<T> _kernel;
};
} // namespace cuas_wrappers
