// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINX_CUAS
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once
#include <dolfinx_cuas/assembly.hpp>

namespace cuas_wrappers
{

/// This class wraps kernels from C++ for use in pybind11,
/// as pybind automatically wraps std::function of pointers to ints,
/// which in turn cannot be transferred back to C++

class KernelWrapper
{
public:
  /// Wrap a Kernel
  KernelWrapper(kernel_fn kernel) : _kernel(kernel) {}

  /// Assignment operator
  KernelWrapper& operator=(kernel_fn kernel)
  {
    this->_kernel = kernel;
    return *this;
  }

  /// Get the C++ kernel
  kernel_fn get() { return _kernel; }

private:
  // The underlying communicator
  kernel_fn _kernel;
};
} // namespace cuas_wrappers
