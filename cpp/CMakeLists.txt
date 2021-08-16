# Top level CMakeLists for custom assemblers
cmake_minimum_required(VERSION 3.10)

# Set project name and version number
project(DOLFINX_CUAS VERSION 0.1.1.0)

#------------------------------------------------------------------------------
# Set CMake options, see `cmake --help-policy CMP00xx`

cmake_policy(VERSION 3.10)
if (POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()
if (POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

# Use C++17
set(CMAKE_CXX_STANDARD 17)

# Require C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Do not enable compler-specific extensions
set(CMAKE_CXX_EXTENSIONS OFF)

#------------------------------------------------------------------------------
# General configuration
# Make sure CMake uses the correct DOLFINConfig.cmake for tests and demos
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_CURRENT_BINARY_DIR}/dolfinx_cuas)

#------------------------------------------------------------------------------
# configurable options for how we want to build

include(FeatureSummary)

option(BUILD_SHARED_LIBS "Build DOLFINX_CUAS with shared libraries." ON)
add_feature_info(BUILD_SHARED_LIBS BUILD_SHARED_LIBS "Build DOLFINX_CUAS with shared libraries.")

option(CMAKE_INSTALL_RPATH_USE_LINK_PATH "Add paths to linker search and installed rpath." ON)
add_feature_info(CMAKE_INSTALL_RPATH_USE_LINK_PATH CMAKE_INSTALL_RPATH_USE_LINK_PATH "Add paths to linker search and installed rpath.")

# Find packages
find_package(DOLFINX 0.1.1.0 REQUIRED)
find_package(Basix 0.1.1.0 REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtensor-blas REQUIRED)

feature_summary(WHAT ALL)
add_subdirectory(contact)

add_library(dolfinx_cuas INTERFACE)

# Basix
target_link_libraries(dolfinx_cuas INTERFACE Basix::basix)

# Installation of header-only DOLFINX_CUAS Library
include(GNUInstallDirs)
install(FILES kernels.hpp  math.hpp matrix_assembly.hpp vector_assembly.hpp utils.hpp utils.h surface_kernels.hpp kernels_non_const_coefficient.hpp vector_kernels.hpp QuadratureRule.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dolfinx_cuas COMPONENT Development)
install(FILES contact/Contact.hpp contact/contact_kernels.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dolfinx_cuas/contact COMPONENT Development)
