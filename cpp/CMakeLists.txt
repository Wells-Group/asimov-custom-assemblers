# Top level CMakeLists for custom assemblers
cmake_minimum_required(VERSION 3.10)

# Set project name and version number
project(DOLFINX_CUAS VERSION 0.3.1.0)

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
find_package(DOLFINX 0.3.1.0 REQUIRED)
find_package(Basix 0.3.1.0 REQUIRED)
find_package(xtensor REQUIRED)

feature_summary(WHAT ALL)

add_library(dolfinx_cuas INTERFACE)

target_link_libraries(dolfinx_cuas INTERFACE Basix::basix)
target_include_directories(dolfinx_cuas INTERFACE ${DOLFINX_CUAS_SOURCE_DIR})

# Installation of header-only DOLFINX_CUAS Library
include(GNUInstallDirs)
install(FILES kernels.hpp kernelwrapper.h  math.hpp matrix_assembly.hpp vector_assembly.hpp utils.hpp surface_kernels.hpp kernels_non_const_coefficient.hpp vector_kernels.hpp QuadratureRule.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dolfinx_cuas COMPONENT Development)

# Create package config for CMAKE
include(CMakePackageConfigHelpers)

configure_package_config_file(${DOLFINX_CUAS_SOURCE_DIR}/DOLFINX_CUASConfig.cmake.in
${CMAKE_BINARY_DIR}/dolfinx_cuas/DOLFINX_CUASConfig.cmake
INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/dolfinx_cuas)

write_basic_package_version_file(${CMAKE_BINARY_DIR}/dolfinx_cuas/DOLFINX_CUASConfigVersion.cmake
VERSION 0.3.1.0 COMPATIBILITY SameMajorVersion)

# Install CMake helper files
install(
  FILES
  ${CMAKE_BINARY_DIR}/dolfinx_cuas/DOLFINX_CUASConfig.cmake
  ${CMAKE_BINARY_DIR}/dolfinx_cuas/DOLFINX_CUASConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/dolfinx_cuas
  COMPONENT Development)

#------------------------------------------------------------------------------
