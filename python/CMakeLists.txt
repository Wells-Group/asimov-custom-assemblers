cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME dolfinx_cuas_pybind11)
project(${PROJECT_NAME})


# Set C++ standard before finding pybind11
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 REQUIRED CONFIG HINTS ${PYBIND11_DIR} ${PYBIND11_ROOT}
  $ENV{PYBIND11_DIR} $ENV{PYBIND11_ROOT})


find_package(DOLFINX REQUIRED)


# Create the binding library
pybind11_add_module(cpp SHARED dolfinx_cuas/wrappers.cpp)


target_link_libraries(cpp PRIVATE pybind11::module dolfinx_cuas)
target_link_libraries(cpp PUBLIC dolfinx)

   
# Add to CMake search path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

# Add DOLFINX libraries and other config

