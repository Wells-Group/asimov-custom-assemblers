
cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME volume)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Ofast -march=native ${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_C_FLAGS "-Ofast -march=native ${CMAKE_C_FLAGS} -Wall")

find_package(DOLFINX REQUIRED)
find_package(Basix REQUIRED)
find_package(xtensor REQUIRED)
find_package(DOLFINX_CUAS REQUIRED)


# Add target to compile UFL files
if (NOT DEFINED PETSC_SCALAR_COMPLEX)
  message(FATAL_ERROR "PETSc scalar type not defined")
endif()
if (PETSC_SCALAR_COMPLEX EQUAL 1)
  set(SCALAR_TYPE "--scalar_type=double _Complex")
endif()

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}.c
  COMMAND ffcx --scalar_type=${FFCX_TYPE} ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}.ufl -o ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${PROJECT_NAME}.ufl
)

add_executable(${PROJECT_NAME} main.cpp ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME})

# Find Boost program_options
if(DEFINED ENV{BOOST_ROOT} OR DEFINED BOOST_ROOT)
  set(Boost_NO_SYSTEM_PATHS on)
endif()
set(Boost_USE_MULTITHREADED $ENV{BOOST_USE_MULTITHREADED})
set(Boost_VERBOSE TRUE)
find_package(Boost 1.70 REQUIRED program_options)


target_link_directories(${PROJECT_NAME} PRIVATE dolfinx_cuas)
target_link_libraries(${PROJECT_NAME} PRIVATE dolfinx)

target_link_libraries(${PROJECT_NAME} PRIVATE ${xtensor_blas_LIBRARY})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LAPACK_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PRIVATE Boost::program_options)

