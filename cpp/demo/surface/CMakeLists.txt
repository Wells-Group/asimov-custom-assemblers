
cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME custom-assembler)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Ofast -march=native ${CMAKE_CXX_FLAGS} -Wall")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

find_package(DOLFINX REQUIRED)
find_package(Basix REQUIRED)
find_package(xtensor REQUIRED)

# Add target to compile UFL files
if (PETSC_SCALAR_COMPLEX EQUAL 1)
  set(SCALAR_TYPE "--scalar_type=double _Complex")
endif()

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/problem.c
  COMMAND ffcx ${SCALAR_TYPE} ${CMAKE_CURRENT_SOURCE_DIR}/problem.ufl -o ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS problem.ufl
)
add_executable(${PROJECT_NAME} main.cpp ${CMAKE_CURRENT_SOURCE_DIR}/problem.c)



target_link_directories(${PROJECT_NAME} PRIVATE dolfinx_cuas)
target_link_libraries(${PROJECT_NAME} PRIVATE dolfinx)

target_link_libraries(${PROJECT_NAME} PRIVATE ${xtensor_blas_LIBRARY})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LAPACK_LIBRARIES})

