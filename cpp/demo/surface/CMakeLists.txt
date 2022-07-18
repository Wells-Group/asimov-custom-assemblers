cmake_minimum_required(VERSION 3.16)

set(PROJECT_NAME custom-assembler)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Ofast -march=native ${CMAKE_CXX_FLAGS} -Wall")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

find_package(DOLFINX REQUIRED)
find_package(Basix REQUIRED)
find_package(xtensor REQUIRED)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(UFL_FILE problem)


# Add target to compile UFL files
if (PETSC_SCALAR_COMPLEX EQUAL 1)
  set(SCALAR_TYPE "--scalar_type=double _Complex")
endif()
# Create compilation command
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${UFL_FILE}.c
  COMMAND ffcx ${SCALAR_TYPE} ${CMAKE_CURRENT_SOURCE_DIR}/${UFL_FILE}.py -o ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS  ${CMAKE_CURRENT_SOURCE_DIR}/${UFL_FILE}.py
  COMMENT "Compiling ${UFL_FILE}.py"
)
add_executable(${PROJECT_NAME} main.cpp ${UFL_FILE}.c)



target_link_directories(${PROJECT_NAME} PRIVATE dolfinx_cuas)
target_link_libraries(${PROJECT_NAME} PRIVATE dolfinx)

target_link_libraries(${PROJECT_NAME} PRIVATE ${xtensor_blas_LIBRARY})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LAPACK_LIBRARIES})

