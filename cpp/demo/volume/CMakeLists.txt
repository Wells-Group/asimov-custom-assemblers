
cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME volume)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Ofast -march=native ${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_C_FLAGS "-Ofast -march=native ${CMAKE_CXX_FLAGS} -Wall")

find_package(DOLFINX REQUIRED)
find_package(Basix REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtensor-blas REQUIRED)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}.c
  COMMAND ffcx ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}.ufl -o ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${PROJECT_NAME}.ufl
)

add_executable(${PROJECT_NAME} main.cpp ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME})


target_link_directories(${PROJECT_NAME} PRIVATE dolfinx_cuas)
target_link_libraries(${PROJECT_NAME} PRIVATE dolfinx)

target_link_libraries(${PROJECT_NAME} PRIVATE ${xtensor_blas_LIBRARY})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LAPACK_LIBRARIES})

