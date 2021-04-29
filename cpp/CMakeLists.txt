cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME custom-assembler)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-DNDEBUG -Ofast -march=native ${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_C_FLAGS "-DNDEBUG -Ofast -march=native ${CMAKE_C_FLAGS} -Wall")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

find_package(DOLFINX REQUIRED)
find_package(Basix REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtensor-blas REQUIRED)



add_executable(lagrange lagrange.cpp ${CMAKE_CURRENT_SOURCE_DIR}/lagrange.c)

add_definitions(-DELEMENT_DEGREE=${ELEMENT_DEGREE})

target_link_libraries(lagrange PRIVATE dolfinx)
target_include_directories(lagrange SYSTEM PUBLIC ${xtensor_INCLUDE_DIRS})
target_include_directories(lagrange SYSTEM PRIVATE ${xtensor_blas_INCLUDE_DIR})
target_link_libraries(lagrange PRIVATE ${xtensor_blas_LIBRARY})
target_link_libraries(lagrange PRIVATE ${LAPACK_LIBRARIES})

message(${CMAKE_CXX_FLAGS})
message(${CMAKE_C_FLAGS})