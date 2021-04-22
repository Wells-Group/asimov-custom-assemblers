cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME custom-assembler)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Ofast -march=native -DNDEBUG ${CMAKE_CXX_FLAGS} -Wall")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

find_package(DOLFINX REQUIRED)
find_package(Basix REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtensor-blas REQUIRED)

add_executable(${PROJECT_NAME} main.cpp ${CMAKE_CURRENT_SOURCE_DIR}/mass.c)

target_link_libraries(${PROJECT_NAME} PRIVATE dolfinx)
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${xtensor_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE ${xtensor_blas_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE ${xtensor_blas_LIBRARY})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LAPACK_LIBRARIES})