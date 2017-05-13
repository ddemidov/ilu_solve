cmake_minimum_required(VERSION 3.0)
project(ilu_solve)

find_package(amgcl)

add_executable(ilu_solve ilu_solve.cpp)
target_link_libraries(ilu_solve amgcl::amgcl)
