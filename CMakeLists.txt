cmake_minimum_required(VERSION 3.0)
project(ilu_solve)

find_package(amgcl)
find_package(VexCL)
find_package(Boost COMPONENTS program_options REQUIRED)

add_executable(ilu_solve ilu_solve.cpp)
target_include_directories(ilu_solve PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(ilu_solve PRIVATE amgcl::amgcl VexCL::CUDA ${Boost_LIBRARIES})
