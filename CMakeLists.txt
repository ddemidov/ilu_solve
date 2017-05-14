cmake_minimum_required(VERSION 3.0)
project(ilu_solve)

find_package(amgcl)
find_package(Boost COMPONENTS program_options REQUIRED)

option(REORDER_MATRICES "Reorder matrices" ON)

add_executable(ilu_solve ilu_solve.cpp)
target_include_directories(ilu_solve PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(ilu_solve PRIVATE amgcl::amgcl ${Boost_LIBRARIES})

if (REORDER_MATRICES)
    target_compile_definitions(ilu_solve PRIVATE REORDER_MATRICES)
endif()
