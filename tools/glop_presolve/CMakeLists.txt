cmake_minimum_required(VERSION 3.20)
project(glop_presolve LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point CMAKE_PREFIX_PATH at an OR-Tools C++ install (see build.sh).
find_package(ortools CONFIG REQUIRED)

add_executable(glop_presolve glop_presolve.cc)
target_link_libraries(glop_presolve PRIVATE ortools::ortools)
# Find libortools.so in the install tree at run time without LD_LIBRARY_PATH.
set_target_properties(glop_presolve PROPERTIES
  BUILD_RPATH "${ortools_DIR}/../.."
  INSTALL_RPATH_USE_LINK_PATH ON)
