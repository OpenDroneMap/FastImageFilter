cmake_minimum_required(VERSION 3.10)
project(fastrasterfilter VERSION 1.0 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
      FORCE)
endif()

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if (NOT WIN32 AND NOT APPLE)
    set(STDPPFS_LIBRARY stdc++fs)
endif()

find_package(OpenMP REQUIRED)
find_package(GDAL REQUIRED)

if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64" OR ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "arm64")
  add_compile_definitions(ARM64)
endif()

include_directories(${GDAL_INCLUDE_DIR})

add_executable(fastrasterfilter main.cpp vendor/mf2d/filter.cpp)
target_link_libraries(fastrasterfilter ${STDPPFS_LIBRARY} OpenMP::OpenMP_CXX ${GDAL_LIBRARY})
install(TARGETS fastrasterfilter RUNTIME DESTINATION bin)