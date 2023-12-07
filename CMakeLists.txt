cmake_minimum_required(VERSION 3.10)
project(fastimagefilter VERSION 1.0 LANGUAGES CXX)

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
find_package(OpenCV HINTS "${OPENCV_DIR}" REQUIRED)

# Only link with required opencv modules.
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${GDAL_INCLUDE_DIR})

add_executable(fastimagefilter main.cpp vendor/mf2d/filter.cpp)
target_link_libraries(fastimagefilter ${STDPPFS_LIBRARY} OpenMP::OpenMP_CXX ${GDAL_LIBRARY})
install(TARGETS fastimagefilter RUNTIME DESTINATION bin)