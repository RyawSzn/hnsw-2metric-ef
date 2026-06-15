cmake_minimum_required(VERSION 3.10...3.26)

# Added 'C' to LANGUAGES because HDF5's CMake module usually looks for the C compiler too.
project(hnswlib LANGUAGES CXX C)

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

# 1. Use find_package to automatically locate apt-installed packages
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)
# Require both C and CXX components for HDF5 so it finds libhdf5.so and libhdf5_cpp.so
find_package(HDF5 REQUIRED COMPONENTS C CXX)

# 2. Setup hnswlib target
add_library(hnswlib INTERFACE)
add_library(hnswlib::hnswlib ALIAS hnswlib)

target_include_directories(hnswlib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Link Eigen and Boost to the hnswlib interface using their modern CMake targets
target_link_libraries(hnswlib INTERFACE Eigen3::Eigen Boost::boost)

# --- (Your compiler flags and examples section stays the same) ---
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(HNSWLIB_EXAMPLES "Build examples and tests." ON)
else()
    option(HNSWLIB_EXAMPLES "Build examples and tests." OFF)
endif()

if(HNSWLIB_EXAMPLES)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        SET( CMAKE_CXX_FLAGS  "-O3 -DHAVE_CXX0X -openmp -fpic -ftree-vectorize" )
        check_cxx_compiler_flag("-march=native" COMPILER_SUPPORT_NATIVE_FLAG)
        if(COMPILER_SUPPORT_NATIVE_FLAG)
            SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native" )
            message("set -march=native flag")
        else()
            check_cxx_compiler_flag("-mcpu=apple-m1" COMPILER_SUPPORT_M1_FLAG)
            if(COMPILER_SUPPORT_M1_FLAG)
                SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1" )
                message("set -mcpu=apple-m1 flag")
            endif()
        endif()
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        SET( CMAKE_CXX_FLAGS  "-O3 -lrt -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=5" )
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        SET( CMAKE_CXX_FLAGS  "/O2 -DHAVE_CXX0X /W1 /openmp /EHsc" )
    endif()
endif()
# ------------------------------------------------------------------

# 3. Setup your custom run executable
add_executable(run experiments_driver/run.cpp)

# Add HDF5 include directories (found automatically by find_package)
target_include_directories(run PRIVATE ${HDF5_INCLUDE_DIRS})

# 4. Link everything together correctly
target_link_libraries(run PRIVATE
    hnswlib                 # Link your hnswlib library (you forgot this in your script!)
    Eigen3::Eigen           # Link Eigen
    Boost::boost            # Link Boost
    ${HDF5_CXX_LIBRARIES}   # Link libhdf5_cpp.so
    ${HDF5_LIBRARIES}       # Link libhdf5.so
)

add_executable(run_lid experiments_driver/run_lid.cpp)
target_include_directories(run_lid PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(run_lid PRIVATE
    hnswlib
    Eigen3::Eigen
    Boost::boost
    ${HDF5_CXX_LIBRARIES}
    ${HDF5_LIBRARIES}
)

add_executable(reader research/reader.cpp)
target_include_directories(reader PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(reader PRIVATE
    hnswlib
    Eigen3::Eigen
    Boost::boost
    ${HDF5_CXX_LIBRARIES}
    ${HDF5_LIBRARIES}
)

add_executable(lookup_dynamic research/lookup_dynamic.cpp)
target_include_directories(lookup_dynamic PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(lookup_dynamic PRIVATE
    hnswlib
    Eigen3::Eigen
    ${HDF5_CXX_LIBRARIES}
    ${HDF5_LIBRARIES}
)

add_executable(lookup_fast_probe research/lookup_fast_probe.cpp)
target_include_directories(lookup_fast_probe PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(lookup_fast_probe PRIVATE
    hnswlib
    Eigen3::Eigen
    ${HDF5_CXX_LIBRARIES}
    ${HDF5_LIBRARIES}
)

add_executable(lookup_2hop research/lookup_2hop.cpp)
target_include_directories(lookup_2hop PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(lookup_2hop PRIVATE
    hnswlib
    Eigen3::Eigen
    ${HDF5_CXX_LIBRARIES}
    ${HDF5_LIBRARIES}
)


add_executable(time_metrics research/time_metrics.cpp)
target_include_directories(time_metrics PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(time_metrics PRIVATE
    hnswlib
    Eigen3::Eigen
    ${HDF5_CXX_LIBRARIES}
    ${HDF5_LIBRARIES}
)
