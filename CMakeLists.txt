cmake_minimum_required(VERSION 3.25)
project(fast_traversal LANGUAGES CXX CUDA)
enable_language(CUDA)



# Add paths to our CMake code to the module path, so they can be found automatically by
# CMake.
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_BUILD_TYPE "Release")

find_library(libcnpy cnpy ${CONDA_PREFIX}/lib)
include_directories(${CONDA_PREFIX}/include)

# TODO: fix this
find_package(Torch REQUIRED)
find_package(CGAL REQUIRED)
# find_package(pybind11 REQUIRED)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.9.2
)
FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()


# Put all the runtime stuff in the same directory.  By default, CMake puts each targets'
# output into their own directory.  We want all the targets to be put in the same
# directory, and we can do this by setting these variables.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(BUILD_DIRECTORY "${CMAKE_BINARY_DIR}/build")
set(CUDA_GENERATED_OUTPUT_DIR "${BUILD_DIRECTORY}")

# Locate the NVRT distribution.  Search the SDK first, then look in the system.
set(OptiX_INSTALL_DIR "/opt/optix" CACHE PATH "Path to OptiX installed location.")
if(NOT "$ENV{OPTIX_PATH}" STREQUAL "")
    set(OptiX_INSTALL_DIR "$ENV{OPTIX_PATH}")
endif()

# Search for the OptiX libraries and include files.
find_package(OptiX REQUIRED)

# Add the path to the OptiX headers to our include paths.
include_directories(
  "${OptiX_INCLUDE}"
  "${CMAKE_CURRENT_SOURCE_DIR}/cuda"
)

#------------------------------------------------------------------------------
# set environment
#------------------------------------------------------------------------------

find_package(OptiX REQUIRED)
find_package(CUDA REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)
include_directories(${OptiX_INCLUDE} ${CUDA_INCLUDE_DIRS})

# configure_file("${CMAKE_SOURCE_DIR}/path_util.py.in" "${CMAKE_SOURCE_DIR}/../examples/path_util.py")


#------------------------------------------------------------------------------
# main build
#------------------------------------------------------------------------------

# pybind11_add_module(optix main.cpp)
find_program(BIN2C bin2c
  DOC "Path to the cuda-sdk bin2c executable.")


macro(cuda_compile_and_embed output_var generated_files)
    set(c_var_name ${output_var})
    list(GET generated_files 0 ptx_file)
    get_filename_component(embedded_file ${ptx_file}_embedded.cpp NAME)
    set(embedded_file "${BUILD_DIRECTORY}/${embedded_file}")
    add_custom_command(
        OUTPUT ${embedded_file}
        COMMAND ${BIN2C} --padd 0 --type char --name ${c_var_name} ${ptx_file} > ${embedded_file}
        DEPENDS ${ptx_files} ${generated_files}
        COMMENT "compiling (and embedding ptx from) ${cuda_file}")
    set(${output_var} ${embedded_file})
    list(APPEND generated_files ${embedded_file})
endmacro()


CUDA_WRAP_SRCS(traversal PTX generated_files 
    src/optix_types.h
    src/optix_traversal.cu
    src/utils/preprocessor.h
    src/utils/vec_math.h
    OPTIONS -rdc true)
cuda_compile_and_embed(ptx_code_file ${generated_files})

# configure_file(${generated_files} file.ptx.h)
add_link_options(-flto=auto)  # disables some warning
pybind11_add_module(tetranerf_cpp_extension 
    src/tetrahedra_tracer.cu 
    src/tetrahedra_tracer.cpp 
    src/triangulation.cpp
    src/py_binding.cpp ${generated_files})
# target_link_libraries( traversal PRIVATE
# #   OptiX::OptiX
# )
target_link_libraries(tetranerf_cpp_extension PRIVATE ${CUDA_LIBRARIES} ${TORCH_LIBRARIES})
target_link_libraries(tetranerf_cpp_extension PRIVATE CGAL::CGAL)
target_compile_features(tetranerf_cpp_extension PRIVATE cxx_std_17)
set_target_properties(tetranerf_cpp_extension PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})
set_target_properties(tetranerf_cpp_extension
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tetranerf/utils/extension"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
target_compile_definitions(tetranerf_cpp_extension PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})