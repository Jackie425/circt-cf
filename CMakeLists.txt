cmake_minimum_required(VERSION 3.20)

project(circt_cf LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

find_package(MLIR REQUIRED CONFIG)
find_package(Boost REQUIRED COMPONENTS headers)
find_package(CIRCT REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using CIRCTConfig.cmake in: ${CIRCT_DIR}")
message(STATUS "Using Boost headers in: ${Boost_INCLUDE_DIRS}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CIRCT_CMAKE_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(AddCIRCT)
include(HandleLLVMOptions)

if(NOT CIRCT_BUILD_TOOLS)
  set(CIRCT_BUILD_TOOLS ON CACHE BOOL "Build CIRCT-based tools" FORCE)
endif()

set(CIRCT_CF_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CIRCT_CF_MAIN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(CIRCT_CF_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
set(MLIR_TABLEGEN_OUTPUT_DIR ${CIRCT_CF_INCLUDE_DIR})

file(MAKE_DIRECTORY ${MLIR_TABLEGEN_OUTPUT_DIR})
file(MAKE_DIRECTORY ${MLIR_TABLEGEN_OUTPUT_DIR}/circt-cf/Instrumentation)
file(MAKE_DIRECTORY ${MLIR_TABLEGEN_OUTPUT_DIR}/circt-cf/Optimize)
file(MAKE_DIRECTORY ${MLIR_TABLEGEN_OUTPUT_DIR}/circt-cf/Optimize/Moore)
file(MAKE_DIRECTORY ${MLIR_TABLEGEN_OUTPUT_DIR}/circt-cf/Optimize/LLHD)

include_directories(
  ${CIRCT_CF_MAIN_INCLUDE_DIR}
  ${CIRCT_CF_INCLUDE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/lib/Instrumentation
  ${CMAKE_CURRENT_BINARY_DIR}/lib/Optimize/Moore
  ${CMAKE_CURRENT_BINARY_DIR}/lib/Optimize/LLHD
)

include_directories(SYSTEM
  ${MLIR_INCLUDE_DIRS}
  ${CIRCT_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

add_subdirectory(lib)
add_subdirectory(tools)

option(CIRCT_CF_ENABLE_TESTS "Enable circt-cf lit tests" ON)
if(CIRCT_CF_ENABLE_TESTS)
  include(CTest)
  enable_testing()
  add_subdirectory(test)
endif()
