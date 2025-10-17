cmake_minimum_required(VERSION 3.20)

project(circt_cfa_trace LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

find_package(MLIR REQUIRED CONFIG)
find_package(CIRCT REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using CIRCTConfig.cmake in: ${CIRCT_DIR}")

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

set(CIRCT_CFA_TRACE_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CIRCT_CFA_TRACE_MAIN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(CIRCT_CFA_TRACE_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
set(MLIR_TABLEGEN_OUTPUT_DIR ${CIRCT_CFA_TRACE_INCLUDE_DIR})

file(MAKE_DIRECTORY ${MLIR_TABLEGEN_OUTPUT_DIR})
file(MAKE_DIRECTORY ${MLIR_TABLEGEN_OUTPUT_DIR}/circt-cfa-trace/Instrumentation)
file(MAKE_DIRECTORY ${MLIR_TABLEGEN_OUTPUT_DIR}/circt-cfa-trace/Optimize)
file(MAKE_DIRECTORY ${MLIR_TABLEGEN_OUTPUT_DIR}/circt-cfa-trace/Optimize/Moore)
file(MAKE_DIRECTORY ${MLIR_TABLEGEN_OUTPUT_DIR}/circt-cfa-trace/Optimize/LLHD)

include_directories(${CIRCT_CFA_TRACE_MAIN_INCLUDE_DIR})
include_directories(${CIRCT_CFA_TRACE_INCLUDE_DIR})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CIRCT_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/lib/Instrumentation)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/lib/Optimize/Moore)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/lib/Optimize/LLHD)

add_subdirectory(lib)
add_subdirectory(tools)

option(CIRCT_CFA_TRACE_ENABLE_TESTS "Enable circt-cfa-trace lit tests" ON)
if(CIRCT_CFA_TRACE_ENABLE_TESTS)
  add_subdirectory(test)
endif()
