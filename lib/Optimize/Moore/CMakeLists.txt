set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/pcov/Optimize/Moore/Passes.td)
mlir_tablegen(pcov/Optimize/Moore/Passes.h.inc
  -gen-pass-decls)
add_public_tablegen_target(PCOVMoorePassIncGen)

add_circt_library(PCOVMooreTransforms
  FoldConstantBranches.cpp
  FoldStaticRegisters.cpp
  MergeProcedures.cpp
  NormalizeProcedures.cpp

  DEPENDS
  PCOVMoorePassIncGen

  LINK_LIBS PUBLIC
  CIRCTMoore
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRSupport
  )
