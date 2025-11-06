set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/circt-cf/Optimize/Moore/Passes.td)
mlir_tablegen(circt-cf/Optimize/Moore/Passes.h.inc
  -gen-pass-decls)
add_public_tablegen_target(CIRCTCFMoorePassIncGen)

add_circt_library(CIRCTCFMooreTransforms
  FoldConstantBranches.cpp
  FoldStaticRegisters.cpp
  MergeProcedures.cpp
  NormalizeProcedures.cpp

  DEPENDS
  CIRCTCFMoorePassIncGen

  LINK_LIBS PUBLIC
  CIRCTMoore
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRSupport
  )
