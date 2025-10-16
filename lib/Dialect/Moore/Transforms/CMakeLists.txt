set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/circt-cfa-trace/Dialect/Moore/Transforms/Passes.td)
mlir_tablegen(circt-cfa-trace/Dialect/Moore/Transforms/Passes.h.inc
  -gen-pass-decls)
add_public_tablegen_target(CIRCTCFATraceMoorePassIncGen)

add_circt_library(CIRCTCFATraceMooreTransforms
  MergeProcedures.cpp
  NormalizeProcedures.cpp

  DEPENDS
  CIRCTCFATraceMoorePassIncGen

  LINK_LIBS PUBLIC
  CIRCTMoore
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRSupport
  )
