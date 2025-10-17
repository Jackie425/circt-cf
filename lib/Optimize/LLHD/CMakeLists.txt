set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/circt-cfa-trace/Optimize/LLHD/Passes.td)
mlir_tablegen(circt-cfa-trace/Optimize/LLHD/Passes.h.inc
  -gen-pass-decls)
add_public_tablegen_target(CIRCTCFATraceLLHDPassIncGen)

add_circt_library(CIRCTCFATraceLLHDTransforms
  LLHDMemToFirMem.cpp

  DEPENDS
  CIRCTCFATraceLLHDPassIncGen

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  CIRCTLLHD
  CIRCTSeq
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRSupport
  )
