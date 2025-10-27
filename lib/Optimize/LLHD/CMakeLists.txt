set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/circt-cf/Optimize/LLHD/Passes.td)
mlir_tablegen(circt-cf/Optimize/LLHD/Passes.h.inc
  -gen-pass-decls)
add_public_tablegen_target(CIRCTCFLLHDPassIncGen)

add_circt_library(CIRCTCFLLHDTransforms
  LLHDMemToFirMem.cpp

  DEPENDS
  CIRCTCFLLHDPassIncGen

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
