set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/circt-cf/Instrumentation/Passes.td)
mlir_tablegen(circt-cf/Instrumentation/InstrumentationPasses.h.inc
  -gen-pass-decls -name Instrumentation)
add_public_tablegen_target(CIRCTCFInstrumentationPassIncGen)

add_circt_library(CIRCTCFInstrumentation
  InsertHWProbe.cpp
  MooreExportProcessCFG.cpp
  MooreInstrumentCoverage.cpp

  DEPENDS
  CIRCTCFInstrumentationPassIncGen

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTMoore
  MLIRIR
  MLIRPass
  MLIRSupport
)
