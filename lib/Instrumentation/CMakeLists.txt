set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/pcov/Instrumentation/Passes.td)
mlir_tablegen(pcov/Instrumentation/InstrumentationPasses.h.inc
  -gen-pass-decls -name Instrumentation)
add_public_tablegen_target(PCOVInstrumentationPassIncGen)

add_circt_library(PCOVInstrumentation
  InsertHWProbe.cpp
  MooreExportProcessCFG.cpp
  MooreInstrumentCoverage.cpp
  MooreInstrumentPathBitmap.cpp
  MooreInstrumentCoverageSum.cpp
  MooreSummarizeCoverage.cpp
  MooreProcedureAnalysis.cpp

  DEPENDS
  PCOVInstrumentationPassIncGen

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTMoore
  MLIRIR
  MLIRPass
  MLIRSupport
)
