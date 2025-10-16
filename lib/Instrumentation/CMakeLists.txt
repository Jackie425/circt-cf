set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/circt-cfa-trace/Instrumentation/Transforms/Passes.td)
mlir_tablegen(circt-cfa-trace/Instrumentation/Transforms/InstrumentationPasses.h.inc
  -gen-pass-decls -name Instrumentation)
add_public_tablegen_target(CIRCTCFATraceInstrumentationPassIncGen)

add_circt_library(CIRCTCFATraceInstrumentation
  InsertHWProbe.cpp

  DEPENDS
  CIRCTCFATraceInstrumentationPassIncGen

  LINK_LIBS PUBLIC
  CIRCTHW
  MLIRIR
  MLIRPass
  MLIRSupport
)