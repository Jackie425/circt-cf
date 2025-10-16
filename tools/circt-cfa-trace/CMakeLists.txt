set(libs
  CIRCTCFATraceInstrumentation
  CIRCTCFATraceMooreTransforms
  CIRCTCFATraceLLHDTransforms
  CIRCTComb
  CIRCTDebug
  CIRCTExportVerilog
  CIRCTHW
  CIRCTHWToSV
  CIRCTImportVerilog
  CIRCTLLHD
  CIRCTLLHDTransforms
  CIRCTMoore
  CIRCTMooreToCore
  CIRCTMooreTransforms
  CIRCTSeq
  CIRCTSeqToSV
  CIRCTSeqTransforms
  CIRCTSV
  CIRCTSVTransforms
  CIRCTSupport
  CIRCTTransforms
  CIRCTVerif
  MLIRBytecodeReader
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIRIR
  MLIRParser
  MLIRSCFDialect
  MLIRSupport
  MLIRLLVMIRTransforms
  )

add_circt_tool(circt-cfa-trace circt-cfa-trace.cpp DEPENDS ${libs})
target_link_libraries(circt-cfa-trace PRIVATE ${libs})

llvm_update_compile_flags(circt-cfa-trace)
mlir_check_all_link_libraries(circt-cfa-trace)
