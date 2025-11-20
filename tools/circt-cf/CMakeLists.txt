set(libs
  CIRCTCFInstrumentation
  CIRCTCFMooreTransforms
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

add_circt_tool(circt-cf circt-cf.cpp DEPENDS ${libs})
target_link_libraries(circt-cf PRIVATE ${libs})

llvm_update_compile_flags(circt-cf)
mlir_check_all_link_libraries(circt-cf)
