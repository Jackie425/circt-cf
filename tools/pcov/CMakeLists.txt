set(libs
  PCOVInstrumentation
  PCOVMooreTransforms
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

add_circt_tool(pcov pcov.cpp DEPENDS ${libs})
target_link_libraries(pcov PRIVATE ${libs})

llvm_update_compile_flags(pcov)
mlir_check_all_link_libraries(pcov)
