# Get all CIRCT dialect and conversion libraries automatically
# This way you don't need to manually add each dialect library
get_property(dialect_libs GLOBAL PROPERTY CIRCT_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY CIRCT_CONVERSION_LIBS)

set(LLVM_LINK_COMPONENTS
  Support
)

add_circt_tool(circt-cfa-trace-opt
  circt-cfa-trace-opt.cpp
)

llvm_update_compile_flags(circt-cfa-trace-opt)

target_link_libraries(circt-cfa-trace-opt
  PRIVATE
  # Link all CIRCT dialects and conversions automatically
  ${dialect_libs}
  ${conversion_libs}
  
  # CIRCT core libraries
  CIRCTSupport
  CIRCTTransforms
  
  # Your custom libraries
  CIRCTCFATraceLLHDTransforms
  CIRCTCFATraceMooreTransforms
  CIRCTCFATraceInstrumentation
  
  # MLIR libraries
  MLIRAffineDialect
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRLLVMDialect
  MLIROptLib
  MLIRPass
  MLIRTransforms
  MLIRSupport
  MLIRParser
  MLIRIR
  MLIRFuncInlinerExtension
  MLIRLLVMIRTransforms
)