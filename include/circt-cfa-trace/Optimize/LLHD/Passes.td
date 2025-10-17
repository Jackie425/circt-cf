//===- Passes.td - LLHD optimization passes --------------*- tablegen -*-===//
//
// Part of the circt-cfa-trace project.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CFA_TRACE_OPTIMIZE_LLHD_PASSES
#define CIRCT_CFA_TRACE_OPTIMIZE_LLHD_PASSES

include "mlir/Pass/PassBase.td"

def LLHDMemToFirMemPass : Pass<"llhd-mem-to-firmem", "::circt::hw::HWModuleOp"> {
  let summary = "Convert LLHD multi-port memories to seq.firmem";
  let description = [{
    Detects LLHD memory patterns built from register arrays and rewrites them
    to `seq.firmem` with explicit read and write ports.
  }];
  let constructor =
      "circt::cfatrace::optimize::llhd::createLLHDMemToFirMemPass()";
}

#endif // CIRCT_CFA_TRACE_OPTIMIZE_LLHD_PASSES
