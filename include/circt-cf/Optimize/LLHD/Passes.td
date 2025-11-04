//===- Passes.td - LLHD optimization passes --------------*- tablegen -*-===//
//
// Part of the circt-cf project.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CF_OPTIMIZE_LLHD_PASSES
#define CIRCT_CF_OPTIMIZE_LLHD_PASSES

include "mlir/Pass/PassBase.td"

def LLHDMemToFirMemPass : Pass<"llhd-mem-to-firmem", "::circt::hw::HWModuleOp"> {
  let summary = "Convert LLHD multi-port memories to seq.firmem";
  let description = [{
    Detects LLHD memory patterns built from register arrays and rewrites them
    to `seq.firmem` with explicit read and write ports.
  }];
  let constructor =
      "circt::pcov::optimize::llhd::createLLHDMemToFirMemPass()";
}

#endif // CIRCT_CF_OPTIMIZE_LLHD_PASSES
