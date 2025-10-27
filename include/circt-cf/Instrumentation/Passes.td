//===- InstrumentationPasses.td - Pass definitions --------------*- tablegen -*-===//
//
// Part of the circt-cf project.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CF_INSTRUMENTATION_PASSES
#define CIRCT_CF_INSTRUMENTATION_PASSES

include "mlir/Pass/PassBase.td"

def InsertHWProbe : Pass<"insert-hw-probes", "::mlir::ModuleOp"> {
  let summary = "Insert CFA trace attributes on circt.hw modules";
  let description = [{
    Adds a marker attribute named `hw.cf` to every `hw.module`
    operation in the input module. External tools can use this marker to
    detect which modules should be instrumented further.
  }];
  let constructor = "circt::svcf::createInsertHWProbePass()";
  let dependentDialects = ["circt::hw::HWDialect"];
}

#endif // CIRCT_CF_INSTRUMENTATION_PASSES
