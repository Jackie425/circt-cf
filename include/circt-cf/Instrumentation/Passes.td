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

def MooreInstrumentCoverage
    : Pass<"moore-instrument-coverage", "::circt::moore::SVModuleOp"> {
  let summary = "Insert coverage bookkeeping into Moore procedures";
  let description = [{
    Identify leaf basic blocks within Moore procedures and insert coverage
    bookkeeping that records the current leaf identifier, previous leaf
    identifier, and a combined edge signature suitable for later bitmap based
    aggregation. This pass also introduces per-procedure coverage variables
    inside the surrounding Moore module.
  }];
  let constructor = "circt::svcf::createMooreInstrumentCoveragePass()";
  let dependentDialects = [
    "circt::moore::MooreDialect",
    "mlir::cf::ControlFlowDialect"
  ];
}

def MooreExportProcessCFG
    : Pass<"moore-export-process-cfg", "::circt::moore::SVModuleOp"> {
  let summary = "Emit CFGs for each Moore procedure as Graphviz .dot files";
  let description = [{
    Traverse every procedure inside a Moore module and materialize the control
    flow graph as a Graphviz .dot document. Each basic block in the procedure
    is represented as a single node and edges connect predecessor and successor
    blocks in the Moore/CF control-flow graph. This pass is intended for
    visualization and debugging purposes.
  }];
  let constructor = "circt::svcf::createMooreExportProcessCFGPass()";
  let dependentDialects = [
    "circt::moore::MooreDialect",
    "mlir::cf::ControlFlowDialect"
  ];
  let options = [
    Option<"outputDir", "output-dir", "std::string", "\".\"",
           "Directory used to store the generated .dot files.">
  ];
}

#endif // CIRCT_CF_INSTRUMENTATION_PASSES
