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
  let constructor = "circt::pcov::createInsertHWProbePass()";
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
  let constructor = "circt::pcov::createMooreInstrumentCoveragePass()";
  let dependentDialects = [
    "circt::moore::MooreDialect",
    "mlir::cf::ControlFlowDialect"
  ];
}

def MooreSummarizeCoverage
    : Pass<"moore-summarize-coverage", "::mlir::ModuleOp"> {
  let summary = "Summarize Moore coverage instrumentation across the design";
  let description = [{
    Collect coverage attributes produced by the Moore instrumentation pass and
    print a design-wide report that enumerates instrumented procedures and
    aggregates their path identifier widths.
  }];
  let constructor = "circt::pcov::createMooreSummarizeCoveragePass()";
  let dependentDialects = [
    "circt::moore::MooreDialect"
  ];
}

def MooreExportProcessCFG
    : Pass<"moore-export-process-cfg", "::circt::moore::SVModuleOp"> {
  let summary = "Emit procedure CFGs as Graphviz .dot documents";
  let description = [{
    Traverse every Moore procedure with a synthesizable, edge-triggered wait
    event and reuse the shared CFG analysis used by coverage instrumentation to
    render a Graphviz .dot document. Each node represents a basic
    block annotated with its path count and edges are labelled with the Ball-
    Larus weight assigned during instrumentation. This pass is intended for
    visualization and debugging. When the `output-dir` option is provided, the
    graphs are written to that directory; otherwise they are printed to
    stdout.
  }];
  let constructor = "circt::pcov::createMooreExportProcessCFGPass()";
  let dependentDialects = [
    "circt::moore::MooreDialect",
    "mlir::cf::ControlFlowDialect"
  ];
  let options = [
    Option<"outputDir", "output-dir", "std::string", "\"\"",
           "Directory used to store the generated .dot files. If empty no "
           "files are written.">
  ];
}

#endif // CIRCT_CF_INSTRUMENTATION_PASSES
