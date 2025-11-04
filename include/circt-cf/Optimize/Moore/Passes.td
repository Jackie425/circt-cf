//===- Passes.td - Moore optimization passes -------------*- tablegen -*-===//
//
// Part of the circt-cf project.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CF_OPTIMIZE_MOORE_PASSES
#define CIRCT_CF_OPTIMIZE_MOORE_PASSES

include "mlir/Pass/PassBase.td"

def NormalizeProcedures : Pass<"moore-normalize-procedures", "::circt::moore::SVModuleOp"> {
  let summary = "Convert always procedures with sensitivity lists to always_comb";
  let description = [{
    This pass identifies Moore `always` procedures that have explicit sensitivity
    lists (via wait_event) and converts them to `always_comb` when they represent
    pure combinational logic.
  }];
  let constructor =
      "circt::pcov::optimize::moore::createNormalizeProceduresPass()";
}

def MergeProcedures : Pass<"moore-merge-procedures", "::circt::moore::SVModuleOp"> {
  let summary = "Merge procedures with identical sensitivity lists";
  let description = [{
    Merge multiple `always` procedures that share the same sensitivity list
    and operate on disjoint slices of the same variables.
  }];
  let constructor =
      "circt::pcov::optimize::moore::createMergeProceduresPass()";
}

#endif // CIRCT_CF_OPTIMIZE_MOORE_PASSES
