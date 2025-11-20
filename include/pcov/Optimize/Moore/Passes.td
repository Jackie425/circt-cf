//===- Passes.td - Moore optimization passes -------------*- tablegen -*-===//
//
// Part of the pcov project.
//
//===----------------------------------------------------------------------===//

#ifndef PCOV_OPTIMIZE_MOORE_PASSES
#define PCOV_OPTIMIZE_MOORE_PASSES

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

def FoldConstantBranches
    : Pass<"moore-fold-constant-branches", "::circt::moore::SVModuleOp"> {
  let summary = "Fold Moore procedures branches with constant conditions";
  let description = [{
    Identify Moore procedural control-flow branches whose conditions are known
    constants and rewrite them to unconditional branches. This removes
    compile-time-only paths before instrumentation runs.
  }];
  let constructor =
      "circt::pcov::optimize::moore::createFoldConstantBranchesPass()";
}

def FoldStaticRegisters : Pass<"moore-fold-static-registers", "::circt::moore::SVModuleOp"> {
  let summary = "Fold sequential Moore variables that are statically constant";
  let description = [{
    Identify Moore variables which are only ever assigned the same constant
    value (including along reset paths) and replace them with direct constant
    drivers. This normalizes processes that otherwise keep redundant state and
    interfere with later lowering passes.
  }];
  let constructor =
      "circt::pcov::optimize::moore::createFoldStaticRegistersPass()";
}

#endif // PCOV_OPTIMIZE_MOORE_PASSES
