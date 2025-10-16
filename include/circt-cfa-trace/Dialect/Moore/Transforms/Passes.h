//===- Passes.h - Moore transform pass entry points -----------*- C++ -*-===//
//
// Part of the circt-cfa-trace project.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CFA_TRACE_DIALECT_MOORE_TRANSFORMS_PASSES_H
#define CIRCT_CFA_TRACE_DIALECT_MOORE_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace circt::cfatrace::moore {

#define GEN_PASS_DECL
#include "circt-cfa-trace/Dialect/Moore/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createNormalizeProceduresPass();
std::unique_ptr<mlir::Pass> createMergeProceduresPass();

void registerMooreTransformPasses();

} // namespace circt::cfatrace::moore

#endif // CIRCT_CFA_TRACE_DIALECT_MOORE_TRANSFORMS_PASSES_H
