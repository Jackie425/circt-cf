//===- Passes.h - Moore optimization pass entry points --------*- C++ -*-===//
//
// Part of the circt-cfa-trace project.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CFA_TRACE_OPTIMIZE_MOORE_PASSES_H
#define CIRCT_CFA_TRACE_OPTIMIZE_MOORE_PASSES_H

#include "mlir/Pass/Pass.h"

namespace circt::cfatrace::optimize::moore {

#define GEN_PASS_DECL
#include "circt-cfa-trace/Optimize/Moore/Passes.h.inc"

std::unique_ptr<mlir::Pass> createNormalizeProceduresPass();
std::unique_ptr<mlir::Pass> createMergeProceduresPass();

void registerMooreTransformPasses();

} // namespace circt::cfatrace::optimize::moore

#endif // CIRCT_CFA_TRACE_OPTIMIZE_MOORE_PASSES_H
