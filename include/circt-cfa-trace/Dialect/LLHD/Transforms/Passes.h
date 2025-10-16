//===- Passes.h - LLHD transform pass entry points ------------*- C++ -*-===//
//
// Part of the circt-cfa-trace project.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CFA_TRACE_DIALECT_LLHD_TRANSFORMS_PASSES_H
#define CIRCT_CFA_TRACE_DIALECT_LLHD_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace circt::cfatrace::llhd {

#define GEN_PASS_DECL
#include "circt-cfa-trace/Dialect/LLHD/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createLLHDMemToFirMemPass();

void registerLLHDTransformPasses();

} // namespace circt::cfatrace::llhd

#endif // CIRCT_CFA_TRACE_DIALECT_LLHD_TRANSFORMS_PASSES_H
