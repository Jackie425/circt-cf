//===- Passes.h - LLHD optimization pass entry points --------*- C++ -*-===//
//
// Part of the circt-cfa-trace project.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CFA_TRACE_OPTIMIZE_LLHD_PASSES_H
#define CIRCT_CFA_TRACE_OPTIMIZE_LLHD_PASSES_H

#include "mlir/Pass/Pass.h"

namespace circt::cfatrace::optimize::llhd {

#define GEN_PASS_DECL
#include "circt-cfa-trace/Optimize/LLHD/Passes.h.inc"

std::unique_ptr<mlir::Pass> createLLHDMemToFirMemPass();

void registerLLHDTransformPasses();

} // namespace circt::cfatrace::optimize::llhd

#endif // CIRCT_CFA_TRACE_OPTIMIZE_LLHD_PASSES_H
