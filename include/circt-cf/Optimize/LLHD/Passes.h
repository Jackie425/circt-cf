//===- Passes.h - LLHD optimization pass entry points --------*- C++ -*-===//
//
// Part of the circt-cf project.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CF_OPTIMIZE_LLHD_PASSES_H
#define CIRCT_CF_OPTIMIZE_LLHD_PASSES_H

#include "mlir/Pass/Pass.h"

namespace circt::cfatrace::optimize::llhd {

#define GEN_PASS_DECL
#include "circt-cf/Optimize/LLHD/Passes.h.inc"

std::unique_ptr<mlir::Pass> createLLHDMemToFirMemPass();

void registerLLHDTransformPasses();

} // namespace circt::cfatrace::optimize::llhd

#endif // CIRCT_CF_OPTIMIZE_LLHD_PASSES_H
