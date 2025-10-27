//===- Passes.h - Moore optimization pass entry points --------*- C++ -*-===//
//
// Part of the circt-cf project.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CF_OPTIMIZE_MOORE_PASSES_H
#define CIRCT_CF_OPTIMIZE_MOORE_PASSES_H

#include "mlir/Pass/Pass.h"

namespace circt::svcf::optimize::moore {

#define GEN_PASS_DECL
#include "circt-cf/Optimize/Moore/Passes.h.inc"

std::unique_ptr<mlir::Pass> createNormalizeProceduresPass();
std::unique_ptr<mlir::Pass> createMergeProceduresPass();

void registerMooreTransformPasses();

} // namespace circt::svcf::optimize::moore

#endif // CIRCT_CF_OPTIMIZE_MOORE_PASSES_H
