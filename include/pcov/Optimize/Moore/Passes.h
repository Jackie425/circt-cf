//===- Passes.h - Moore optimization pass entry points --------*- C++ -*-===//
//
// Part of the pcov project.
//
//===----------------------------------------------------------------------===//

#ifndef PCOV_OPTIMIZE_MOORE_PASSES_H
#define PCOV_OPTIMIZE_MOORE_PASSES_H

#include "mlir/Pass/Pass.h"

namespace circt::pcov::optimize::moore {

#define GEN_PASS_DECL
#include "pcov/Optimize/Moore/Passes.h.inc"

std::unique_ptr<mlir::Pass> createNormalizeProceduresPass();
std::unique_ptr<mlir::Pass> createMergeProceduresPass();
std::unique_ptr<mlir::Pass> createFoldConstantBranchesPass();
std::unique_ptr<mlir::Pass> createFoldStaticRegistersPass();

} // namespace circt::pcov::optimize::moore

#endif // PCOV_OPTIMIZE_MOORE_PASSES_H
