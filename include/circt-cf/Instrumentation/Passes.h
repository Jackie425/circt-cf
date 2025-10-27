//===- Passes.h - Instrumentation pass entry points -------------*- C++ -*-===//
//
// Part of the circt-cf project.
//
//===----------------------------------------------------------------------===//
//
// This header declares the instrumentation pass entry points.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CF_INSTRUMENTATION_PASSES_H
#define CIRCT_CF_INSTRUMENTATION_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace hw {
class HWDialect;
} // namespace hw

namespace cfatrace {

#define GEN_PASS_DECL
#include "circt-cf/Instrumentation/InstrumentationPasses.h.inc"

std::unique_ptr<mlir::Pass> createInsertHWProbePass();
void registerInsertHWProbePasses();

} // namespace cfatrace
} // namespace circt

#endif // CIRCT_CF_INSTRUMENTATION_PASSES_H
