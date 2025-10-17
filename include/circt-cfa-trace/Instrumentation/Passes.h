//===- Passes.h - Instrumentation pass entry points -------------*- C++ -*-===//
//
// Part of the circt-cfa-trace project.
//
//===----------------------------------------------------------------------===//
//
// This header declares the instrumentation pass entry points.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CFA_TRACE_INSTRUMENTATION_PASSES_H
#define CIRCT_CFA_TRACE_INSTRUMENTATION_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace hw {
class HWDialect;
} // namespace hw

namespace cfatrace {

#define GEN_PASS_DECL
#include "circt-cfa-trace/Instrumentation/InstrumentationPasses.h.inc"

std::unique_ptr<mlir::Pass> createInsertHWProbePass();
void registerInsertHWProbePasses();

} // namespace cfatrace
} // namespace circt

#endif // CIRCT_CFA_TRACE_INSTRUMENTATION_PASSES_H
