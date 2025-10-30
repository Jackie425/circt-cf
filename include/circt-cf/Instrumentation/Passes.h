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

namespace svcf {

#define GEN_PASS_DECL
#include "circt-cf/Instrumentation/InstrumentationPasses.h.inc"

std::unique_ptr<mlir::Pass> createInsertHWProbePass();
void registerInsertHWProbePasses();

std::unique_ptr<mlir::Pass> createMooreInstrumentCoveragePass();
void registerMooreInstrumentCoveragePass();

std::unique_ptr<mlir::Pass> createMooreExportProcessCFGPass();
void registerMooreExportProcessCFGPass();

} // namespace svcf
} // namespace circt

#endif // CIRCT_CF_INSTRUMENTATION_PASSES_H
