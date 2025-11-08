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

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace hw {
class HWDialect;
} // namespace hw

namespace pcov {

#define GEN_PASS_DECL
#include "circt-cf/Instrumentation/InstrumentationPasses.h.inc"

std::unique_ptr<mlir::Pass> createInsertHWProbePass();
void registerInsertHWProbePasses();

std::unique_ptr<mlir::Pass> createMooreInstrumentCoveragePass();
void registerMooreInstrumentCoveragePass();

std::unique_ptr<mlir::Pass> createMooreInstrumentPathBitmapPass();
void registerMooreInstrumentPathBitmapPass();

std::unique_ptr<mlir::Pass> createMooreSummarizeCoveragePass();
void registerMooreSummarizeCoveragePass();

std::unique_ptr<mlir::Pass> createMooreExportProcessCFGPass();
std::unique_ptr<mlir::Pass>
createMooreExportProcessCFGPass(llvm::StringRef outputDir);
void registerMooreExportProcessCFGPass();

} // namespace pcov
} // namespace circt

#endif // CIRCT_CF_INSTRUMENTATION_PASSES_H
