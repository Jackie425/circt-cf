//===- circt-cf-opt.cpp - The circt-cf-opt driver ----------===//
//
// CFA Trace optimizer driver for testing passes
//
//===----------------------------------------------------------------------===//

#include "circt-cf/Optimize/Moore/Passes.h"
#include "circt-cf/Instrumentation/Passes.h"
#include "circt/InitAllDialects.h"
#include "circt/Support/Version.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/PrettyStackTrace.h"

int main(int argc, char **argv) {
  // Set the bug report message to indicate users should file issues
  llvm::setBugReportMsg(circt::circtBugReportMsg);

  mlir::DialectRegistry registry;

  // Register MLIR dialects
  registry.insert<mlir::affine::AffineDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();

  // Register ALL CIRCT dialects (so your passes can use any dialect)
  // This is flexible - if you add a pass that uses a new dialect, it will work
  circt::registerAllDialects(registry);

  // DON'T call circt::registerAllPasses() here!
  // Only register the passes you actually need for testing

  // Register ONLY your custom passes
  circt::pcov::registerInsertHWProbePasses();
  circt::pcov::registerMooreInstrumentCoveragePass();
  circt::pcov::registerMooreInstrumentPathBitmapPass();
  circt::pcov::registerMooreInstrumentCoverageSumPass();
  circt::pcov::registerMooreSummarizeCoveragePass();
  circt::pcov::registerMooreExportProcessCFGPass();
  // If you need more custom passes in the future, add them here:
  // circt::pcov::registerOtherCustomPasses();

  // Register inliner extensions (useful for optimization)
  mlir::func::registerInlinerExtension(registry);
  mlir::LLVM::registerInlinerInterface(registry);

  // Register commonly used MLIR passes (optional, add as needed)
  mlir::registerCSEPass();
  mlir::registerCanonicalizerPass();
  mlir::registerSymbolDCEPass();

  // Add version printer
  llvm::cl::AddExtraVersionPrinter([](llvm::raw_ostream &os) {
    os << circt::getCirctVersion() << '\n';
  });

  return mlir::failed(mlir::MlirOptMain(
      argc, argv, "CIRCT CF modular optimizer driver", registry));
}
