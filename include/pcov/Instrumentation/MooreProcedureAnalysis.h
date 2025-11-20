#ifndef PCOV_INSTRUMENTATION_MOOREPROCEDUREANALYSIS_H
#define PCOV_INSTRUMENTATION_MOOREPROCEDUREANALYSIS_H

#include "circt/Dialect/Moore/MooreOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>
#include <utility>

namespace circt::pcov {

using BlockEdge = std::pair<mlir::Block *, mlir::Block *>;

struct MooreProcedureCFGAnalysis {
  mlir::Block *entryBlock = nullptr;
  mlir::Block *exitBlock = nullptr;
  unsigned pathIdWidth = 0;
  llvm::DenseMap<mlir::Block *, uint64_t> numPaths;
  llvm::DenseMap<BlockEdge, uint64_t> edgeWeights;
  llvm::SmallVector<mlir::Block *, 16> topoOrder;

  bool isInstrumentable() const { return entryBlock && exitBlock; }
};

struct MooreProcedureAnalysisResult {
  std::optional<MooreProcedureCFGAnalysis> analysis;
  bool fatalError = false;
};

MooreProcedureAnalysisResult
analyzeMooreProcedure(moore::ProcedureOp proc, bool emitDiagnostics = true);

} // namespace circt::pcov

#endif // PCOV_INSTRUMENTATION_MOOREPROCEDUREANALYSIS_H
