#include "circt-cf/Instrumentation/MooreProcedureAnalysis.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace circt;

namespace circt::pcov {

namespace {

static bool isEdgeTriggeredClock(moore::DetectEventOp detect) {
  using moore::Edge;
  switch (detect.getEdge()) {
  case Edge::PosEdge:
  case Edge::NegEdge:
    return true;
  default:
    return false;
  }
}

static bool hasEdgeTriggeredClock(moore::WaitEventOp waitEvent) {
  bool hasClock = false;
  waitEvent.getBody().walk([&](moore::DetectEventOp detect) {
    hasClock |= isEdgeTriggeredClock(detect);
  });
  return hasClock;
}

} // namespace

MooreProcedureAnalysisResult analyzeMooreProcedure(moore::ProcedureOp proc,
                                                   bool emitDiagnostics) {
  MooreProcedureAnalysisResult result;
  Region &body = proc.getBody();
  if (body.empty())
    return result;

  MooreProcedureCFGAnalysis analysis;
  analysis.entryBlock = &body.front();

  auto getEntryWaitEvent = [](Block *block) -> moore::WaitEventOp {
    if (!block || block->empty())
      return nullptr;
    return dyn_cast<moore::WaitEventOp>(&block->front());
  };

  auto waitEvent = getEntryWaitEvent(analysis.entryBlock);
  if (!waitEvent || !hasEdgeTriggeredClock(waitEvent)) {
    if (!analysis.entryBlock->empty())
      if (auto br =
              dyn_cast<cf::BranchOp>(analysis.entryBlock->getTerminator())) {
        Block *next = br.getDest();
        auto nextWait = getEntryWaitEvent(next);
        if (nextWait && hasEdgeTriggeredClock(nextWait)) {
          analysis.entryBlock = next;
          waitEvent = nextWait;
        }
      }
  }

  if (!waitEvent || !hasEdgeTriggeredClock(waitEvent))
    return result;

  for (Block &block : body) {
    if (auto *term = block.getTerminator(); isa<moore::ReturnOp>(term)) {
      if (analysis.exitBlock && analysis.exitBlock != &block) {
        if (emitDiagnostics) {
          proc.emitError("procedure must have a unique return block for CFG "
                         "analysis");
        }
        result.fatalError = true;
        return result;
      }
      analysis.exitBlock = &block;
    }
  }

  if (!analysis.exitBlock) {
    if (emitDiagnostics) {
      proc.emitError("procedure lacks a return block, cannot analyze CFG");
    }
    result.fatalError = true;
    return result;
  }

  llvm::DenseMap<Block *, llvm::SmallVector<Block *, 2>> successors;
  llvm::DenseMap<Block *, unsigned> indegree;

  for (Block &block : body)
    indegree.try_emplace(&block, 0);

  auto recordEdge = [&](Block *from, Block *to) {
    successors[from].push_back(to);
    ++indegree[to];
  };

  for (Block &block : body) {
    Operation *terminator = block.getTerminator();

    if (auto cond = dyn_cast<cf::CondBranchOp>(terminator)) {
      recordEdge(&block, cond.getTrueDest());
      recordEdge(&block, cond.getFalseDest());
      continue;
    }

    if (auto br = dyn_cast<cf::BranchOp>(terminator)) {
      recordEdge(&block, br.getDest());
      continue;
    }

    if (isa<moore::ReturnOp>(terminator))
      continue;

    if (emitDiagnostics) {
      terminator->emitError("unsupported terminator for Moore CFG analysis");
    }
    result.fatalError = true;
    return result;
  }

  llvm::SmallVector<Block *, 16> worklist;
  for (Block &block : body)
    if (indegree.lookup(&block) == 0)
      worklist.push_back(&block);

  while (!worklist.empty()) {
    Block *current = worklist.pop_back_val();
    analysis.topoOrder.push_back(current);
    for (Block *succ : successors[current]) {
      unsigned &count = indegree[succ];
      if (--count == 0)
        worklist.push_back(succ);
    }
  }

  if (analysis.topoOrder.size() != body.getBlocks().size()) {
    if (emitDiagnostics)
      proc.emitError("procedure contains intra-cycle backedges; expected a DAG");
    result.fatalError = true;
    return result;
  }

  llvm::SmallVector<Block *, 16> reverseTopo(analysis.topoOrder.rbegin(),
                                             analysis.topoOrder.rend());
  analysis.numPaths.try_emplace(analysis.exitBlock, 1);

  for (Block *block : reverseTopo) {
    if (block == analysis.exitBlock)
      continue;

    auto succIt = successors.find(block);
    if (succIt == successors.end()) {
      if (emitDiagnostics) {
        proc.emitError(
            "block without successors cannot reach the unique exit block");
      }
      result.fatalError = true;
      return result;
    }

    uint64_t pathCount = 0;
    for (Block *succ : succIt->second) {
      auto numIt = analysis.numPaths.find(succ);
      assert(numIt != analysis.numPaths.end() &&
             "successor should already have a path count");
      pathCount += numIt->second;
    }
    analysis.numPaths.try_emplace(block, pathCount);
  }

  auto entryIt = analysis.numPaths.find(analysis.entryBlock);
  assert(entryIt != analysis.numPaths.end() &&
         "entry block must have a path count");
  uint64_t totalPaths = entryIt->second;
  if (totalPaths == 0) {
    if (emitDiagnostics)
      proc.emitError("entry block has zero outgoing paths to exit");
    result.fatalError = true;
    return result;
  }

  uint64_t base = 0;
  for (Block *block : analysis.topoOrder) {
    base = 0;
    auto succIt = successors.find(block);
    if (succIt == successors.end())
      continue;
    for (Block *succ : succIt->second) {
      analysis.edgeWeights.try_emplace(BlockEdge{block, succ}, base);
      base += analysis.numPaths.lookup(succ);
    }
  }

  analysis.pathIdWidth = totalPaths == 1 ? 1 : llvm::Log2_64_Ceil(totalPaths);

  result.analysis = std::move(analysis);
  return result;
}

} // namespace circt::pcov
