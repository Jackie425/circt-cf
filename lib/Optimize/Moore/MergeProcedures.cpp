//===- MergeProcedures.cpp - Merge Moore Procedures -----------*- C++ -*-===//
//
// Part of the circt-cfa-trace project.
//
//===----------------------------------------------------------------------===//

#include "circt-cfa-trace/Optimize/Moore/Passes.h"

#include "circt/Dialect/Moore/MooreOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include <optional>
#include <string>

#define DEBUG_TYPE "moore-merge-procedures"

using namespace mlir;
using namespace circt;
using namespace ::circt::moore;

namespace circt::cfatrace::optimize::moore {
#define GEN_PASS_DEF_MERGEPROCEDURES
#include "circt-cfa-trace/Optimize/Moore/Passes.h.inc"
} // namespace circt::cfatrace::optimize::moore

namespace circt::cfatrace::optimize::moore {
namespace {

/// Represents a write target (variable or bit range)
struct WriteTarget {
  Value variable;
  std::optional<int> lowBit;
  std::optional<int> highBit;

  WriteTarget(Value var) : variable(var) {}
  WriteTarget(Value var, int low, int high)
      : variable(var), lowBit(low), highBit(high) {}

  bool overlaps(const WriteTarget &other) const {
    if (variable != other.variable)
      return false;
    if (!lowBit.has_value() || !other.lowBit.has_value())
      return true;
    return std::max(lowBit.value(), other.lowBit.value()) <=
           std::min(highBit.value(), other.highBit.value());
  }

  std::string toString() const {
    std::string result;
    llvm::raw_string_ostream os(result);

    if (auto varOp = variable.getDefiningOp<VariableOp>()) {
      os << (varOp.getName() ? varOp.getName().value() : "var");
    } else if (auto netOp = variable.getDefiningOp<NetOp>()) {
      os << (netOp.getName() ? netOp.getName().value() : "net");
    } else {
      os << "signal";
    }

    if (lowBit.has_value()) {
      if (lowBit.value() == highBit.value())
        os << "[" << lowBit.value() << "]";
      else
        os << "[" << highBit.value() << ":" << lowBit.value() << "]";
    }
    return os.str();
  }
};

struct SensitivityListHash {
  llvm::hash_code hash;
  SmallVector<std::pair<Edge, Value>, 4> events;

  bool operator==(const SensitivityListHash &other) const {
    if (events.size() != other.events.size())
      return false;
    for (size_t i = 0; i < events.size(); ++i) {
      if (events[i].first != other.events[i].first ||
          events[i].second != other.events[i].second)
        return false;
    }
    return true;
  }

  std::string toString() const {
    std::string result = "@(";
    for (size_t i = 0; i < events.size(); ++i) {
      if (i > 0)
        result += ", ";
      switch (events[i].first) {
      case Edge::PosEdge:
        result += "posedge ";
        break;
      case Edge::NegEdge:
        result += "negedge ";
        break;
      case Edge::BothEdges:
        result += "edge ";
        break;
      case Edge::AnyChange:
        break;
      }
      Value val = events[i].second;
      if (auto varOp = val.getDefiningOp<VariableOp>()) {
        result += varOp.getName() ? varOp.getName().value().str() : "var";
      } else if (auto netOp = val.getDefiningOp<NetOp>()) {
        result += netOp.getName() ? netOp.getName().value().str() : "net";
      } else {
        result += "signal";
      }
    }
    result += ")";
    return result;
  }
};

struct ProcedureAnalyzer {
  ProcedureOp proc;
  WaitEventOp waitEvent = nullptr;
  SensitivityListHash sensitivityHash;
  SmallVector<WriteTarget, 4> writeTargets;
  bool isBlocking = false;
  bool isValid = false;

  explicit ProcedureAnalyzer(ProcedureOp p) : proc(p) { analyze(); }

  void analyze() {
    if (proc.getKind() != ProcedureKind::Always)
      return;

    Block &body = proc.getBody().front();
    if (body.empty())
      return;

    waitEvent = dyn_cast<WaitEventOp>(&body.front());
    if (!waitEvent)
      return;

    if (!extractSensitivityList())
      return;

    bool hasControlFlow = false;
    proc.walk([&](Operation *op) {
      if (op->getDialect() && op->getDialect()->getNamespace() == "cf") {
        hasControlFlow = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (hasControlFlow)
      return;

    bool hasBlocking = false, hasNonBlocking = false;
    proc.walk([&](BlockingAssignOp op) {
      hasBlocking = true;
      writeTargets.push_back(extractWriteTarget(op.getDst()));
    });
    proc.walk([&](NonBlockingAssignOp op) {
      hasNonBlocking = true;
      writeTargets.push_back(extractWriteTarget(op.getDst()));
    });

    if (hasBlocking && hasNonBlocking)
      return;

    if (writeTargets.empty())
      return;

    isBlocking = hasBlocking;
    isValid = true;
  }

  bool extractSensitivityList() {
    auto &body = waitEvent.getBody().front();
    sensitivityHash.events.clear();

    for (Operation &op : body) {
      if (auto detect = dyn_cast<DetectEventOp>(op)) {
        Value signal = detect.getInput();
        if (auto readOp = signal.getDefiningOp<ReadOp>())
          signal = readOp.getInput();
        sensitivityHash.events.push_back({detect.getEdge(), signal});
      }
    }

    llvm::sort(sensitivityHash.events, [](const auto &a, const auto &b) {
      if (a.first != b.first)
        return a.first < b.first;
      return a.second.getAsOpaquePointer() <
             b.second.getAsOpaquePointer();
    });

    sensitivityHash.hash = llvm::hash_combine_range(
        sensitivityHash.events.begin(), sensitivityHash.events.end());

    return !sensitivityHash.events.empty();
  }

  WriteTarget extractWriteTarget(Value dst) {
    if (auto extractOp = dst.getDefiningOp<ExtractRefOp>()) {
      Value baseVar = extractOp.getInput();
      int lowBit = extractOp.getLowBit();
      int width = 1;
      if (auto intType = dyn_cast<IntType>(
              cast<RefType>(extractOp.getType()).getNestedType()))
        width = intType.getWidth();
      return WriteTarget(baseVar, lowBit, lowBit + width - 1);
    }
    return WriteTarget(dst);
  }

  bool canMergeWith(const ProcedureAnalyzer &other) const {
    if (!isValid || !other.isValid)
      return false;
    if (!(sensitivityHash == other.sensitivityHash))
      return false;
    if (isBlocking != other.isBlocking)
      return false;

    for (const auto &thisTarget : writeTargets)
      for (const auto &otherTarget : other.writeTargets)
        if (thisTarget.overlaps(otherTarget))
          return false;

    SmallPtrSet<Value, 4> thisVars, otherVars;
    for (const auto &t : writeTargets)
      thisVars.insert(t.variable);
    for (const auto &t : other.writeTargets)
      otherVars.insert(t.variable);

    for (Value var : thisVars)
      if (!otherVars.contains(var))
        return false;

    if (thisVars.size() == 1) {
      for (const auto &t : writeTargets)
        if (!t.lowBit.has_value())
          return false;
      for (const auto &t : other.writeTargets)
        if (!t.lowBit.has_value())
          return false;
    }

    return true;
  }
};

class MergeProceduresPass
    : public impl::MergeProceduresBase<MergeProceduresPass> {
public:
  void runOnOperation() override;

private:
  void mergeProcedures(SmallVectorImpl<ProcedureOp> &procedures,
                       SmallVectorImpl<ProcedureAnalyzer *> &analyzers);
};

void MergeProceduresPass::mergeProcedures(
    SmallVectorImpl<ProcedureOp> &procedures,
    SmallVectorImpl<ProcedureAnalyzer *> &analyzers) {
  if (procedures.size() < 2)
    return;

  Operation *parentOp = procedures[0]->getParentOp();
  std::string moduleName = "<unknown>";
  while (parentOp) {
    if (auto modOp = dyn_cast<SVModuleOp>(parentOp)) {
      moduleName = modOp.getSymName().str();
      break;
    }
    parentOp = parentOp->getParentOp();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "\n╔══════════════════════════════════════════════════════════\n";
    llvm::dbgs() << "║ Merging " << procedures.size() << " procedures in @"
                 << moduleName << "\n";
    llvm::dbgs() << "╠══════════════════════════════════════════════════════════\n";
    llvm::dbgs() << "║ Sensitivity: "
                 << analyzers[0]->sensitivityHash.toString() << "\n";
    llvm::dbgs() << "║ Assignment type: "
                 << (analyzers[0]->isBlocking ? "blocking (=)"
                                              : "non-blocking (<=)")
                 << "\n";
    llvm::dbgs() << "╠══════════════════════════════════════════════════════════\n";

    for (size_t i = 0; i < analyzers.size(); ++i) {
      llvm::dbgs() << "║ Procedure " << (i + 1) << " writes: ";
      for (size_t j = 0; j < analyzers[i]->writeTargets.size(); ++j) {
        if (j > 0)
          llvm::dbgs() << ", ";
        llvm::dbgs() << analyzers[i]->writeTargets[j].toString();
      }
      llvm::dbgs() << "\n";
    }
    llvm::dbgs() << "╠══════════════════════════════════════════════════════════\n";
    llvm::dbgs() << "║ Merging into first procedure, removing "
                 << (procedures.size() - 1) << " duplicate(s)\n";
    llvm::dbgs() << "╚══════════════════════════════════════════════════════════\n";
  });

  ProcedureOp targetProc = procedures[0];
  Block &targetBody = targetProc.getBody().front();
  Operation *targetReturn = targetBody.getTerminator();

  OpBuilder builder(targetProc.getContext());
  builder.setInsertionPoint(targetReturn);

  for (size_t i = 1; i < procedures.size(); ++i) {
    ProcedureOp srcProc = procedures[i];
    Block &srcBody = srcProc.getBody().front();

    bool afterWaitEvent = false;
    IRMapping mapping;
    for (Operation &op : srcBody) {
      if (isa<WaitEventOp>(&op)) {
        afterWaitEvent = true;
        continue;
      }
      if (afterWaitEvent && !isa<::circt::moore::ReturnOp>(&op))
        builder.clone(op, mapping);
    }

    srcProc.erase();
  }
}

void MergeProceduresPass::runOnOperation() {
  auto module = getOperation();

  SmallVector<ProcedureAnalyzer, 0> allAnalyzers;
  SmallVector<ProcedureOp> allProcs;

  module.walk([&](ProcedureOp proc) {
    ProcedureAnalyzer analyzer(proc);
    if (analyzer.isValid) {
      allAnalyzers.push_back(std::move(analyzer));
      allProcs.push_back(proc);
    }
  });

  if (allAnalyzers.size() < 2)
    return;

  DenseMap<llvm::hash_code, SmallVector<size_t>> groups;
  for (size_t i = 0; i < allAnalyzers.size(); ++i)
    groups[allAnalyzers[i].sensitivityHash.hash].push_back(i);

  int totalMerged = 0;
  for (auto &[hash, indices] : groups) {
    if (indices.size() < 2)
      continue;

    SmallPtrSet<size_t, 8> processed;
    for (size_t idx : indices) {
      if (processed.contains(idx))
        continue;

      SmallVector<size_t> mergeSet = {idx};
      processed.insert(idx);

      for (size_t otherIdx : indices) {
        if (processed.contains(otherIdx))
          continue;

        bool compatible = true;
        for (size_t existingIdx : mergeSet) {
          if (!allAnalyzers[existingIdx].canMergeWith(
                  allAnalyzers[otherIdx])) {
            compatible = false;
            break;
          }
        }

        if (compatible) {
          mergeSet.push_back(otherIdx);
          processed.insert(otherIdx);
        }
      }

      if (mergeSet.size() >= 2) {
        SmallVector<ProcedureOp> procsToMerge;
        SmallVector<ProcedureAnalyzer *> analyzersToMerge;
        for (size_t i : mergeSet) {
          procsToMerge.push_back(allProcs[i]);
          analyzersToMerge.push_back(&allAnalyzers[i]);
        }
        mergeProcedures(procsToMerge, analyzersToMerge);
        totalMerged += mergeSet.size() - 1;
      }
    }
  }

  LLVM_DEBUG({
    if (totalMerged > 0) {
      llvm::dbgs() << "\n╔══════════════════════════════════════════════════════════\n";
      llvm::dbgs() << "║ SUMMARY: " << totalMerged << " procedure(s) merged\n";
      llvm::dbgs() << "╚══════════════════════════════════════════════════════════\n\n";
    }
  });
}

} // namespace

std::unique_ptr<mlir::Pass> createMergeProceduresPass() {
  return std::make_unique<MergeProceduresPass>();
}

void registerMooreTransformPasses() {
  static bool initOnce = []() {
    mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
      return createMergeProceduresPass();
    });
    mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
      return createNormalizeProceduresPass();
    });
    return true;
  }();
  (void)initOnce;
}

} // namespace circt::cfatrace::optimize::moore
