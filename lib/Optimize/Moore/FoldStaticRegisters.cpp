//===- FoldStaticRegisters.cpp - Collapse constant Moore variables -------===//
//
// Part of the circt-cf project.
//
//===----------------------------------------------------------------------===//

#include "circt-cf/Optimize/Moore/Passes.h"

#include "circt/Dialect/Moore/MooreAttributes.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Moore/MooreTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "moore-fold-static-registers"

using namespace mlir;
using namespace circt;
using namespace ::circt::moore;

namespace circt::pcov::optimize::moore {
#define GEN_PASS_DEF_FOLDSTATICREGISTERS
#include "circt-cf/Optimize/Moore/Passes.h.inc"
} // namespace circt::pcov::optimize::moore

namespace circt::pcov::optimize::moore {
namespace {

/// Helper that recursively tries to determine whether a value resolves to a
/// compile-time constant. The evaluator is intentionally lightweight and only
/// handles the handful of operations we care about for the targeted constant
/// registers.
class ConstantEvaluator {
public:
  std::optional<FVIntegerAttr> evaluate(Value value) {
    if (auto it = cache.find(value); it != cache.end())
      return it->second;

    // Block arguments are never considered constant here.
    if (!value.getDefiningOp()) {
      cache[value] = std::nullopt;
      return std::nullopt;
    }

    auto *op = value.getDefiningOp();

    if (auto constOp = dyn_cast<ConstantOp>(op)) {
      auto attr = constOp.getValueAttr();
      cache[value] = attr;
      return attr;
    }

    if (auto eqOp = dyn_cast<EqOp>(op)) {
      auto lhsAttr = evaluate(eqOp.getLhs());
      auto rhsAttr = evaluate(eqOp.getRhs());
      if (!lhsAttr || !rhsAttr) {
        cache[value] = std::nullopt;
        return std::nullopt;
      }

      FVIntegerAttr resultAttr = FVIntegerAttr::get(
          op->getContext(),
          FVInt(cast<IntType>(eqOp.getType()).getWidth(),
                lhsAttr->getValue() == rhsAttr->getValue() ? 1 : 0));
      cache[value] = resultAttr;
      return resultAttr;
    }

    if (auto condOp = dyn_cast<ConditionalOp>(op)) {
      auto condAttr = evaluate(condOp.getCondition());
      if (!condAttr) {
        cache[value] = std::nullopt;
        return std::nullopt;
      }

      bool isTrue = !condAttr->getValue().isZero();
      Region &chosenRegion =
          isTrue ? condOp.getTrueRegion() : condOp.getFalseRegion();
      auto *terminator = chosenRegion.front().getTerminator();
      auto yieldOp = cast<YieldOp>(terminator);

      if (condOp->getNumResults() != 1 || yieldOp->getNumOperands() != 1) {
        cache[value] = std::nullopt;
        return std::nullopt;
      }

      auto yielded = evaluate(yieldOp.getOperand());
      cache[value] = yielded;
      return yielded;
    }

    cache[value] = std::nullopt;
    return std::nullopt;
  }

private:
  DenseMap<Value, std::optional<FVIntegerAttr>> cache;
};

struct FoldStaticRegistersPass
    : public impl::FoldStaticRegistersBase<FoldStaticRegistersPass> {
  void runOnOperation() override;
};

void FoldStaticRegistersPass::runOnOperation() {
  auto module = getOperation();
  StringRef moduleName =
      module.getSymNameAttr() ? module.getSymName() : "<anonymous>";
  std::string moduleStr = moduleName.str();

  DenseMap<Value, FVIntegerAttr> candidateValues;
  DenseMap<Value, SmallVector<NonBlockingAssignOp>> candidateAssigns;
  DenseSet<Value> rejected;

  module.walk([&](ProcedureOp proc) {
    ConstantEvaluator evaluator;

    proc.walk([&](NonBlockingAssignOp assign) {
      Value dst = assign.getDst();

      // Only consider whole-variable assignments.
      auto varOp = dst.getDefiningOp<VariableOp>();
      if (!varOp) {
        rejected.insert(dst);
        return;
      }

      if (rejected.contains(dst))
        return;

      auto attrOpt = evaluator.evaluate(assign.getSrc());
      if (!attrOpt) {
        rejected.insert(dst);
        return;
      }

      auto inserted = candidateValues.try_emplace(dst, *attrOpt);
      if (!inserted.second && inserted.first->second != *attrOpt) {
        rejected.insert(dst);
        candidateValues.erase(dst);
        candidateAssigns.erase(dst);
        return;
      }

      candidateAssigns[dst].push_back(assign);
    });
  });

  // Filter out any destinations we rejected while walking.
  for (Value dst : rejected) {
    candidateValues.erase(dst);
    candidateAssigns.erase(dst);
  }

  if (candidateValues.empty())
    return;

  for (auto &[dst, attr] : candidateValues) {
    auto varOp = dst.getDefiningOp<VariableOp>();
    if (!varOp)
      continue;

    auto refType = dyn_cast<RefType>(varOp.getType());
    if (!refType)
      continue;
    auto elemType = dyn_cast<IntType>(refType.getNestedType());
    if (!elemType)
      continue;

    // Ensure no unexpected users remain.
    bool hasUnexpectedUser = llvm::any_of(varOp->getUsers(), [&](Operation *op) {
      if (isa<ReadOp>(op))
        return false;
      if (auto nb = dyn_cast<NonBlockingAssignOp>(op))
        return nb.getDst() != dst;
      return true;
    });
    if (hasUnexpectedUser)
      continue;

    SmallVector<ReadOp> reads;
    for (Operation *user :
         llvm::make_early_inc_range(varOp->getUsers())) {
      if (auto read = dyn_cast<ReadOp>(user))
        reads.push_back(read);
    }

    for (ReadOp read : reads) {
      OpBuilder builder(read);
      Value constVal =
          ConstantOp::create(builder, read.getLoc(), elemType,
                              attr.getValue());
      read.replaceAllUsesWith(constVal);
      read.erase();
    }

    if (auto it = candidateAssigns.find(dst); it != candidateAssigns.end()) {
      for (NonBlockingAssignOp assign : it->second)
        assign.erase();
    }

    if (varOp->use_empty()) {
      LLVM_DEBUG({
        std::string varName = "<unnamed>";
        if (auto nameAttr = varOp.getNameAttr();
            nameAttr && !nameAttr.getValue().empty())
          varName = nameAttr.getValue().str();
        llvm::dbgs() << "FoldStaticRegisters: removed "
                     << moduleStr << "::" << varName << "\n";
      });
      varOp.erase();
    }
  }
}

} // namespace

std::unique_ptr<mlir::Pass> createFoldStaticRegistersPass() {
  return std::make_unique<FoldStaticRegistersPass>();
}

} // namespace circt::pcov::optimize::moore
