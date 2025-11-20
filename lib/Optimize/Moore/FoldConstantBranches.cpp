//===- FoldConstantBranches.cpp - Fold Moore constant branches -*- C++ -*-===//
//
// Part of the pcov project.
//
//===----------------------------------------------------------------------===//

#include "pcov/Optimize/Moore/Passes.h"

#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Moore/MooreTypes.h"
#include "circt/Support/FVInt.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <optional>
#include <string>

#define DEBUG_TYPE "moore-fold-constant-branches"

using namespace mlir;
using namespace circt;
using namespace ::circt::moore;

namespace {

std::optional<unsigned> getBitWidth(Type type) {
  if (auto intType = llvm::dyn_cast<::circt::moore::IntType>(type))
    return intType.getWidth();
  return std::nullopt;
}

std::optional<FVInt> makeBoolFV(bool value, Type type) {
  auto width = getBitWidth(type);
  if (!width)
    return std::nullopt;
  return value ? FVInt::getAllOnes(*width) : FVInt::getZero(*width);
}

} // namespace

namespace circt::pcov::optimize::moore {
#define GEN_PASS_DEF_FOLDCONSTANTBRANCHES
#include "pcov/Optimize/Moore/Passes.h.inc"
} // namespace circt::pcov::optimize::moore

namespace circt::pcov::optimize::moore {

class FoldConstantBranchesPass
    : public impl::FoldConstantBranchesBase<FoldConstantBranchesPass> {
public:
  void runOnOperation() override;

private:
  struct ValueCache {
    llvm::DenseMap<Value, FVInt> known;
    llvm::DenseSet<Value> unknown;
  };

  std::optional<FVInt> evaluateFV(Value value, ValueCache &cache);
  std::optional<bool> evaluateBool(Value value, ValueCache &cache);
  bool tryFoldCondBranch(cf::CondBranchOp condBr, ValueCache &cache,
                         unsigned &foldedBranches,
                         std::optional<int64_t> procIndex);
};

void FoldConstantBranchesPass::runOnOperation() {
  auto module = getOperation();
  unsigned foldedBranches = 0;

  module.walk([&](ProcedureOp proc) {
    ValueCache cache;
    unsigned localFolded = 0;
    std::optional<int64_t> procIndex;
    if (auto attr =
            proc->getAttrOfType<IntegerAttr>("pcov.coverage.proc_index"))
      procIndex = attr.getInt();

    proc.walk([&](cf::CondBranchOp condBr) {
      if (tryFoldCondBranch(condBr, cache, foldedBranches, procIndex))
        ++localFolded;
    });

    if (localFolded != 0) {
      std::string procLoc;
      llvm::raw_string_ostream os(procLoc);
      proc.getLoc().print(os);
      os.flush();

      LLVM_DEBUG({
        llvm::dbgs() << "  procedure at " << procLoc << " folded "
                     << localFolded << " branches";
        if (procIndex)
          llvm::dbgs() << " (proc_index=" << *procIndex << ")";
        llvm::dbgs() << "\n";
      });
    }
  });

  if (foldedBranches != 0) {
    auto moduleName = module.getSymNameAttr();
    LLVM_DEBUG({
      llvm::dbgs() << "FoldConstantBranches: folded " << foldedBranches
                   << " conditional branches in module "
                   << (moduleName ? moduleName.getValue() : "<<anonymous>>")
                   << "\n";
    });
  }
}

std::optional<bool>
FoldConstantBranchesPass::evaluateBool(Value value, ValueCache &cache) {
  llvm::APInt apValue;
  if (matchPattern(value, m_ConstantInt(&apValue)))
    return !apValue.isZero();

  if (auto toBool = value.getDefiningOp<ToBuiltinBoolOp>()) {
    auto boolValue = evaluateFV(toBool.getInput(), cache);
    if (!boolValue || boolValue->hasUnknown())
      return std::nullopt;
    return !boolValue->isZero();
  }

  auto fv = evaluateFV(value, cache);
  if (!fv || fv->hasUnknown())
    return std::nullopt;
  return !fv->isZero();
}

std::optional<FVInt>
FoldConstantBranchesPass::evaluateFV(Value value, ValueCache &cache) {
  if (!value)
    return std::nullopt;

  if (auto it = cache.known.find(value); it != cache.known.end())
    return it->second;
  if (cache.unknown.contains(value))
    return std::nullopt;

  Operation *def = value.getDefiningOp();
  if (!def) {
    cache.unknown.insert(value);
    return std::nullopt;
  }

  std::optional<FVInt> result;

  if (auto constOp = dyn_cast<ConstantOp>(def)) {
    result = constOp.getValueAttr().getValue();
  } else if (auto notOp = dyn_cast<NotOp>(def)) {
    if (auto operand = evaluateFV(notOp.getInput(), cache))
      result = ~(*operand);
  } else if (auto andOp = dyn_cast<AndOp>(def)) {
    auto lhs = evaluateFV(andOp.getLhs(), cache);
    auto rhs = evaluateFV(andOp.getRhs(), cache);
    if (lhs && rhs) {
      FVInt computed = *lhs;
      computed &= *rhs;
      result = std::move(computed);
    }
  } else if (auto orOp = dyn_cast<OrOp>(def)) {
    auto lhs = evaluateFV(orOp.getLhs(), cache);
    auto rhs = evaluateFV(orOp.getRhs(), cache);
    if (lhs && rhs) {
      FVInt computed = *lhs;
      computed |= *rhs;
      result = std::move(computed);
    }
  } else if (auto xorOp = dyn_cast<XorOp>(def)) {
    auto lhs = evaluateFV(xorOp.getLhs(), cache);
    auto rhs = evaluateFV(xorOp.getRhs(), cache);
    if (lhs && rhs) {
      FVInt computed = *lhs;
      computed ^= *rhs;
      result = std::move(computed);
    }
  } else if (auto truncOp = dyn_cast<TruncOp>(def)) {
    if (auto operand = evaluateFV(truncOp.getInput(), cache))
      if (auto width = getBitWidth(truncOp.getType()))
        result = operand->trunc(*width);
  } else if (auto zextOp = dyn_cast<ZExtOp>(def)) {
    if (auto operand = evaluateFV(zextOp.getInput(), cache))
      if (auto width = getBitWidth(zextOp.getType()))
        result = operand->zext(*width);
  } else if (auto sextOp = dyn_cast<SExtOp>(def)) {
    if (auto operand = evaluateFV(sextOp.getInput(), cache))
      if (auto width = getBitWidth(sextOp.getType()))
        result = operand->sext(*width);
  } else if (auto eqOp = dyn_cast<EqOp>(def)) {
    auto lhs = evaluateFV(eqOp.getLhs(), cache);
    auto rhs = evaluateFV(eqOp.getRhs(), cache);
    if (lhs && rhs && !lhs->hasUnknown() && !rhs->hasUnknown())
      if (auto boolVal = makeBoolFV(*lhs == *rhs, eqOp.getResult().getType()))
        result = *boolVal;
  } else if (auto neOp = dyn_cast<NeOp>(def)) {
    auto lhs = evaluateFV(neOp.getLhs(), cache);
    auto rhs = evaluateFV(neOp.getRhs(), cache);
    if (lhs && rhs && !lhs->hasUnknown() && !rhs->hasUnknown())
      if (auto boolVal = makeBoolFV(*lhs != *rhs, neOp.getResult().getType()))
        result = *boolVal;
  } else if (auto caseEqOp = dyn_cast<CaseEqOp>(def)) {
    auto lhs = evaluateFV(caseEqOp.getLhs(), cache);
    auto rhs = evaluateFV(caseEqOp.getRhs(), cache);
    if (lhs && rhs)
      if (auto boolVal =
              makeBoolFV(*lhs == *rhs, caseEqOp.getResult().getType()))
        result = *boolVal;
  } else if (auto caseNeOp = dyn_cast<CaseNeOp>(def)) {
    auto lhs = evaluateFV(caseNeOp.getLhs(), cache);
    auto rhs = evaluateFV(caseNeOp.getRhs(), cache);
    if (lhs && rhs)
      if (auto boolVal =
              makeBoolFV(*lhs != *rhs, caseNeOp.getResult().getType()))
        result = *boolVal;
  } else if (auto caseZEqOp = dyn_cast<CaseZEqOp>(def)) {
    auto lhs = evaluateFV(caseZEqOp.getLhs(), cache);
    auto rhs = evaluateFV(caseZEqOp.getRhs(), cache);
    if (lhs && rhs && !lhs->hasUnknown() && !rhs->hasUnknown())
      if (auto boolVal =
              makeBoolFV(*lhs == *rhs, caseZEqOp.getResult().getType()))
        result = *boolVal;
  } else if (auto caseXZEqOp = dyn_cast<CaseXZEqOp>(def)) {
    auto lhs = evaluateFV(caseXZEqOp.getLhs(), cache);
    auto rhs = evaluateFV(caseXZEqOp.getRhs(), cache);
    if (lhs && rhs && !lhs->hasUnknown() && !rhs->hasUnknown())
      if (auto boolVal =
              makeBoolFV(*lhs == *rhs, caseXZEqOp.getResult().getType()))
        result = *boolVal;
  }

  if (result)
    cache.known.try_emplace(value, *result);
  else
    cache.unknown.insert(value);
  return result;
}

bool FoldConstantBranchesPass::tryFoldCondBranch(cf::CondBranchOp condBr,
                                                 ValueCache &cache,
                                                 unsigned &foldedBranches,
                                                 std::optional<int64_t> procIndex) {
  auto decision = evaluateBool(condBr.getCondition(), cache);
  if (!decision)
    return false;

  OpBuilder builder(condBr);
  if (*decision) {
    builder.create<cf::BranchOp>(condBr.getLoc(), condBr.getTrueDest(),
                                 condBr.getTrueDestOperands());
  } else {
    builder.create<cf::BranchOp>(condBr.getLoc(), condBr.getFalseDest(),
                                 condBr.getFalseDestOperands());
  }

  LLVM_DEBUG({
    std::string locStr;
    llvm::raw_string_ostream os(locStr);
    condBr.getLoc().print(os);
    os.flush();
    llvm::dbgs() << "FoldConstantBranches: resolved conditional branch at "
                 << locStr << " to " << (*decision ? "true" : "false")
                 << " successor";
    if (procIndex)
      llvm::dbgs() << " (proc_index=" << *procIndex << ")";
    llvm::dbgs() << "\n";
  });

  condBr.erase();
  ++foldedBranches;
  return true;
}

std::unique_ptr<mlir::Pass> createFoldConstantBranchesPass() {
  return std::make_unique<FoldConstantBranchesPass>();
}

} // namespace circt::pcov::optimize::moore
