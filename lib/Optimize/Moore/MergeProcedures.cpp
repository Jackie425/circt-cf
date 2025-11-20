//===- MergeProcedures.cpp - Merge simple Moore procedures -----*- C++ -*-===//
//
// Part of the pcov project.
//
//===----------------------------------------------------------------------===//

#include "pcov/Optimize/Moore/Passes.h"

#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Moore/MooreTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "moore-merge-procedures"

using namespace mlir;
using namespace circt;
using namespace ::circt::moore;

namespace circt::pcov::optimize::moore {
#define GEN_PASS_DEF_MERGEPROCEDURES
#include "pcov/Optimize/Moore/Passes.h.inc"
} // namespace circt::pcov::optimize::moore

namespace circt::pcov::optimize::moore {
namespace {

struct BitAssignInfo {
  ProcedureOp proc;
  ExtractRefOp dstExtract;
  ReadOp srcRead;
  ExtractOp srcExtract;
  BlockingAssignOp assign;
  int64_t bitIndex;
};

struct MergeKey {
  Value dstBase;
  Value srcBase;
};

struct MergeKeyInfo : DenseMapInfo<MergeKey> {
  static MergeKey getEmptyKey() {
    return {DenseMapInfo<Value>::getEmptyKey(),
            DenseMapInfo<Value>::getEmptyKey()};
  }

  static MergeKey getTombstoneKey() {
    return {DenseMapInfo<Value>::getTombstoneKey(),
            DenseMapInfo<Value>::getTombstoneKey()};
  }

  static unsigned getHashValue(const MergeKey &key) {
    return llvm::hash_combine(key.dstBase, key.srcBase);
  }

  static bool isEqual(const MergeKey &lhs, const MergeKey &rhs) {
    return lhs.dstBase == rhs.dstBase && lhs.srcBase == rhs.srcBase;
  }
};

class MergeProceduresPass
    : public impl::MergeProceduresBase<MergeProceduresPass> {
public:
  void runOnOperation() override;

private:
  std::optional<BitAssignInfo> matchSimpleBitAssign(ProcedureOp proc);
  bool mergeGroup(ArrayRef<BitAssignInfo> group, StringRef moduleName);
};

/// Match the specific pattern we see from expanded generates:
///   %dst = moore.extract_ref %dstBase from const
///   %srcVal = moore.read %srcBase
///   %src = moore.extract %srcVal from const
///   moore.blocking_assign %dst, %src
///   moore.return
std::optional<BitAssignInfo>
MergeProceduresPass::matchSimpleBitAssign(ProcedureOp proc) {
  if (proc.getKind() != ProcedureKind::AlwaysComb)
    return std::nullopt;

  Block &body = proc.getBody().front();
  if (body.getOperations().size() != 5)
    return std::nullopt;

  auto it = body.begin();
  auto dstExtract = dyn_cast<ExtractRefOp>(&*it++);
  auto srcRead = dyn_cast<ReadOp>(&*it++);
  auto srcExtract = dyn_cast<ExtractOp>(&*it++);
  auto assign = dyn_cast<BlockingAssignOp>(&*it++);
  auto ret = dyn_cast<ReturnOp>(&*it++);

  if (!dstExtract || !srcRead || !srcExtract || !assign || !ret)
    return std::nullopt;

  if (assign.getDst() != dstExtract.getResult())
    return std::nullopt;
  if (assign.getSrc() != srcExtract.getResult())
    return std::nullopt;
  if (srcExtract.getInput() != srcRead.getResult())
    return std::nullopt;

  auto dstRefType = dyn_cast<RefType>(dstExtract.getResult().getType());
  if (!dstRefType)
    return std::nullopt;
  if (dstRefType.getNestedType() != srcExtract.getResult().getType())
    return std::nullopt;

  // We only merge single-bit assignments to keep the transformation safe.
  if (auto intType = dyn_cast<IntType>(dstRefType.getNestedType())) {
    if (intType.getWidth() != 1)
      return std::nullopt;
  } else {
    return std::nullopt;
  }

  if (dstExtract.getLowBit() != srcExtract.getLowBit())
    return std::nullopt;

  BitAssignInfo info{proc, dstExtract, srcRead, srcExtract, assign,
                     dstExtract.getLowBit()};
  return info;
}

bool MergeProceduresPass::mergeGroup(ArrayRef<BitAssignInfo> group,
                                     StringRef moduleName) {
  if (group.size() < 2)
    return false;

  DenseSet<int64_t> seenBits;
  for (const auto &info : group) {
    if (!seenBits.insert(info.bitIndex).second)
      return false;
  }

  SmallVector<BitAssignInfo> ordered(group.begin(), group.end());
  llvm::sort(ordered, [](const BitAssignInfo &lhs, const BitAssignInfo &rhs) {
    return lhs.bitIndex < rhs.bitIndex;
  });

  ProcedureOp baseProc = ordered.front().proc;
  Block &baseBlock = baseProc.getBody().front();
  Operation *insertBefore = baseBlock.getTerminator();

  unsigned merged = 0;
  for (auto it = std::next(ordered.begin()); it != ordered.end(); ++it) {
    Block &otherBlock = it->proc.getBody().front();
    for (Operation &op :
         llvm::make_early_inc_range(otherBlock.getOperations())) {
      if (isa<ReturnOp>(op))
        continue;
      op.moveBefore(insertBefore);
    }
    it->proc.erase();
    ++merged;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "MergeProcedures: merged " << merged
                 << " always_comb procedures assigning bits of "
                 << moduleName << "\n";
  });

  return merged != 0;
}

void MergeProceduresPass::runOnOperation() {
  auto module = getOperation();
  StringRef moduleName = module.getSymNameAttr()
                             ? module.getSymNameAttr().getValue()
                             : "<anonymous>";

  DenseMap<MergeKey, SmallVector<BitAssignInfo>, MergeKeyInfo> groups;
  module.walk([&](ProcedureOp proc) {
    if (auto info = matchSimpleBitAssign(proc)) {
      MergeKey key{info->dstExtract.getInput(), info->srcRead.getInput()};
      groups[key].push_back(*info);
    }
  });

  unsigned mergedGroups = 0;
  for (auto &entry : groups)
    if (mergeGroup(entry.second, moduleName))
      ++mergedGroups;

  LLVM_DEBUG({
    if (mergedGroups != 0)
      llvm::dbgs() << "MergeProcedures: merged " << mergedGroups
                   << " procedure groups in module " << moduleName << "\n";
  });
}

} // namespace

std::unique_ptr<mlir::Pass> createMergeProceduresPass() {
  return std::make_unique<MergeProceduresPass>();
}

} // namespace circt::pcov::optimize::moore
