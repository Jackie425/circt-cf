//===- NormalizeProcedures.cpp - Normalize Moore Procedures ----*- C++ -*-===//
//
// Part of the circt-cfa-trace project.
//
//===----------------------------------------------------------------------===//

#include "circt-cfa-trace/Optimize/Moore/Passes.h"

#include "circt/Dialect/Moore/MooreOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "moore-normalize-procedures"

using namespace mlir;
using namespace circt;
using namespace ::circt::moore;

namespace circt::cfatrace::optimize::moore {
#define GEN_PASS_DEF_NORMALIZEPROCEDURES
#include "circt-cfa-trace/Optimize/Moore/Passes.h.inc"
} // namespace circt::cfatrace::optimize::moore

namespace circt::cfatrace::optimize::moore {
namespace {

struct ReadAnalyzer {
  SmallPtrSet<Value, 16> readValues;

  void analyze(Region &region) {
    region.walk([&](Operation *op) {
      if (auto readOp = dyn_cast<ReadOp>(op)) {
        Value input = readOp.getInput();

        bool isOnlyLHS = true;
        for (Operation *user : readOp->getUsers()) {
          if (auto assignOp = dyn_cast<BlockingAssignOp>(user)) {
            if (assignOp.getDst() == readOp.getResult())
              continue;
          }
          isOnlyLHS = false;
          break;
        }

        if (isOnlyLHS)
          return;

        if (auto netOp = input.getDefiningOp<NetOp>()) {
          if (Value assignValue = netOp.getAssignment()) {
            if (auto *def = assignValue.getDefiningOp();
                def && def->hasTrait<OpTrait::ConstantLike>())
              return;
          }
        }

        readValues.insert(input);
      }
    });
  }
};

struct SensitivityAnalyzer {
  SmallPtrSet<Value, 16> sensitiveValues;

  bool analyze(WaitEventOp waitEvent) {
    for (Operation &op : waitEvent.getBody().front()) {
      if (auto detectOp = dyn_cast<DetectEventOp>(&op)) {
        Value input = detectOp.getInput();
        if (auto readOp = input.getDefiningOp<ReadOp>())
          input = readOp.getInput();
        sensitiveValues.insert(input);
      }
    }
    return !sensitiveValues.empty();
  }

  bool covers(Value val) const { return sensitiveValues.contains(val); }
};

class NormalizeProceduresPass
    : public impl::NormalizeProceduresBase<NormalizeProceduresPass> {
public:
  void runOnOperation() override;

private:
  bool canConvertToAlwaysComb(ProcedureOp proc);
  void convertToAlwaysComb(ProcedureOp proc);
};

bool NormalizeProceduresPass::canConvertToAlwaysComb(ProcedureOp proc) {
  if (proc.getKind() != ProcedureKind::Always)
    return false;

  Block &body = proc.getBody().front();
  if (body.empty())
    return false;

  auto waitEvent = dyn_cast<WaitEventOp>(&body.front());
  if (!waitEvent)
    return false;

  bool hasSequentialFeatures = false;
  proc.walk([&](Operation *op) {
    if (isa<NonBlockingAssignOp>(op))
      hasSequentialFeatures = true;
  });

  if (hasSequentialFeatures) {
    LLVM_DEBUG(llvm::dbgs() << "Procedure at " << proc.getLoc()
                            << " has non-blocking assignments\n");
    return false;
  }

  SensitivityAnalyzer sensitivity;
  if (!sensitivity.analyze(waitEvent)) {
    LLVM_DEBUG(llvm::dbgs() << "Procedure at " << proc.getLoc()
                            << " has no sensitivity list\n");
    return false;
  }

  ReadAnalyzer reads;
  reads.analyze(proc.getBody());

  for (Value readVal : reads.readValues) {
    if (!sensitivity.covers(readVal)) {
      LLVM_DEBUG({
        llvm::dbgs() << "Procedure at " << proc.getLoc()
                     << " reads value not in sensitivity list:\n";
        llvm::dbgs() << "  Read value: " << readVal << "\n";
        if (auto defOp = readVal.getDefiningOp())
          llvm::dbgs() << "  Defined by: " << *defOp << "\n";
      });
      return false;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Procedure at " << proc.getLoc()
                          << " can be converted to always_comb\n");
  return true;
}

void NormalizeProceduresPass::convertToAlwaysComb(ProcedureOp proc) {
  OpBuilder builder(proc);

  Block &body = proc.getBody().front();
  auto waitEvent = cast<WaitEventOp>(&body.front());

  Block &waitBody = waitEvent.getBody().front();
  SmallVector<Operation *> toMove;
  for (Operation &op : llvm::make_early_inc_range(waitBody))
    if (!isa<DetectEventOp>(&op))
      toMove.push_back(&op);

  for (Operation *op : toMove)
    op->moveBefore(waitEvent);

  waitEvent.erase();

  proc->setAttr("kind",
                ProcedureKindAttr::get(builder.getContext(),
                                        ProcedureKind::AlwaysComb));

  LLVM_DEBUG(llvm::dbgs() << "Converted procedure to always_comb at "
                          << proc.getLoc() << "\n");
}

void NormalizeProceduresPass::runOnOperation() {
  auto module = getOperation();

  SmallVector<ProcedureOp> toConvert;
  module.walk([&](ProcedureOp proc) {
    if (canConvertToAlwaysComb(proc))
      toConvert.push_back(proc);
  });

  for (ProcedureOp proc : toConvert)
    convertToAlwaysComb(proc);

  LLVM_DEBUG(llvm::dbgs() << "Converted " << toConvert.size()
                          << " procedures to always_comb\n");
}

} // namespace

std::unique_ptr<mlir::Pass> createNormalizeProceduresPass() {
  return std::make_unique<NormalizeProceduresPass>();
}

} // namespace circt::cfatrace::optimize::moore
