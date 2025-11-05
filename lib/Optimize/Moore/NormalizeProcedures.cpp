//===- NormalizeProcedures.cpp - Normalize Moore Procedures ----*- C++ -*-===//
//
// Part of the circt-cf project.
//
//===----------------------------------------------------------------------===//

#include "circt-cf/Optimize/Moore/Passes.h"

#include "circt/Dialect/Moore/MooreOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "moore-normalize-procedures"

using namespace mlir;
using namespace circt;
using namespace ::circt::moore;

namespace circt::pcov::optimize::moore {
#define GEN_PASS_DEF_NORMALIZEPROCEDURES
#include "circt-cf/Optimize/Moore/Passes.h.inc"
} // namespace circt::pcov::optimize::moore

namespace circt::pcov::optimize::moore {
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
  bool convertToAlwaysComb(ProcedureOp proc, const std::string &moduleName);
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
    return false;
  }

  SensitivityAnalyzer sensitivity;
  if (!sensitivity.analyze(waitEvent))
    return false;

  ReadAnalyzer reads;
  reads.analyze(proc.getBody());

  for (Value readVal : reads.readValues) {
    if (!sensitivity.covers(readVal)) {
      return false;
    }
  }
  return true;
}

bool NormalizeProceduresPass::convertToAlwaysComb(ProcedureOp proc,
                                                  const std::string &moduleName) {
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
  std::string locStr;
  llvm::raw_string_ostream os(locStr);
  proc.getLoc().print(os);
  os.flush();

  std::string message = "NormalizeProcedures: converted ";
  message += moduleName;
  message += " @ ";
  message += locStr;
  message += "\n";
  LLVM_DEBUG(llvm::dbgs() << message);

  return true;
}

void NormalizeProceduresPass::runOnOperation() {
  auto module = getOperation();
  auto moduleNameAttr = module.getSymNameAttr();
  std::string moduleName =
      (moduleNameAttr && !moduleNameAttr.getValue().empty())
          ? moduleNameAttr.getValue().str()
          : "<anonymous>";

  SmallVector<ProcedureOp> toConvert;
  module.walk([&](ProcedureOp proc) {
    if (canConvertToAlwaysComb(proc))
      toConvert.push_back(proc);
  });

  for (ProcedureOp proc : toConvert)
    convertToAlwaysComb(proc, moduleName);
}

} // namespace

std::unique_ptr<mlir::Pass> createNormalizeProceduresPass() {
  return std::make_unique<NormalizeProceduresPass>();
}

} // namespace circt::pcov::optimize::moore
