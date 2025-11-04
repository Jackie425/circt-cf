//===- MooreInstrumentCoverage.cpp - Moore coverage instrumentation -------===//
//
// Part of the circt-cf project.
//
//===----------------------------------------------------------------------===//

#include "circt-cf/Instrumentation/Passes.h"

#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Moore/MooreTypes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cassert>
#include <optional>
#include <utility>

using namespace mlir;
using namespace circt;

namespace circt::pcov {
#define GEN_PASS_DEF_MOOREINSTRUMENTCOVERAGE
#include "circt-cf/Instrumentation/InstrumentationPasses.h.inc"
} // namespace circt::pcov

namespace circt::pcov {
namespace {

struct CoverageVars {
  Value cfFpPrevReg;
  Value cfFpCurrReg;
  Value cfTransSigReg;
  Value cfTransBitmapReg;

  bool isValid() const {
    return cfFpPrevReg && cfFpCurrReg && cfTransSigReg && cfTransBitmapReg;
  }
};

struct CoverageConfig {
  unsigned fpWidth = 0;
  unsigned transSigWidth = 0;
  unsigned transBitmapWidth = 128;

  moore::IntType fpType;
  moore::RefType fpRefType;

  moore::IntType transSigType;
  moore::RefType transSigRefType;

  moore::IntType transBitmapType;
  moore::RefType transBitmapRefType;
};

struct MergeSliceInfo {
  unsigned start = 0;
  unsigned width = 0;
  APInt clearMask;
  llvm::DenseMap<Block *, unsigned> predCode;
  llvm::DenseMap<Block *, APInt> insertConst;

  MergeSliceInfo() = default;
  MergeSliceInfo(unsigned start, unsigned width, const APInt &clearMask)
      : start(start), width(width), clearMask(clearMask) {}
};

struct ProcedureAnalysis {
  Block *returnBlock = nullptr;
  llvm::DenseMap<Block *, MergeSliceInfo> mergeInfoByBlock;
  unsigned fpWidth = 0;
  llvm::SmallVector<Block *> exitBlocks;
  llvm::DenseMap<Block *, unsigned> exitBlockIndex;
};

class MooreInstrumentCoveragePass
    : public impl::MooreInstrumentCoverageBase<MooreInstrumentCoveragePass> {
public:
  void runOnOperation() override;

private:
  CoverageConfig buildCoverageConfig(mlir::MLIRContext *context,
                                     unsigned fpWidth) const;
  std::optional<ProcedureAnalysis>
  analyzeProcedure(moore::ProcedureOp proc) const;
  CoverageVars getOrCreateCoverageVars(moore::ProcedureOp proc,
                                       unsigned procIndex,
                                       const CoverageConfig &config);

  void addCoverageEntryAndArguments(moore::ProcedureOp proc,
                                    const CoverageConfig &config,
                                    const ProcedureAnalysis &analysis);
  void rewriteTerminators(moore::ProcedureOp proc,
                          const CoverageConfig &config,
                          const ProcedureAnalysis &analysis);
  void instrumentExitBlocks(moore::ProcedureOp proc, unsigned procIndex,
                            const CoverageVars &vars,
                            const CoverageConfig &config,
                            const ProcedureAnalysis &analysis,
                            UnitAttr instrumentedAttr);

  Value createConstant(OpBuilder &builder, Location loc, moore::IntType type,
                       const APInt &value) const;
  Value createZeroConstant(OpBuilder &builder, Location loc,
                           moore::IntType type) const;
  Value updateFootprintForSuccessor(OpBuilder &builder, Location loc,
                                    Block *pred, Block *succ, Value fpAcc,
                                    const CoverageConfig &config,
                                    const ProcedureAnalysis &analysis) const;
  Value buildTransitionHash(OpBuilder &builder, Location loc,
                            const CoverageConfig &config,
                            Value transSig) const;

  llvm::DenseMap<Operation *, CoverageVars> coverageVars;
};

CoverageConfig
MooreInstrumentCoveragePass::buildCoverageConfig(mlir::MLIRContext *context,
                                                 unsigned fpWidth) const {
  CoverageConfig config;
  config.fpWidth = fpWidth;
  config.transSigWidth = fpWidth * 2;
  config.transBitmapWidth = 128;

  config.fpType = moore::IntType::getLogic(context, fpWidth);
  config.fpRefType =
      moore::RefType::get(llvm::cast<moore::UnpackedType>(config.fpType));

  config.transSigType = moore::IntType::getLogic(context, config.transSigWidth);
  config.transSigRefType = moore::RefType::get(
      llvm::cast<moore::UnpackedType>(config.transSigType));

  config.transBitmapType =
      moore::IntType::getLogic(context, config.transBitmapWidth);
  config.transBitmapRefType = moore::RefType::get(
      llvm::cast<moore::UnpackedType>(config.transBitmapType));

  return config;
}

std::optional<ProcedureAnalysis>
MooreInstrumentCoveragePass::analyzeProcedure(moore::ProcedureOp proc) const {
  ProcedureAnalysis analysis;

  // Locate the unique return block.
  for (Block &block : proc.getBody()) {
    if (isa_and_nonnull<moore::ReturnOp>(block.getTerminator())) {
      analysis.returnBlock = &block;
      break;
    }
  }

  if (!analysis.returnBlock) {
    proc.emitError("procedure lacks a return block, cannot instrument coverage");
    return std::nullopt;
  }

  llvm::DenseMap<Block *, llvm::SmallVector<Block *, 4>> predecessors;

  // Gather predecessor information and identify exit blocks.
  for (Block &block : proc.getBody()) {
    Operation *terminator = block.getTerminator();

    if (auto cond = dyn_cast<cf::CondBranchOp>(terminator)) {
      predecessors[cond.getTrueDest()].push_back(&block);
      predecessors[cond.getFalseDest()].push_back(&block);
    } else if (auto br = dyn_cast<cf::BranchOp>(terminator)) {
      predecessors[br.getDest()].push_back(&block);

      if (&block != analysis.returnBlock &&
          br.getDest() == analysis.returnBlock) {
        unsigned exitIndex = analysis.exitBlocks.size();
        analysis.exitBlocks.push_back(&block);
        analysis.exitBlockIndex.try_emplace(&block, exitIndex);
      }
    }
  }

  struct MergeTmpInfo {
    Block *block = nullptr;
    llvm::SmallVector<Block *, 4> preds;
    unsigned start = 0;
    unsigned width = 0;
  };
  llvm::SmallVector<MergeTmpInfo> mergeTmpInfos;

  unsigned offset = 0;
  for (Block &block : proc.getBody()) {
    Block *blockPtr = &block;
    auto predIt = predecessors.find(blockPtr);
    unsigned predCount =
        predIt != predecessors.end() ? predIt->second.size() : 0;
    bool isMerge = predCount >= 2 || blockPtr == analysis.returnBlock;
    if (!isMerge)
      continue;

    MergeTmpInfo info;
    info.block = blockPtr;
    if (predIt != predecessors.end())
      info.preds = predIt->second;
    info.width =
        std::max(1u, llvm::Log2_64_Ceil(static_cast<uint64_t>(info.preds.size()) +
                                        1));
    info.start = offset;
    offset += info.width;
    mergeTmpInfos.push_back(std::move(info));
  }

  analysis.fpWidth = offset;

  if (analysis.fpWidth == 0)
    return analysis;

  for (const MergeTmpInfo &tmp : mergeTmpInfos) {
    APInt sliceMask(analysis.fpWidth, 0);
    for (unsigned i = 0; i < tmp.width; ++i)
      sliceMask.setBit(tmp.start + i);
    APInt clearMask = ~sliceMask;

    MergeSliceInfo info(tmp.start, tmp.width, clearMask);

    unsigned code = 1;
    for (Block *pred : tmp.preds) {
      info.predCode.try_emplace(pred, code);

      APInt codeValue(tmp.width, code);
      APInt insertValue = codeValue.zextOrTrunc(analysis.fpWidth);
      if (tmp.start != 0)
        insertValue <<= tmp.start;
      info.insertConst.try_emplace(pred, insertValue);
      ++code;
    }

    analysis.mergeInfoByBlock.try_emplace(tmp.block, std::move(info));
  }

  return analysis;
}

CoverageVars MooreInstrumentCoveragePass::getOrCreateCoverageVars(
    moore::ProcedureOp proc, unsigned procIndex,
    const CoverageConfig &config) {
  Operation *key = proc.getOperation();
  auto it = coverageVars.find(key);
  if (it != coverageVars.end())
    return it->second;

  auto module = proc->getParentOfType<moore::SVModuleOp>();
  if (!module) {
    proc.emitError("expected procedure to be nested within a moore.module");
    signalPassFailure();
    return {};
  }

  OpBuilder builder(module);
  builder.setInsertionPoint(proc);
  Location loc = proc.getLoc();

  auto makeName = [&](StringRef suffix) {
    return (Twine("cov_proc") + Twine(procIndex) + "_" + suffix).str();
  };

  auto zeroFp = createZeroConstant(builder, loc, config.fpType);
  auto zeroTransSig = createZeroConstant(builder, loc, config.transSigType);
  auto zeroTransBitmap =
      createZeroConstant(builder, loc, config.transBitmapType);

  Value cfFpPrev = moore::VariableOp::create(
      builder, loc, config.fpRefType,
      builder.getStringAttr(makeName("cf_fp_prev_reg")), zeroFp);
  cfFpPrev.getDefiningOp()->setAttr(
      "pcov.coverage.role", builder.getStringAttr("cf_fp_prev"));

  Value cfFpCurr = moore::VariableOp::create(
      builder, loc, config.fpRefType,
      builder.getStringAttr(makeName("cf_fp_curr_reg")), zeroFp);
  cfFpCurr.getDefiningOp()->setAttr(
      "pcov.coverage.role", builder.getStringAttr("cf_fp_curr"));

  Value cfTransSig = moore::VariableOp::create(
      builder, loc, config.transSigRefType,
      builder.getStringAttr(makeName("cf_trans_sig_reg")), zeroTransSig);
  cfTransSig.getDefiningOp()->setAttr(
      "pcov.coverage.role", builder.getStringAttr("cf_trans_sig"));

  Value cfTransBitmap = moore::VariableOp::create(
      builder, loc, config.transBitmapRefType,
      builder.getStringAttr(makeName("cf_trans_bitmap_reg")),
      zeroTransBitmap);
  cfTransBitmap.getDefiningOp()->setAttr(
      "pcov.coverage.role", builder.getStringAttr("cf_trans_bitmap"));

  CoverageVars vars{cfFpPrev, cfFpCurr, cfTransSig, cfTransBitmap};
  coverageVars.try_emplace(key, vars);
  return vars;
}

Value MooreInstrumentCoveragePass::createConstant(OpBuilder &builder,
                                                  Location loc,
                                                  moore::IntType type,
                                                  const APInt &value) const {
  return moore::ConstantOp::create(builder, loc, type, value);
}

Value MooreInstrumentCoveragePass::createZeroConstant(OpBuilder &builder,
                                                      Location loc,
                                                      moore::IntType type) const {
  return createConstant(builder, loc, type, APInt(type.getWidth(), 0));
}

Value MooreInstrumentCoveragePass::updateFootprintForSuccessor(
    OpBuilder &builder, Location loc, Block *pred, Block *succ, Value fpAcc,
    const CoverageConfig &config,
    const ProcedureAnalysis &analysis) const {
  auto mergeInfoIt = analysis.mergeInfoByBlock.find(succ);
  if (mergeInfoIt == analysis.mergeInfoByBlock.end())
    return fpAcc;

  const MergeSliceInfo &info = mergeInfoIt->second;
  if (info.width == 0)
    return fpAcc;
  auto insertIt = info.insertConst.find(pred);
  assert(insertIt != info.insertConst.end() &&
         "expected predecessor to have insert constant");

  Value clearMaskConst =
      createConstant(builder, loc, config.fpType, info.clearMask);
  Value insertConst =
      createConstant(builder, loc, config.fpType, insertIt->second);

  Value cleared = builder.create<moore::AndOp>(loc, config.fpType, fpAcc,
                                               clearMaskConst);
  return builder.create<moore::OrOp>(loc, config.fpType, cleared, insertConst);
}

void MooreInstrumentCoveragePass::addCoverageEntryAndArguments(
    moore::ProcedureOp proc, const CoverageConfig &config,
    const ProcedureAnalysis &analysis) {
  Region &body = proc.getBody();
  Block &originalEntry = body.front();

  // Create the new coverage entry block at the beginning of the region.
  auto *coverageEntry = new Block();
  body.getBlocks().insert(body.begin(), coverageEntry);

  OpBuilder builder(proc.getContext());
  builder.setInsertionPointToStart(coverageEntry);
  Location loc = proc.getLoc();

  Value fpInit = createZeroConstant(builder, loc, config.fpType);
  builder.create<cf::BranchOp>(loc, &originalEntry, ValueRange{fpInit});

  // Add the footprint accumulator argument to every original block.
  for (Block &block : body) {
    if (&block == coverageEntry)
      continue;
    block.insertArgument(0u, config.fpType, loc);
  }
}

void MooreInstrumentCoveragePass::rewriteTerminators(
    moore::ProcedureOp proc, const CoverageConfig &config,
    const ProcedureAnalysis &analysis) {
  for (Block &block : proc.getBody()) {
    // The synthetic coverage entry block has no block argument; skip it.
    if (block.getNumArguments() == 0)
      continue;

    Operation *terminator = block.getTerminator();
    OpBuilder builder(terminator);
    Location loc = terminator->getLoc();
    Value fpAcc = block.getArgument(0);

    if (auto cond = dyn_cast<cf::CondBranchOp>(terminator)) {
      Block *trueDest = cond.getTrueDest();
      Block *falseDest = cond.getFalseDest();

      Value fpTrue =
          updateFootprintForSuccessor(builder, loc, &block, trueDest, fpAcc,
                                      config, analysis);
      Value fpFalse =
          updateFootprintForSuccessor(builder, loc, &block, falseDest, fpAcc,
                                      config, analysis);

      SmallVector<Value> trueArgs;
      trueArgs.push_back(fpTrue);
      trueArgs.append(cond.getTrueOperands().begin(),
                      cond.getTrueOperands().end());

      SmallVector<Value> falseArgs;
      falseArgs.push_back(fpFalse);
      falseArgs.append(cond.getFalseOperands().begin(),
                       cond.getFalseOperands().end());

      builder.create<cf::CondBranchOp>(loc, cond.getCondition(), trueDest,
                                       trueArgs, falseDest, falseArgs);
      cond.erase();
      continue;
    }

    if (auto br = dyn_cast<cf::BranchOp>(terminator)) {
      Block *dest = br.getDest();
      Value fpNext = updateFootprintForSuccessor(builder, loc, &block, dest,
                                                 fpAcc, config, analysis);

      SmallVector<Value> operands;
      operands.push_back(fpNext);
      operands.append(br.getDestOperands().begin(),
                      br.getDestOperands().end());
      builder.create<cf::BranchOp>(loc, dest, operands);
      br.erase();
      continue;
    }
  }
}

Value MooreInstrumentCoveragePass::buildTransitionHash(
    OpBuilder &builder, Location loc, const CoverageConfig &config,
    Value transSig) const {
  unsigned sigWidth = config.transSigWidth;
  unsigned hashWidth = config.transBitmapWidth;

  if (sigWidth <= hashWidth) {
    if (sigWidth == hashWidth)
      return transSig;
    return builder.create<moore::ZExtOp>(loc, config.transBitmapType, transSig);
  }

  Value hash =
      createZeroConstant(builder, loc, config.transBitmapType);

  for (unsigned offset = 0; offset < sigWidth; offset += hashWidth) {
    unsigned chunkWidth = std::min(hashWidth, sigWidth - offset);

    Value shifted = transSig;
    if (offset != 0) {
      APInt shiftValue(config.transSigWidth, offset);
      Value shiftConst =
          createConstant(builder, loc, config.transSigType, shiftValue);
      shifted = builder.create<moore::ShrOp>(loc, config.transSigType, shifted,
                                             shiftConst);
    }

    moore::IntType chunkType =
        moore::IntType::getLogic(builder.getContext(), chunkWidth);
    Value chunk = builder.create<moore::TruncOp>(loc, chunkType, shifted);
    if (chunkWidth < hashWidth)
      chunk = builder.create<moore::ZExtOp>(loc, config.transBitmapType, chunk);

    hash =
        builder.create<moore::XorOp>(loc, config.transBitmapType, hash, chunk);
  }

  return hash;
}

void MooreInstrumentCoveragePass::instrumentExitBlocks(
    moore::ProcedureOp proc, unsigned procIndex, const CoverageVars &vars,
    const CoverageConfig &config, const ProcedureAnalysis &analysis,
    UnitAttr instrumentedAttr) {
  Builder attrBuilder(proc.getContext());

  for (Block *exitBlock : analysis.exitBlocks) {
    Operation *terminator = exitBlock->getTerminator();
    OpBuilder builder(terminator);
    Location loc = terminator->getLoc();

    auto branch = dyn_cast<cf::BranchOp>(terminator);
    assert(branch && "exit block must terminate with cf.br");
    Value fpFinal = branch.getDestOperands().front();

    // Write current footprint.
    moore::NonBlockingAssignOp::create(builder, loc, vars.cfFpCurrReg, fpFinal);

    // Read previous footprint.
    Value prevFp =
        moore::ReadOp::create(builder, loc, vars.cfFpPrevReg);

    // Build transition signature.
    Value prevExt =
        builder.create<moore::ZExtOp>(loc, config.transSigType, prevFp);
    Value currExt =
        builder.create<moore::ZExtOp>(loc, config.transSigType, fpFinal);

    APInt shiftValue(config.transSigWidth, config.fpWidth);
    Value shiftConst =
        createConstant(builder, loc, config.transSigType, shiftValue);
    Value prevShift =
        builder.create<moore::ShlOp>(loc, config.transSigType, prevExt,
                                     shiftConst);
    Value transSig =
        builder.create<moore::OrOp>(loc, config.transSigType, prevShift,
                                    currExt);

    moore::NonBlockingAssignOp::create(builder, loc, vars.cfTransSigReg,
                                       transSig);

    // Compute transition hash and update bitmap register.
    Value transHash = buildTransitionHash(builder, loc, config, transSig);
    moore::NonBlockingAssignOp::create(builder, loc, vars.cfTransBitmapReg,
                                       transHash);

    // Roll the previous footprint.
    moore::NonBlockingAssignOp::create(builder, loc, vars.cfFpPrevReg, fpFinal);

    auto exitIndexIt = analysis.exitBlockIndex.find(exitBlock);
    assert(exitIndexIt != analysis.exitBlockIndex.end() &&
           "missing exit block index");
    unsigned recordedExitIndex = exitIndexIt->second;
    IntegerAttr exitIndexAttr =
        attrBuilder.getI32IntegerAttr(recordedExitIndex);
    terminator->setAttr("pcov.coverage.instrumented", instrumentedAttr);
    terminator->setAttr("pcov.cff_exit_block", exitIndexAttr);
    terminator->setAttr("pcov.leaf_id", exitIndexAttr);
  }

  // Annotate the procedure itself.
  Builder builder(proc.getContext());
  proc->setAttr("pcov.coverage.instrumented", instrumentedAttr);
  proc->setAttr("pcov.coverage.proc_index",
                builder.getI32IntegerAttr(procIndex));
  proc->setAttr("pcov.coverage.fp_width",
                builder.getI32IntegerAttr(config.fpWidth));
}

void MooreInstrumentCoveragePass::runOnOperation() {
  moore::SVModuleOp module = getOperation();
  auto *context = module.getContext();
  UnitAttr instrumentedAttr = UnitAttr::get(context);

  unsigned procIndex = 0;
  for (moore::ProcedureOp proc : module.getOps<moore::ProcedureOp>()) {
    auto analysisOpt = analyzeProcedure(proc);
    if (!analysisOpt) {
      signalPassFailure();
      return;
    }
    ProcedureAnalysis analysis = *analysisOpt;

    // If there are no conditional edges, record metadata and move on.
    if (analysis.fpWidth == 0) {
      Builder builder(context);
      proc->setAttr("pcov.coverage.proc_index",
                    builder.getI32IntegerAttr(procIndex));
      proc->setAttr("pcov.coverage.fp_width", builder.getI32IntegerAttr(0));
      ++procIndex;
      continue;
    }

    CoverageConfig config = buildCoverageConfig(context, analysis.fpWidth);
    CoverageVars vars = getOrCreateCoverageVars(proc, procIndex, config);
    if (!vars.isValid())
      return;

    addCoverageEntryAndArguments(proc, config, analysis);
    rewriteTerminators(proc, config, analysis);
    instrumentExitBlocks(proc, procIndex, vars, config, analysis,
                         instrumentedAttr);

    ++procIndex;
  }
}

} // namespace

std::unique_ptr<mlir::Pass> createMooreInstrumentCoveragePass() {
  return std::make_unique<MooreInstrumentCoveragePass>();
}

void registerMooreInstrumentCoveragePass() {
  mlir::PassRegistration<MooreInstrumentCoveragePass>();
}

} // namespace circt::pcov
