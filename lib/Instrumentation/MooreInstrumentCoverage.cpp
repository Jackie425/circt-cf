//===- MooreInstrumentCoverage.cpp - Moore path coverage instrumentation --===//
//
// Part of the circt-cf project.
//
//===----------------------------------------------------------------------===//

#include "circt-cf/Instrumentation/MooreProcedureAnalysis.h"
#include "circt-cf/Instrumentation/Passes.h"

#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Moore/MooreTypes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
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

struct CoverageConfig {
  unsigned pathIdWidth = 0;
  moore::IntType pathIdType;
  moore::RefType pathIdRefType;
};

struct CoverageVars {
  Value pathIdReg;
  bool isValid() const { return pathIdReg != nullptr; }
};

struct LocalCoverageReg {
  Value reg;
};

struct ModuleCoverageInfo {
  moore::SVModuleOp module;
  SmallVector<LocalCoverageReg, 4> localRegs;
  SmallVector<moore::InstanceOp, 4> instances;
  unsigned localWidth = 0;
  unsigned aggregatedWidth = 0;
};

enum class VisitState { Visiting, Visited };

class MooreInstrumentCoveragePass
    : public impl::MooreInstrumentCoverageBase<MooreInstrumentCoveragePass> {
public:
  void runOnOperation() override;

private:
  CoverageConfig buildCoverageConfig(MLIRContext *context,
                                     unsigned pathIdWidth) const;
  CoverageVars getOrCreateCoverageVars(moore::ProcedureOp proc,
                                       unsigned procIndex,
                                       const CoverageConfig &config);

  void addCoverageEntryAndArguments(moore::ProcedureOp proc,
                                    const CoverageConfig &config,
                                    const MooreProcedureCFGAnalysis &analysis);
  void rewriteTerminators(moore::ProcedureOp proc,
                          const CoverageConfig &config,
                          const MooreProcedureCFGAnalysis &analysis);
  void instrumentExitBlock(moore::ProcedureOp proc, unsigned procIndex,
                           const CoverageVars &vars,
                           UnitAttr instrumentedAttr,
                           const MooreProcedureCFGAnalysis &analysis);

  Value createConstant(OpBuilder &builder, Location loc, moore::IntType type,
                       const APInt &value) const;
  Value createZeroConstant(OpBuilder &builder, Location loc,
                           moore::IntType type) const;
  Value accumulateForEdge(OpBuilder &builder, Location loc, Block *pred,
                          Block *succ, Value pathSum,
                          const CoverageConfig &config,
                          const MooreProcedureCFGAnalysis &analysis) const;

  llvm::DenseMap<Operation *, CoverageVars> coverageVars;
  llvm::DenseMap<Operation *, ModuleCoverageInfo> moduleInfos;
  bool cycleDetected = false;

  LogicalResult instrumentModule(moore::SVModuleOp module,
                                 UnitAttr instrumentedAttr);
  unsigned computeAggregatedWidth(
      moore::SVModuleOp module, llvm::DenseMap<Operation *, VisitState> &state,
      const llvm::StringMap<moore::SVModuleOp> &moduleLookup);
  void addCoveragePortToModule(moore::SVModuleOp module,
                               moore::IntType coverageType,
                               StringAttr coveragePortNameAttr);
  moore::InstanceOp rewriteInstanceWithCoverage(
      moore::InstanceOp inst, moore::IntType coverageType,
      StringAttr coveragePortNameAttr, Value &coverageValue);
  Value buildModuleCoverageValue(
      moore::SVModuleOp module, const ModuleCoverageInfo &info,
      const llvm::DenseMap<Operation *, Value> &instanceCoverageSignals,
      OpBuilder &builder);
};

CoverageConfig
MooreInstrumentCoveragePass::buildCoverageConfig(MLIRContext *context,
                                                 unsigned pathIdWidth) const {
  CoverageConfig config;
  config.pathIdWidth = std::max(1u, pathIdWidth);
  config.pathIdType = moore::IntType::getLogic(context, config.pathIdWidth);
  config.pathIdRefType =
      moore::RefType::get(llvm::cast<moore::UnpackedType>(config.pathIdType));
  return config;
}

CoverageVars MooreInstrumentCoveragePass::getOrCreateCoverageVars(
    moore::ProcedureOp proc, unsigned procIndex,
    const CoverageConfig &config) {
  Operation *key = proc.getOperation();
  if (auto it = coverageVars.find(key); it != coverageVars.end())
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

  Value zero = createZeroConstant(builder, loc, config.pathIdType);
  Value pathIdReg = moore::VariableOp::create(
      builder, loc, config.pathIdRefType,
      builder.getStringAttr(makeName("path_id_reg")), zero);

  pathIdReg.getDefiningOp()->setAttr(
      "pcov.coverage.kind", builder.getStringAttr("path"));
  pathIdReg.getDefiningOp()->setAttr(
      "pcov.coverage.proc_index", builder.getI32IntegerAttr(procIndex));
  pathIdReg.getDefiningOp()->setAttr(
      "pcov.coverage.path_id_width",
      builder.getI32IntegerAttr(config.pathIdWidth));

  CoverageVars vars{pathIdReg};
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

Value MooreInstrumentCoveragePass::accumulateForEdge(
    OpBuilder &builder, Location loc, Block *pred, Block *succ, Value pathSum,
    const CoverageConfig &config,
    const MooreProcedureCFGAnalysis &analysis) const {
  auto it = analysis.edgeWeights.find(BlockEdge{pred, succ});
  assert(it != analysis.edgeWeights.end() && "missing Ball-Larus edge weight");
  uint64_t weightValue = it->second;
  if (weightValue == 0)
    return pathSum;

  Value weightConst = createConstant(builder, loc, config.pathIdType,
                                     APInt(config.pathIdWidth, weightValue));
  return builder.create<moore::AddOp>(loc, config.pathIdType, pathSum,
                                      weightConst);
}

LogicalResult
MooreInstrumentCoveragePass::instrumentModule(moore::SVModuleOp module,
                                              UnitAttr instrumentedAttr) {
  auto infoIt = moduleInfos.find(module.getOperation());
  assert(infoIt != moduleInfos.end() && "expected module info to exist");
  ModuleCoverageInfo &info = infoIt->second;
  info.module = module;
  info.localRegs.clear();
  info.instances.clear();
  info.localWidth = 0;
  info.aggregatedWidth = 0;

  if (!module.getBodyRegion().empty()) {
    Block &body = module.getBodyRegion().front();
    for (moore::InstanceOp inst : body.getOps<moore::InstanceOp>())
      info.instances.push_back(inst);
  }

  MLIRContext *context = module.getContext();
  Builder attrBuilder(context);

  unsigned procIndex = 0;
  for (moore::ProcedureOp proc : module.getOps<moore::ProcedureOp>()) {
    MooreProcedureAnalysisResult analysisResult =
        analyzeMooreProcedure(proc, /*emitDiagnostics=*/true);
    if (analysisResult.fatalError) {
      signalPassFailure();
      return failure();
    }
    if (!analysisResult.analysis)
      continue;

    const MooreProcedureCFGAnalysis &analysis = *analysisResult.analysis;
    CoverageConfig config = buildCoverageConfig(context, analysis.pathIdWidth);
    CoverageVars vars = getOrCreateCoverageVars(proc, procIndex, config);
    if (!vars.isValid())
      continue;

    addCoverageEntryAndArguments(proc, config, analysis);
    rewriteTerminators(proc, config, analysis);
    instrumentExitBlock(proc, procIndex, vars, instrumentedAttr, analysis);

    proc->setAttr("pcov.coverage.instrumented", instrumentedAttr);
    proc->setAttr("pcov.coverage.kind", attrBuilder.getStringAttr("path"));
    proc->setAttr("pcov.coverage.proc_index",
                  attrBuilder.getI32IntegerAttr(procIndex));
    proc->setAttr("pcov.coverage.path_id_width",
                  attrBuilder.getI32IntegerAttr(config.pathIdWidth));

    info.localRegs.push_back(LocalCoverageReg{vars.pathIdReg});
    info.localWidth += config.pathIdWidth;
    ++procIndex;
  }

  return success();
}

unsigned MooreInstrumentCoveragePass::computeAggregatedWidth(
    moore::SVModuleOp module, llvm::DenseMap<Operation *, VisitState> &state,
    const llvm::StringMap<moore::SVModuleOp> &moduleLookup) {
  Operation *key = module.getOperation();
  if (auto it = state.find(key); it != state.end()) {
    if (it->second == VisitState::Visiting) {
      module.emitError(
          "recursive module instantiations are not supported by coverage "
          "instrumentation");
      cycleDetected = true;
      signalPassFailure();
      return 0;
    }
    return moduleInfos.lookup(key).aggregatedWidth;
  }

  state[key] = VisitState::Visiting;
  ModuleCoverageInfo &info = moduleInfos[key];
  unsigned width = info.localWidth;

  for (moore::InstanceOp inst : info.instances) {
    auto childIt = moduleLookup.find(inst.getModuleNameAttr().getValue());
    if (childIt == moduleLookup.end())
      continue;
    width += computeAggregatedWidth(childIt->second, state, moduleLookup);
    if (cycleDetected)
      break;
  }

  info.aggregatedWidth = width;
  state[key] = VisitState::Visited;
  return width;
}

void MooreInstrumentCoveragePass::addCoveragePortToModule(
    moore::SVModuleOp module, moore::IntType coverageType,
    StringAttr coveragePortNameAttr) {
  hw::ModuleType oldType = module.getModuleType();
  SmallVector<hw::ModulePort, 8> ports(oldType.getPorts().begin(),
                                       oldType.getPorts().end());
  ports.push_back({coveragePortNameAttr, coverageType,
                   hw::ModulePort::Direction::Output});
  auto newType = hw::ModuleType::get(module.getContext(), ports);
  module.setModuleTypeAttr(TypeAttr::get(newType));
}

moore::InstanceOp MooreInstrumentCoveragePass::rewriteInstanceWithCoverage(
    moore::InstanceOp inst, moore::IntType coverageType,
    StringAttr coveragePortNameAttr, Value &coverageValue) {
  OpBuilder builder(inst);
  SmallVector<Type> newResultTypes(inst.getResultTypes().begin(),
                                   inst.getResultTypes().end());
  newResultTypes.push_back(coverageType);

  SmallVector<Attribute> outputNameAttrs;
  if (auto names = inst.getOutputNamesAttr())
    llvm::append_range(outputNameAttrs, names.getValue());
  outputNameAttrs.push_back(coveragePortNameAttr);

  OperationState state(inst.getLoc(), inst->getName());
  state.addOperands(inst.getOperands());
  state.addTypes(newResultTypes);
  for (auto attr : inst->getAttrs()) {
    if (attr.getName() == "outputNames")
      continue;
    state.addAttribute(attr.getName(), attr.getValue());
  }
  state.addAttribute("outputNames", builder.getArrayAttr(outputNameAttrs));

  Operation *newOp = builder.create(state);
  auto newInst = cast<moore::InstanceOp>(newOp);

  auto oldResults = inst.getResults();
  auto newResults = newInst.getResults();
  for (auto [oldResult, newResult] :
       llvm::zip(oldResults, newResults.take_front(oldResults.size())))
    oldResult.replaceAllUsesWith(newResult);

  coverageValue = newResults.back();
  inst.erase();
  return newInst;
}

Value MooreInstrumentCoveragePass::buildModuleCoverageValue(
    moore::SVModuleOp module, const ModuleCoverageInfo &info,
    const llvm::DenseMap<Operation *, Value> &instanceCoverageSignals,
    OpBuilder &builder) {
  SmallVector<Value> segments;
  segments.reserve(info.localRegs.size() + info.instances.size());
  Location loc = module.getLoc();

  for (const LocalCoverageReg &reg : info.localRegs) {
    auto refType = llvm::cast<moore::RefType>(reg.reg.getType());
    segments.push_back(
        builder.create<moore::ReadOp>(loc, refType.getNestedType(), reg.reg));
  }

  for (moore::InstanceOp inst : info.instances) {
    if (auto it = instanceCoverageSignals.find(inst.getOperation());
        it != instanceCoverageSignals.end())
      segments.push_back(it->second);
  }

  assert(!segments.empty() && "expected at least one coverage segment");
  if (segments.size() == 1)
    return segments.front();
  return builder.create<moore::ConcatOp>(loc, segments);
}

void MooreInstrumentCoveragePass::addCoverageEntryAndArguments(
    moore::ProcedureOp proc, const CoverageConfig &config,
    const MooreProcedureCFGAnalysis &analysis) {
  Region &body = proc.getBody();
  Block &originalEntry = *analysis.entryBlock;

  auto *coverageEntry = new Block();
  body.getBlocks().insert(body.begin(), coverageEntry);

  OpBuilder builder(proc.getContext());
  builder.setInsertionPointToStart(coverageEntry);
  Location loc = proc.getLoc();

  Value zero = createZeroConstant(builder, loc, config.pathIdType);
  builder.create<cf::BranchOp>(loc, &originalEntry, ValueRange{zero});

  for (Block &block : body) {
    if (&block == coverageEntry)
      continue;
    block.insertArgument(0u, config.pathIdType, loc);
  }
}

void MooreInstrumentCoveragePass::rewriteTerminators(
    moore::ProcedureOp proc, const CoverageConfig &config,
    const MooreProcedureCFGAnalysis &analysis) {
  for (Block &block : proc.getBody()) {
    if (block.getNumArguments() == 0)
      continue;

    Operation *terminator = block.getTerminator();
    OpBuilder builder(terminator);
    Location loc = terminator->getLoc();
    Value pathSum = block.getArgument(0);

    if (auto cond = dyn_cast<cf::CondBranchOp>(terminator)) {
      Block *trueDest = cond.getTrueDest();
      Block *falseDest = cond.getFalseDest();
      Value truePath = accumulateForEdge(builder, loc, &block, trueDest,
                                         pathSum, config, analysis);
      Value falsePath = accumulateForEdge(builder, loc, &block, falseDest,
                                          pathSum, config, analysis);

      SmallVector<Value> trueArgs;
      trueArgs.push_back(truePath);
      trueArgs.append(cond.getTrueOperands().begin(),
                      cond.getTrueOperands().end());

      SmallVector<Value> falseArgs;
      falseArgs.push_back(falsePath);
      falseArgs.append(cond.getFalseOperands().begin(),
                       cond.getFalseOperands().end());

      builder.create<cf::CondBranchOp>(loc, cond.getCondition(), trueDest,
                                       trueArgs, falseDest, falseArgs);
      cond.erase();
      continue;
    }

    if (auto br = dyn_cast<cf::BranchOp>(terminator)) {
      Block *dest = br.getDest();
      Value nextSum = accumulateForEdge(builder, loc, &block, dest, pathSum,
                                        config, analysis);

      SmallVector<Value> operands;
      operands.push_back(nextSum);
      operands.append(br.getDestOperands().begin(),
                      br.getDestOperands().end());

      builder.create<cf::BranchOp>(loc, dest, operands);
      br.erase();
      continue;
    }
  }
}

void MooreInstrumentCoveragePass::instrumentExitBlock(
    moore::ProcedureOp proc, unsigned procIndex, const CoverageVars &vars,
    UnitAttr instrumentedAttr,
    const MooreProcedureCFGAnalysis &analysis) {
  if (!analysis.exitBlock)
    return;

  Operation *terminator = analysis.exitBlock->getTerminator();
  auto returnOp = dyn_cast<moore::ReturnOp>(terminator);
  if (!returnOp) {
    proc.emitError("expected the unique exit block to end with moore.return");
    signalPassFailure();
    return;
  }

  OpBuilder builder(returnOp);
  Location loc = returnOp.getLoc();
  Value pathSum = analysis.exitBlock->getArgument(0);
  moore::NonBlockingAssignOp::create(builder, loc, vars.pathIdReg, pathSum);

  Builder attrBuilder(proc.getContext());
  returnOp->setAttr("pcov.coverage.instrumented", instrumentedAttr);
  returnOp->setAttr("pcov.coverage.proc_index",
                    attrBuilder.getI32IntegerAttr(procIndex));
  returnOp->setAttr("pcov.coverage.kind",
                    attrBuilder.getStringAttr("path"));
}

void MooreInstrumentCoveragePass::runOnOperation() {
  ModuleOp top = getOperation();
  MLIRContext *context = top.getContext();
  UnitAttr instrumentedAttr = UnitAttr::get(context);

  coverageVars.clear();
  moduleInfos.clear();
  cycleDetected = false;

  SmallVector<moore::SVModuleOp, 32> modules;
  llvm::StringMap<moore::SVModuleOp> moduleLookup;
  for (moore::SVModuleOp module : top.getOps<moore::SVModuleOp>()) {
    modules.push_back(module);
    ModuleCoverageInfo &info = moduleInfos[module.getOperation()];
    info.module = module;
    info.localRegs.clear();
    info.instances.clear();
    info.localWidth = 0;
    info.aggregatedWidth = 0;

    if (auto nameAttr = module.getSymNameAttr())
      moduleLookup.try_emplace(nameAttr.getValue(), module);
  }

  for (moore::SVModuleOp module : modules)
    if (failed(instrumentModule(module, instrumentedAttr)))
      return;

  llvm::DenseMap<Operation *, VisitState> visitState;
  for (moore::SVModuleOp module : modules)
    computeAggregatedWidth(module, visitState, moduleLookup);

  if (cycleDetected)
    return;

  Builder attrBuilder(context);
  StringAttr coveragePortNameAttr = attrBuilder.getStringAttr("pcov_path");
  llvm::DenseMap<Operation *, moore::IntType> coverageTypes;

  for (moore::SVModuleOp module : modules) {
    auto &info = moduleInfos[module.getOperation()];
    if (!info.aggregatedWidth)
      continue;

    moore::IntType coverageType =
        moore::IntType::getLogic(context, info.aggregatedWidth);
    coverageTypes[module.getOperation()] = coverageType;
    addCoveragePortToModule(module, coverageType, coveragePortNameAttr);
    module->setAttr("pcov.coverage.total_path_bus_width",
                    attrBuilder.getI64IntegerAttr(info.aggregatedWidth));
  }

  if (coverageTypes.empty())
    return;

  llvm::DenseMap<Operation *, Value> instanceCoverageSignals;
  for (moore::SVModuleOp module : modules) {
    auto &info = moduleInfos[module.getOperation()];
    if (info.instances.empty())
      continue;

    for (moore::InstanceOp &inst : info.instances) {
      auto childIt = moduleLookup.find(inst.getModuleNameAttr().getValue());
      if (childIt == moduleLookup.end())
        continue;

      auto &childInfo = moduleInfos[childIt->second.getOperation()];
      if (!childInfo.aggregatedWidth)
        continue;

      moore::IntType coverageType =
          coverageTypes.lookup(childIt->second.getOperation());
      Value coverageValue;
      moore::InstanceOp newInst = rewriteInstanceWithCoverage(
          inst, coverageType, coveragePortNameAttr, coverageValue);
      inst = newInst;
      instanceCoverageSignals[newInst.getOperation()] = coverageValue;
    }
  }

  for (moore::SVModuleOp module : modules) {
    auto &info = moduleInfos[module.getOperation()];
    if (!info.aggregatedWidth)
      continue;

    auto outputOp = module.getOutputOp();
    if (!outputOp) {
      module.emitError("expected module to terminate with moore.output");
      signalPassFailure();
      return;
    }

    OpBuilder builder(outputOp);
    builder.setInsertionPoint(outputOp);
    Value coverageValue =
        buildModuleCoverageValue(module, info, instanceCoverageSignals,
                                 builder);

    SmallVector<Value> newOutputs(outputOp.getOperands().begin(),
                                  outputOp.getOperands().end());
    newOutputs.push_back(coverageValue);
    outputOp->setOperands(newOutputs);
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
