//===- MooreInstrumentCoverageSum.cpp - Coverage sum instrumentation ------===//
//
// Part of the circt-cf project.
//
//===----------------------------------------------------------------------===//

#include "circt-cf/Instrumentation/Passes.h"

#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include <cstdlib>
#include <optional>

using namespace mlir;
using namespace circt;

namespace circt::pcov {
#define GEN_PASS_DEF_MOOREINSTRUMENTCOVERAGESUM
#include "circt-cf/Instrumentation/InstrumentationPasses.h.inc"
} // namespace circt::pcov

namespace circt::pcov {
namespace {

static constexpr StringLiteral covCountKind = "path_covcount";
static constexpr StringLiteral covCountPointAttr =
    "pcov.coverage.point_count";
static constexpr StringLiteral covSumPortName = "pcov_covsum";

enum class VisitState { Visiting, Visited };

struct ProcCountInfo {
  moore::VariableOp countVar;
  unsigned pointCount = 0;
};

struct ModuleCovSumInfo {
  moore::SVModuleOp module;
  SmallVector<moore::InstanceOp> instances;
  SmallVector<ProcCountInfo> localCounts;
  unsigned localPointCount = 0;
  unsigned aggregatePointCount = 0;
  bool needsCovSum = false;
  moore::IntType covSumType;
};

class MooreInstrumentCoverageSumPass
    : public impl::MooreInstrumentCoverageSumBase<
          MooreInstrumentCoverageSumPass> {
public:
  void runOnOperation() override;

private:
  void gatherModuleInfo(ModuleOp top);
  unsigned computeAggregatePoints(
      moore::SVModuleOp module, llvm::DenseMap<Operation *, VisitState> &state,
      const llvm::StringMap<moore::SVModuleOp> &moduleLookup);
  void addCovSumPortToModule(moore::SVModuleOp module,
                             moore::IntType covSumType);
  Value createZeroConstant(OpBuilder &builder, Location loc,
                           moore::IntType type) const;
  Value widenValue(OpBuilder &builder, Location loc, Value input,
                   moore::IntType targetType) const;
  LogicalResult buildModuleCovSumValue(
      ModuleCovSumInfo &info,
      const llvm::DenseMap<Operation *, Value> &instanceCovSignals);
  void rewriteInstancesWithCovSum(
      ModuleCovSumInfo &info,
      const llvm::StringMap<moore::SVModuleOp> &moduleLookup,
      llvm::DenseMap<Operation *, Value> &instanceCovSignals);

  llvm::DenseMap<Operation *, ModuleCovSumInfo> moduleInfos;
  SmallVector<moore::SVModuleOp, 32> modules;
  StringAttr covSumPortNameAttr;
};

void MooreInstrumentCoverageSumPass::gatherModuleInfo(ModuleOp top) {
  moduleInfos.clear();
  modules.clear();

  for (moore::SVModuleOp module : top.getOps<moore::SVModuleOp>()) {
    modules.push_back(module);
    ModuleCovSumInfo &info = moduleInfos[module.getOperation()];
    info.module = module;

    if (!module.getBodyRegion().empty()) {
      Block &body = module.getBodyRegion().front();
      for (moore::InstanceOp inst : body.getOps<moore::InstanceOp>())
        info.instances.push_back(inst);
    }

    for (moore::VariableOp var : module.getOps<moore::VariableOp>()) {
      auto kindAttr = var->getAttrOfType<StringAttr>("pcov.coverage.kind");
      if (!kindAttr || kindAttr.getValue() != covCountKind)
        continue;
      unsigned points = 0;
      if (auto pointAttr = var->getAttrOfType<IntegerAttr>(covCountPointAttr))
        points = pointAttr.getInt();
      if (points == 0) {
        if (auto refType = llvm::dyn_cast<moore::RefType>(var.getType()))
          if (auto intType =
                  llvm::dyn_cast<moore::IntType>(refType.getNestedType()))
            points = intType.getWidth();
      }
      if (points == 0)
        points = 1;
      info.localCounts.push_back(ProcCountInfo{var, points});
      info.localPointCount += points;
    }
  }
}

unsigned MooreInstrumentCoverageSumPass::computeAggregatePoints(
    moore::SVModuleOp module, llvm::DenseMap<Operation *, VisitState> &state,
    const llvm::StringMap<moore::SVModuleOp> &moduleLookup) {
  Operation *key = module.getOperation();
  if (auto it = state.find(key); it != state.end()) {
    if (it->second == VisitState::Visiting) {
      module.emitError(
          "recursive module instantiations are not supported by covsum "
          "instrumentation");
      signalPassFailure();
      return 0;
    }
    return moduleInfos.lookup(key).aggregatePointCount;
  }

  state[key] = VisitState::Visiting;
  ModuleCovSumInfo &info = moduleInfos[key];

  uint64_t total = info.localPointCount;
  for (moore::InstanceOp inst : info.instances) {
    auto childIt = moduleLookup.find(inst.getModuleNameAttr().getValue());
    if (childIt == moduleLookup.end())
      continue;
    total += computeAggregatePoints(childIt->second, state, moduleLookup);
  }

  info.aggregatePointCount = static_cast<unsigned>(total);
  info.needsCovSum = total != 0;
  state[key] = VisitState::Visited;
  return info.aggregatePointCount;
}

void MooreInstrumentCoverageSumPass::addCovSumPortToModule(
    moore::SVModuleOp module, moore::IntType covSumType) {
  hw::ModuleType oldType = module.getModuleType();
  SmallVector<hw::ModulePort> ports(oldType.getPorts().begin(),
                                    oldType.getPorts().end());
  ports.push_back(
      {covSumPortNameAttr, covSumType, hw::ModulePort::Direction::Output});
  auto newType = hw::ModuleType::get(module.getContext(), ports);
  module.setModuleTypeAttr(TypeAttr::get(newType));
}

Value MooreInstrumentCoverageSumPass::createZeroConstant(
    OpBuilder &builder, Location loc, moore::IntType type) const {
  return moore::ConstantOp::create(builder, loc, type,
                                   llvm::APInt(type.getWidth(), 0));
}

Value MooreInstrumentCoverageSumPass::widenValue(OpBuilder &builder,
                                                 Location loc, Value input,
                                                 moore::IntType targetType) const {
  auto srcType = llvm::dyn_cast<moore::IntType>(input.getType());
  if (!srcType || srcType.getWidth() == targetType.getWidth())
    return input;
  if (srcType.getWidth() > targetType.getWidth()) {
    builder.getBlock()->getParentOp()->emitError(
        "covsum source wider than target");
    return input;
  }
  return moore::ZExtOp::create(builder, loc, targetType, input);
}

void MooreInstrumentCoverageSumPass::rewriteInstancesWithCovSum(
    ModuleCovSumInfo &info,
    const llvm::StringMap<moore::SVModuleOp> &moduleLookup,
    llvm::DenseMap<Operation *, Value> &instanceCovSignals) {
  if (!info.needsCovSum)
    return;

  for (moore::InstanceOp &inst : info.instances) {
    auto childIt = moduleLookup.find(inst.getModuleNameAttr().getValue());
    if (childIt == moduleLookup.end())
      continue;
    moore::SVModuleOp childModule = childIt->second;
    ModuleCovSumInfo &childInfo = moduleInfos[childModule.getOperation()];
    if (!childInfo.needsCovSum)
      continue;

    OpBuilder builder(inst);
    SmallVector<Type> newResultTypes(inst.getResultTypes().begin(),
                                     inst.getResultTypes().end());
    newResultTypes.push_back(childInfo.covSumType);

    SmallVector<Attribute> outputNameAttrs;
    if (auto names = inst.getOutputNamesAttr())
      llvm::append_range(outputNameAttrs, names.getValue());
    outputNameAttrs.push_back(covSumPortNameAttr);

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

    Value covValue = newResults.back();
    instanceCovSignals[newInst.getOperation()] = covValue;
    inst.erase();
    inst = newInst;
  }
}

LogicalResult MooreInstrumentCoverageSumPass::buildModuleCovSumValue(
    ModuleCovSumInfo &info,
    const llvm::DenseMap<Operation *, Value> &instanceCovSignals) {
  moore::SVModuleOp module = info.module;
  auto outputOp = module.getOutputOp();
  if (!outputOp) {
    module.emitError("expected module to terminate with moore.output");
    return failure();
  }

  OpBuilder builder(outputOp);
  builder.setInsertionPoint(outputOp);
  Value total =
      createZeroConstant(builder, module.getLoc(), info.covSumType);

  for (ProcCountInfo &count : info.localCounts) {
    auto refType = llvm::cast<moore::RefType>(count.countVar.getType());
    auto intType = llvm::cast<moore::IntType>(refType.getNestedType());
    Value read =
        moore::ReadOp::create(builder, module.getLoc(), intType, count.countVar);
    Value widened = widenValue(builder, module.getLoc(), read, info.covSumType);
    total = moore::AddOp::create(builder, module.getLoc(), info.covSumType,
                                 total, widened);
  }

  for (moore::InstanceOp inst : info.instances) {
    auto it = instanceCovSignals.find(inst.getOperation());
    if (it == instanceCovSignals.end())
      continue;
    Value childVal = it->second;
    Value widened =
        widenValue(builder, module.getLoc(), childVal, info.covSumType);
    total = moore::AddOp::create(builder, module.getLoc(), info.covSumType,
                                 total, widened);
  }

  SmallVector<Value> newOutputs(outputOp.getOperands().begin(),
                                outputOp.getOperands().end());
  newOutputs.push_back(total);
  outputOp->setOperands(newOutputs);
  return success();
}

void MooreInstrumentCoverageSumPass::runOnOperation() {
  ModuleOp top = getOperation();
  MLIRContext *context = top.getContext();
  covSumPortNameAttr = StringAttr::get(context, covSumPortName);

  gatherModuleInfo(top);
  if (modules.empty())
    return;

  llvm::StringMap<moore::SVModuleOp> moduleLookup;
  for (moore::SVModuleOp module : modules)
    if (auto nameAttr = module.getSymNameAttr())
      moduleLookup.try_emplace(nameAttr.getValue(), module);

  llvm::DenseMap<Operation *, VisitState> visitState;
  for (moore::SVModuleOp module : modules)
    computeAggregatePoints(module, visitState, moduleLookup);

  llvm::DenseMap<Operation *, Value> instanceCovSignals;

  for (moore::SVModuleOp module : modules) {
    ModuleCovSumInfo &info = moduleInfos[module.getOperation()];
    if (!info.needsCovSum)
      continue;

    unsigned width = std::max(
        1u, llvm::Log2_64_Ceil(static_cast<uint64_t>(info.aggregatePointCount) +
                               1));
    info.covSumType = moore::IntType::getLogic(context, width);
    addCovSumPortToModule(module, info.covSumType);
  }

  for (moore::SVModuleOp module : modules) {
    ModuleCovSumInfo &info = moduleInfos[module.getOperation()];
    rewriteInstancesWithCovSum(info, moduleLookup, instanceCovSignals);
  }

  for (moore::SVModuleOp module : modules) {
    ModuleCovSumInfo &info = moduleInfos[module.getOperation()];
    if (!info.needsCovSum)
      continue;
    if (failed(buildModuleCovSumValue(info, instanceCovSignals))) {
      signalPassFailure();
      return;
    }
  }
}

} // namespace

std::unique_ptr<mlir::Pass> createMooreInstrumentCoverageSumPass() {
  return std::make_unique<MooreInstrumentCoverageSumPass>();
}

void registerMooreInstrumentCoverageSumPass() {
  mlir::PassRegistration<MooreInstrumentCoverageSumPass>();
}

} // namespace circt::pcov
