//===- MooreInstrumentPathBitmap.cpp - Path bitmap instrumentation --------===//
//
// Part of the circt-cf project.
//
//===----------------------------------------------------------------------===//

#include "circt-cf/Instrumentation/MooreProcedureAnalysis.h"
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
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <optional>

using namespace mlir;
using namespace circt;

namespace circt::pcov {
#define GEN_PASS_DEF_MOOREINSTRUMENTPATHBITMAP
#include "circt-cf/Instrumentation/InstrumentationPasses.h.inc"
} // namespace circt::pcov

namespace circt::pcov {
namespace {

static constexpr StringLiteral metaResetPortName = "pcov_meta_reset";
static constexpr StringLiteral bitmapKind = "path_bitmap";
static constexpr StringLiteral metaResetPortIndexAttr =
    "pcov.meta_reset.port_index";

enum class VisitState { Visiting, Visited };

static bool isDebugEnabled() {
  static bool enabled = std::getenv("PCOV_BITMAP_DEBUG") != nullptr;
  return enabled;
}

static void debugLog(StringRef message) {
  if (!isDebugEnabled())
    return;
  llvm::errs() << "[pcov-bitmap] " << message << "\n";
}

static std::string getModuleName(moore::SVModuleOp module) {
  if (auto nameAttr = module.getSymNameAttr())
    return nameAttr.getValue().str();
  return "<anonymous>";
}

static Block *getModuleBodyBlock(moore::SVModuleOp module) {
  if (module.getBodyRegion().empty())
    return nullptr;
  return &module.getBodyRegion().front();
}

struct ProcBitmapInfo {
  moore::ProcedureOp proc;
  moore::VariableOp pathIdVar;
  moore::IntType pathIdType;
  moore::RefType pathIdRefType;
  unsigned procIndex = 0;
};

struct ModuleBitmapInfo {
  moore::SVModuleOp module;
  SmallVector<moore::InstanceOp> instances;
  SmallVector<ProcBitmapInfo> procs;
  bool hasLocalInstrumentation = false;
  bool needsMetaReset = false;
  Value metaResetPortValue;
  Value metaResetNet;
};

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

class MooreInstrumentPathBitmapPass
    : public impl::MooreInstrumentPathBitmapBase<
          MooreInstrumentPathBitmapPass> {
public:
  void runOnOperation() override;

private:
  void gatherModuleInfo(ModuleOp top);
  bool computeNeedsMetaReset(
      moore::SVModuleOp module, llvm::DenseMap<Operation *, VisitState> &state,
      const llvm::StringMap<moore::SVModuleOp> &moduleLookup);
  Value ensureMetaResetPort(ModuleBitmapInfo &info);
  Value getOrCreateMetaResetNet(ModuleBitmapInfo &info);
  LogicalResult buildBitmapController(ModuleBitmapInfo &moduleInfo,
                                      const ProcBitmapInfo &procInfo);
  LogicalResult instrumentModule(ModuleBitmapInfo &info);
  void rewriteInstances(ModuleBitmapInfo &info,
                        const llvm::StringMap<moore::SVModuleOp> &moduleLookup);
  Value createConstant(OpBuilder &builder, Location loc, moore::IntType type,
                       const llvm::APInt &value) const;
  Value createZeroConstant(OpBuilder &builder, Location loc,
                           moore::IntType type) const;

  llvm::DenseMap<Operation *, ModuleBitmapInfo> moduleInfos;
  SmallVector<moore::SVModuleOp, 32> modules;
  moore::IntType metaResetType;
  moore::RefType metaResetRefType;
  StringAttr metaResetNameAttr;
};

void MooreInstrumentPathBitmapPass::gatherModuleInfo(ModuleOp top) {
  moduleInfos.clear();
  modules.clear();

  for (moore::SVModuleOp module : top.getOps<moore::SVModuleOp>()) {
    modules.push_back(module);
    ModuleBitmapInfo &info = moduleInfos[module.getOperation()];
    info.module = module;

    if (!module.getBodyRegion().empty()) {
      Block &body = module.getBodyRegion().front();
      for (moore::InstanceOp inst : body.getOps<moore::InstanceOp>())
        info.instances.push_back(inst);
    }

    llvm::DenseMap<unsigned, moore::ProcedureOp> procsByIndex;
    for (moore::ProcedureOp proc : module.getOps<moore::ProcedureOp>()) {
      if (auto indexAttr =
              proc->getAttrOfType<IntegerAttr>("pcov.coverage.proc_index"))
        procsByIndex.try_emplace(indexAttr.getInt(), proc);
    }

    for (moore::VariableOp var : module.getOps<moore::VariableOp>()) {
      auto kindAttr = var->getAttrOfType<StringAttr>("pcov.coverage.kind");
      if (!kindAttr || kindAttr.getValue() != "path")
        continue;
      auto indexAttr =
          var->getAttrOfType<IntegerAttr>("pcov.coverage.proc_index");
      if (!indexAttr)
        continue;
      auto procIt = procsByIndex.find(indexAttr.getInt());
      if (procIt == procsByIndex.end())
        continue;

      auto refType = llvm::cast<moore::RefType>(var.getType());
      auto pathIdType =
          llvm::dyn_cast<moore::IntType>(refType.getNestedType());
      if (!pathIdType)
        continue;

      ProcBitmapInfo procInfo;
      procInfo.proc = procIt->second;
      procInfo.pathIdVar = var;
      procInfo.pathIdType = pathIdType;
      procInfo.pathIdRefType = refType;
      procInfo.procIndex = indexAttr.getInt();
      info.procs.push_back(procInfo);
    }

    info.hasLocalInstrumentation = !info.procs.empty();
  }
}

bool MooreInstrumentPathBitmapPass::computeNeedsMetaReset(
    moore::SVModuleOp module, llvm::DenseMap<Operation *, VisitState> &state,
    const llvm::StringMap<moore::SVModuleOp> &moduleLookup) {
  Operation *key = module.getOperation();
  if (auto it = state.find(key); it != state.end()) {
    if (it->second == VisitState::Visiting) {
      module.emitError(
          "recursive module instantiations are not supported by bitmap "
          "instrumentation");
      signalPassFailure();
      return false;
    }
    return moduleInfos.lookup(key).needsMetaReset;
  }

  state[key] = VisitState::Visiting;
  ModuleBitmapInfo &info = moduleInfos[key];
  bool required = info.hasLocalInstrumentation;

  for (moore::InstanceOp inst : info.instances) {
    auto childIt = moduleLookup.find(inst.getModuleNameAttr().getValue());
    if (childIt == moduleLookup.end())
      continue;
    bool childRequired =
        computeNeedsMetaReset(childIt->second, state, moduleLookup);
    required |= childRequired;
  }

  info.needsMetaReset = required;
  state[key] = VisitState::Visited;
  return required;
}

Value MooreInstrumentPathBitmapPass::ensureMetaResetPort(ModuleBitmapInfo &info) {
  if (isDebugEnabled())
    debugLog((Twine("  ensureMetaResetPort for module ") +
              StringRef(getModuleName(info.module)))
                 .str());

  moore::SVModuleOp module = info.module;
  Block *body = getModuleBodyBlock(module);
  if (!body) {
    module.emitError(
        "module lacks a body block; cannot add meta reset input");
    signalPassFailure();
    return {};
  }

  if (info.metaResetPortValue)
    return info.metaResetPortValue;

  if (auto indexAttr =
          module->getAttrOfType<IntegerAttr>(metaResetPortIndexAttr)) {
    unsigned index = indexAttr.getInt();
    if (index >= body->getNumArguments()) {
      module.emitError("stored meta reset argument index (")
          << index << ") exceeds block argument count ("
          << body->getNumArguments() << ")";
      signalPassFailure();
      return {};
    }
    info.metaResetPortValue = body->getArgument(index);
    if (isDebugEnabled())
      debugLog((Twine("    Reusing previously recorded meta reset port ") +
                Twine(index) + " in module " + StringRef(getModuleName(module)))
                   .str());
    return info.metaResetPortValue;
  }

  hw::ModuleType type = module.getModuleType();
  unsigned expectedInputs = type.getNumInputs();
  if (isDebugEnabled())
    debugLog((Twine("    module ") + StringRef(getModuleName(module)) +
              " has " + Twine(body->getNumArguments()) +
              " block args, expects " + Twine(expectedInputs))
                 .str());
  if (body->getNumArguments() != expectedInputs) {
    module.emitError("mismatched number of block arguments (")
        << body->getNumArguments() << ") vs. module inputs (" << expectedInputs
        << ") while wiring meta reset";
    signalPassFailure();
    return {};
  }

  SmallVector<hw::ModulePort> ports(type.getPorts().begin(),
                                    type.getPorts().end());
  ports.push_back({metaResetNameAttr, metaResetType,
                   hw::ModulePort::Direction::Input});
  hw::ModuleType newType = hw::ModuleType::get(module.getContext(), ports);
  module.setModuleTypeAttr(TypeAttr::get(newType));

  Value arg = body->addArgument(metaResetType, module.getLoc());
  info.metaResetPortValue = arg;
  module->setAttr(metaResetPortIndexAttr,
                  IntegerAttr::get(IntegerType::get(module.getContext(), 32),
                                   body->getNumArguments() - 1));
  if (isDebugEnabled())
    debugLog((Twine("    Added meta reset port to module ") +
              StringRef(getModuleName(module)) + ", new index " +
              Twine(body->getNumArguments() - 1))
                 .str());
  return arg;
}

Value MooreInstrumentPathBitmapPass::getOrCreateMetaResetNet(
    ModuleBitmapInfo &info) {
  if (isDebugEnabled())
    debugLog((Twine("    getOrCreateMetaResetNet for module ") +
              StringRef(getModuleName(info.module)))
                 .str());
  if (info.metaResetNet) {
    if (isDebugEnabled())
      debugLog((Twine("    Reusing cached meta reset net in module ") +
                StringRef(getModuleName(info.module)))
                   .str());
    return info.metaResetNet;
  }

  moore::SVModuleOp module = info.module;
  Block *body = getModuleBodyBlock(module);
  if (!body) {
    module.emitError(
        "module lacks a body block; cannot create meta reset net");
    signalPassFailure();
    return {};
  }

  for (moore::NetOp net : body->getOps<moore::NetOp>()) {
    if (auto nameAttr = net.getNameAttr()) {
      if (nameAttr == metaResetNameAttr) {
        info.metaResetNet = net.getResult();
        if (isDebugEnabled())
          debugLog((Twine("    Found existing meta reset net in module ") +
                    StringRef(getModuleName(module)))
                       .str());
        break;
      }
    }
  }
  if (info.metaResetNet)
    return info.metaResetNet;

  OpBuilder builder(module);
  builder.setInsertionPointToStart(body);
  auto kindAttr =
      moore::NetKindAttr::get(module.getContext(), moore::NetKind::Wire);
  auto net = moore::NetOp::create(builder, module.getLoc(), metaResetRefType,
                                  metaResetNameAttr, kindAttr, Value());
  info.metaResetNet = net.getResult();

  auto outputOp = module.getOutputOp();
  if (!outputOp) {
    module.emitError("expected module to terminate with moore.output");
    signalPassFailure();
    return {};
  }
  OpBuilder assignBuilder(outputOp);
  bool alreadyAssigned = false;
  for (moore::ContinuousAssignOp assign :
       body->getOps<moore::ContinuousAssignOp>()) {
    if (assign.getDst() == info.metaResetNet &&
        assign.getSrc() == info.metaResetPortValue) {
      alreadyAssigned = true;
      break;
    }
  }
  if (!alreadyAssigned)
    moore::ContinuousAssignOp::create(assignBuilder, module.getLoc(),
                                      info.metaResetNet,
                                      info.metaResetPortValue);
  if (isDebugEnabled())
    debugLog((Twine("    Created meta reset net in module ") +
              StringRef(getModuleName(module)))
                 .str());
  return info.metaResetNet;
}

Value MooreInstrumentPathBitmapPass::createConstant(OpBuilder &builder,
                                                    Location loc,
                                                    moore::IntType type,
                                                    const llvm::APInt &value) const {
  return moore::ConstantOp::create(builder, loc, type, value);
}

Value MooreInstrumentPathBitmapPass::createZeroConstant(OpBuilder &builder,
                                                        Location loc,
                                                        moore::IntType type) const {
  return createConstant(builder, loc, type, llvm::APInt(type.getWidth(), 0));
}

LogicalResult MooreInstrumentPathBitmapPass::buildBitmapController(
    ModuleBitmapInfo &moduleInfo, const ProcBitmapInfo &procInfo) {
  moore::ProcedureOp sourceProc = procInfo.proc;
  moore::VariableOp pathIdVar = procInfo.pathIdVar;
  MLIRContext *context = sourceProc.getContext();
  auto analysisResult = analyzeMooreProcedure(sourceProc,
                                              /*emitDiagnostics=*/false);
  if (!analysisResult.analysis)
    return success();

  Block *entryBlock = analysisResult.analysis->entryBlock;
  if (!entryBlock || entryBlock->empty()) {
    sourceProc.emitError("expected entry block with wait_event");
    return failure();
  }
  auto waitEvent = dyn_cast<moore::WaitEventOp>(&entryBlock->front());
  if (!waitEvent) {
    sourceProc.emitError("expected wait_event at start of entry block");
    return failure();
  }

  Value clockSignal;
  moore::Edge clockEdge = moore::Edge::PosEdge;
  for (Operation &op : waitEvent.getBody().front()) {
    auto detect = dyn_cast<moore::DetectEventOp>(&op);
    if (!detect)
      continue;
    if (!isEdgeTriggeredClock(detect))
      continue;
    Value input = detect.getInput();
    if (auto read = input.getDefiningOp<moore::ReadOp>()) {
      clockSignal = read.getInput();
      clockEdge = detect.getEdge();
      break;
    }
  }

  if (!clockSignal) {
    sourceProc.emitError(
        "unable to locate clock detect_event for bitmap controller");
    return failure();
  }
  if (!moduleInfo.metaResetNet) {
    sourceProc.emitError("meta reset net missing for bitmap controller");
    return failure();
  }

  Value bitmapZero;
  {
    if (isDebugEnabled())
      debugLog((Twine("    buildBitmapController module ") +
                StringRef(getModuleName(moduleInfo.module)) + " proc " +
                Twine(procInfo.procIndex))
                   .str());
    OpBuilder builder(moduleInfo.module);
    builder.setInsertionPointAfter(pathIdVar.getOperation());
    bitmapZero = createZeroConstant(builder, sourceProc.getLoc(),
                                    procInfo.pathIdType);
    auto refType = procInfo.pathIdRefType;
    auto bitmapName =
        (Twine("cov_proc") + Twine(procInfo.procIndex) + "_bitmap_reg").str();
    Value bitmapReg = moore::VariableOp::create(
        builder, sourceProc.getLoc(), refType,
        builder.getStringAttr(bitmapName), bitmapZero);
    bitmapReg.getDefiningOp()->setAttr("pcov.coverage.kind",
                                       builder.getStringAttr(bitmapKind));
    bitmapReg.getDefiningOp()->setAttr(
        "pcov.coverage.proc_index",
        builder.getI32IntegerAttr(static_cast<int32_t>(procInfo.procIndex)));
    bitmapReg.getDefiningOp()->setAttr(
        "pcov.coverage.path_id_width",
        builder.getI32IntegerAttr(procInfo.pathIdType.getWidth()));

    OpBuilder procBuilder(moduleInfo.module);
    procBuilder.setInsertionPointAfter(sourceProc);
    auto procOp = moore::ProcedureOp::create(procBuilder, sourceProc.getLoc(),
                                             moore::ProcedureKind::Always);
    Region &procRegion = procOp.getBody();
    Block *controllerEntry = &procRegion.emplaceBlock();
    Block *resetBlock = &procRegion.emplaceBlock();
    Block *updateBlock = &procRegion.emplaceBlock();
    Block *exitBlock = &procRegion.emplaceBlock();

    {
      OpBuilder entryBuilder(controllerEntry, controllerEntry->begin());
      auto wait =
          moore::WaitEventOp::create(entryBuilder, procOp.getLoc());
      Block *waitBody = &wait.getBody().emplaceBlock();
      OpBuilder waitBuilder(waitBody, waitBody->begin());
      auto clockRefType = llvm::cast<moore::RefType>(clockSignal.getType());
      auto clockValueType = clockRefType.getNestedType();
      auto clockRead = moore::ReadOp::create(waitBuilder, procOp.getLoc(),
                                             clockValueType, clockSignal);
      auto edgeAttr = moore::EdgeAttr::get(context, clockEdge);
      moore::DetectEventOp::create(waitBuilder, procOp.getLoc(), edgeAttr,
                                   clockRead, Value());

      auto metaRead = moore::ReadOp::create(waitBuilder, procOp.getLoc(),
                                            metaResetType,
                                            moduleInfo.metaResetNet);
      auto posEdgeAttr =
          moore::EdgeAttr::get(context, moore::Edge::PosEdge);
      moore::DetectEventOp::create(waitBuilder, procOp.getLoc(), posEdgeAttr,
                                   metaRead, Value());

      auto entryBuilderAfter = OpBuilder::atBlockEnd(controllerEntry);
      Value metaSample = moore::ReadOp::create(entryBuilderAfter,
                                               procOp.getLoc(), metaResetType,
                                               moduleInfo.metaResetNet);
      Value metaOne = createConstant(entryBuilderAfter, procOp.getLoc(),
                                     metaResetType, llvm::APInt(1, 1));
      Value metaEq = moore::EqOp::create(entryBuilderAfter, procOp.getLoc(),
                                         metaSample, metaOne);
      Value metaCond =
          moore::ToBuiltinBoolOp::create(entryBuilderAfter, procOp.getLoc(),
                                         metaEq);
      cf::CondBranchOp::create(entryBuilderAfter, procOp.getLoc(), metaCond,
                               resetBlock, ValueRange(), updateBlock,
                               ValueRange());
    }

    {
      OpBuilder resetBuilder(resetBlock, resetBlock->begin());
      moore::NonBlockingAssignOp::create(resetBuilder, procOp.getLoc(),
                                         bitmapReg, bitmapZero);
      cf::BranchOp::create(resetBuilder, procOp.getLoc(), exitBlock);
    }

    {
      OpBuilder updateBuilder(updateBlock, updateBlock->begin());
      Value currentBitmap =
          moore::ReadOp::create(updateBuilder, procOp.getLoc(),
                                procInfo.pathIdType, bitmapReg);
      Value pathId = moore::ReadOp::create(updateBuilder, procOp.getLoc(),
                                           procInfo.pathIdType,
                                           pathIdVar.getResult());
      Value one = createConstant(updateBuilder, procOp.getLoc(),
                                 procInfo.pathIdType,
                                 llvm::APInt(procInfo.pathIdType.getWidth(), 1));
      Value shifted = moore::ShlOp::create(updateBuilder, procOp.getLoc(),
                                           procInfo.pathIdType, one, pathId);
      Value nextBitmap =
          moore::OrOp::create(updateBuilder, procOp.getLoc(),
                              procInfo.pathIdType, currentBitmap, shifted);
      moore::NonBlockingAssignOp::create(updateBuilder, procOp.getLoc(),
                                         bitmapReg, nextBitmap);
      cf::BranchOp::create(updateBuilder, procOp.getLoc(), exitBlock);
    }

    OpBuilder exitBuilder(exitBlock, exitBlock->begin());
    moore::ReturnOp::create(exitBuilder, procOp.getLoc());

    procOp->setAttr("pcov.coverage.kind",
                    procBuilder.getStringAttr(bitmapKind));
    procOp->setAttr(
        "pcov.coverage.proc_index",
        procBuilder.getI32IntegerAttr(static_cast<int32_t>(procInfo.procIndex)));
    procOp->setAttr("pcov.coverage.path_id_width",
                    procBuilder.getI32IntegerAttr(procInfo.pathIdType.getWidth()));
  }

  return success();
}

LogicalResult
MooreInstrumentPathBitmapPass::instrumentModule(ModuleBitmapInfo &info) {
  if (!info.needsMetaReset) {
    if (isDebugEnabled())
      debugLog((Twine("Skipping module ") + StringRef(getModuleName(info.module)) +
                " (no meta reset required)")
                   .str());
    return success();
  }

  std::string moduleName = getModuleName(info.module);
  if (isDebugEnabled())
    debugLog((Twine("Instrumenting module ") + StringRef(moduleName)).str());

  Value port = ensureMetaResetPort(info);
  if (!port)
    return failure();
  if (isDebugEnabled())
    debugLog((Twine("  Module ") + StringRef(moduleName) +
              " meta reset port value ready")
                 .str());

  if (!info.hasLocalInstrumentation) {
    if (isDebugEnabled())
      debugLog((Twine("Module ") + StringRef(moduleName) +
                " has no local coverage registers; only forwarding meta reset")
                   .str());
    return success();
  }

  Value metaNet = getOrCreateMetaResetNet(info);
  if (!metaNet)
    return failure();
  if (isDebugEnabled())
    debugLog((Twine("  Module ") + StringRef(moduleName) +
              " meta reset net ready")
                 .str());

  for (const ProcBitmapInfo &procInfo : info.procs) {
    if (isDebugEnabled())
      debugLog((Twine("  Building bitmap controller for proc ") +
                Twine(procInfo.procIndex) + " in module " + StringRef(moduleName))
                   .str());
    if (failed(buildBitmapController(info, procInfo)))
      return failure();
  }
  return success();
}

void MooreInstrumentPathBitmapPass::rewriteInstances(
    ModuleBitmapInfo &info,
    const llvm::StringMap<moore::SVModuleOp> &moduleLookup) {
  if (!info.needsMetaReset)
    return;

  Value metaResetValue = info.metaResetPortValue;
  if (!metaResetValue)
    return;

  for (moore::InstanceOp &inst : info.instances) {
    auto childIt = moduleLookup.find(inst.getModuleNameAttr().getValue());
    if (childIt == moduleLookup.end())
      continue;
    moore::SVModuleOp childModule = childIt->second;
    ModuleBitmapInfo &childInfo = moduleInfos[childModule.getOperation()];
    if (!childInfo.needsMetaReset)
      continue;

    bool alreadyWired = false;
    if (auto names = inst.getInputNamesAttr()) {
      for (Attribute attr : names.getValue())
        if (auto strAttr = dyn_cast<StringAttr>(attr))
          if (strAttr == metaResetNameAttr)
            alreadyWired = true;
    }
    if (alreadyWired)
      continue;

    OpBuilder builder(inst);
    SmallVector<Value> newInputs(inst.getInputs().begin(),
                                 inst.getInputs().end());
    newInputs.push_back(metaResetValue);

    SmallVector<Attribute> inputNames;
    if (auto names = inst.getInputNamesAttr())
      llvm::append_range(inputNames, names.getValue());
    inputNames.push_back(metaResetNameAttr);

    OperationState state(inst.getLoc(), inst->getName());
    state.addOperands(newInputs);
    for (auto attr : inst->getAttrs()) {
      if (attr.getName() == "inputNames")
        continue;
      state.addAttribute(attr.getName(), attr.getValue());
    }
    state.addAttribute("inputNames", builder.getArrayAttr(inputNames));
    state.addTypes(inst.getResultTypes());

    Operation *newOp = builder.create(state);
    auto newInst = cast<moore::InstanceOp>(newOp);
    for (auto [oldRes, newRes] :
         llvm::zip(inst.getResults(), newInst.getResults()))
      oldRes.replaceAllUsesWith(newRes);
    inst.erase();
    inst = newInst;
  }
}

void MooreInstrumentPathBitmapPass::runOnOperation() {
  ModuleOp top = getOperation();
  MLIRContext *context = top.getContext();
  metaResetType = moore::IntType::getLogic(context, 1);
  metaResetRefType =
      moore::RefType::get(llvm::cast<moore::UnpackedType>(metaResetType));
  metaResetNameAttr = StringAttr::get(context, metaResetPortName);

  gatherModuleInfo(top);
  if (modules.empty())
    return;

  llvm::StringMap<moore::SVModuleOp> moduleLookup;
  for (moore::SVModuleOp module : modules)
    if (auto nameAttr = module.getSymNameAttr())
      moduleLookup.try_emplace(nameAttr.getValue(), module);

  llvm::DenseMap<Operation *, VisitState> visitState;
  for (moore::SVModuleOp module : modules)
    computeNeedsMetaReset(module, visitState, moduleLookup);

  for (moore::SVModuleOp module : modules) {
    ModuleBitmapInfo &info = moduleInfos[module.getOperation()];
    if (failed(instrumentModule(info)))
      return signalPassFailure();
  }

  for (moore::SVModuleOp module : modules) {
    ModuleBitmapInfo &info = moduleInfos[module.getOperation()];
    rewriteInstances(info, moduleLookup);
  }
}

} // namespace

std::unique_ptr<mlir::Pass> createMooreInstrumentPathBitmapPass() {
  return std::make_unique<MooreInstrumentPathBitmapPass>();
}

void registerMooreInstrumentPathBitmapPass() {
  mlir::PassRegistration<MooreInstrumentPathBitmapPass>();
}

} // namespace circt::pcov
