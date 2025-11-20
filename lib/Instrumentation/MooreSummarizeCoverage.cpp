//===- MooreSummarizeCoverage.cpp - Moore coverage summary reporting ------===//
//
// Part of the pcov project.
//
//===----------------------------------------------------------------------===//

#include "pcov/Instrumentation/Passes.h"

#include "circt/Dialect/Moore/MooreOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <string>
#include <utility>

using namespace mlir;
using namespace circt;

namespace circt::pcov {
#define GEN_PASS_DEF_MOORESUMMARIZECOVERAGE
#include "pcov/Instrumentation/InstrumentationPasses.h.inc"
} // namespace circt::pcov

namespace circt::pcov {
namespace {

struct ProcedureCoverageInfo {
  unsigned procIndex = 0;
  unsigned pathIdWidth = 0;
  uint64_t pointCount = 0;
  std::string procName;
};

struct ModuleCoverageInfo {
  std::string moduleName;
  SmallVector<ProcedureCoverageInfo, 8> procedures;
  unsigned totalPathWidth = 0;
  uint64_t totalPointCount = 0;
  uint64_t hierarchicalPathWidth = 0;
  uint64_t hierarchicalPointCount = 0;
  bool referencedByInstance = false;
};

class MooreSummarizeCoveragePass
    : public impl::MooreSummarizeCoverageBase<MooreSummarizeCoveragePass> {
public:
  void runOnOperation() override;
};

void MooreSummarizeCoveragePass::runOnOperation() {
  ModuleOp top = getOperation();

  SmallVector<ModuleCoverageInfo, 8> modules;
  modules.reserve(8);
  llvm::StringSet<> referencedModules;
  llvm::StringMap<std::pair<uint64_t, uint64_t>> moduleHierarchyStats;

  for (moore::SVModuleOp svModule : top.getOps<moore::SVModuleOp>()) {
    ModuleCoverageInfo moduleInfo;
    if (auto nameAttr = svModule.getSymNameAttr())
      moduleInfo.moduleName = nameAttr.getValue().str();
    else
      moduleInfo.moduleName = "<anonymous>";

    for (moore::ProcedureOp proc : svModule.getOps<moore::ProcedureOp>()) {
      if (!proc->hasAttr("pcov.coverage.instrumented"))
        continue;

      ProcedureCoverageInfo procInfo;
      if (auto indexAttr =
              proc->getAttrOfType<IntegerAttr>("pcov.coverage.proc_index"))
        procInfo.procIndex = static_cast<unsigned>(indexAttr.getInt());
      if (auto widthAttr =
              proc->getAttrOfType<IntegerAttr>("pcov.coverage.path_id_width"))
        procInfo.pathIdWidth = static_cast<unsigned>(widthAttr.getInt());
      if (auto pointAttr =
              proc->getAttrOfType<IntegerAttr>("pcov.coverage.point_count"))
        procInfo.pointCount = pointAttr.getInt();

      if (auto symName =
              proc->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
        procInfo.procName = symName.getValue().str();

      moduleInfo.totalPathWidth += procInfo.pathIdWidth;
      moduleInfo.totalPointCount += procInfo.pointCount;
      moduleInfo.procedures.push_back(std::move(procInfo));
    }

    uint64_t pathBusWidth = 0;
    uint64_t coveragePoints = 0;
    if (auto attr = svModule->getAttrOfType<IntegerAttr>(
            "pcov.coverage.total_path_bus_width"))
      pathBusWidth = attr.getInt();
    if (auto attr = svModule->getAttrOfType<IntegerAttr>(
            "pcov.coverage.total_cov_points"))
      coveragePoints = attr.getInt();
    moduleHierarchyStats[moduleInfo.moduleName] =
        std::make_pair(pathBusWidth, coveragePoints);
    moduleInfo.hierarchicalPathWidth = pathBusWidth;
    moduleInfo.hierarchicalPointCount = coveragePoints;

    if (!moduleInfo.procedures.empty())
      modules.push_back(std::move(moduleInfo));
  }

  for (moore::SVModuleOp svModule : top.getOps<moore::SVModuleOp>())
    for (moore::InstanceOp inst : svModule.getOps<moore::InstanceOp>())
      if (auto nameAttr = inst.getModuleNameAttr())
        referencedModules.insert(nameAttr.getValue());

  for (ModuleCoverageInfo &info : modules)
    info.referencedByInstance = referencedModules.contains(info.moduleName);

  if (modules.empty()) {
    llvm::outs() << "[pcov] No instrumented Moore processes were found\n";
    return;
  }

  uint64_t totalWidth = 0;
  uint64_t totalProcedures = 0;
  uint64_t totalPoints = 0;
  uint64_t hierarchicalWidth = 0;
  uint64_t hierarchicalPoints = 0;
  for (const ModuleCoverageInfo &moduleInfo : modules) {
    totalWidth += moduleInfo.totalPathWidth;
    totalProcedures += moduleInfo.procedures.size();
    totalPoints += moduleInfo.totalPointCount;
  }

  for (const auto &entry : moduleHierarchyStats) {
    if (!referencedModules.contains(entry.getKey())) {
      hierarchicalWidth += entry.getValue().first;
      hierarchicalPoints += entry.getValue().second;
    }
  }

  llvm::outs() << "[pcov] Moore coverage summary: " << totalProcedures
               << " process(es) across " << modules.size()
               << " module(s), definition path_id width = " << totalWidth
               << ", hierarchical path bus width = " << hierarchicalWidth
               << ", definition coverage points = " << totalPoints
               << ", hierarchical coverage points = " << hierarchicalPoints
               << "\n";

  for (const ModuleCoverageInfo &moduleInfo : modules) {
    llvm::outs() << "[pcov]   module " << moduleInfo.moduleName << ": "
                 << moduleInfo.procedures.size()
                 << " process(es), definition path_id width = "
                 << moduleInfo.totalPathWidth << ", coverage points = "
                 << moduleInfo.totalPointCount;
    if (moduleInfo.hierarchicalPathWidth)
      llvm::outs() << ", hierarchical path bus width = "
                   << moduleInfo.hierarchicalPathWidth;
    if (moduleInfo.hierarchicalPointCount)
      llvm::outs() << ", hierarchical coverage points = "
                   << moduleInfo.hierarchicalPointCount;
    if (!moduleInfo.referencedByInstance)
      llvm::outs() << " [root]";
    llvm::outs() << "\n";
    for (const ProcedureCoverageInfo &procInfo : moduleInfo.procedures) {
      llvm::outs() << "[pcov]     proc#" << procInfo.procIndex;
      if (!procInfo.procName.empty())
        llvm::outs() << " (" << procInfo.procName << ")";
      llvm::outs() << ": path_id width = " << procInfo.pathIdWidth
                   << ", coverage points = " << procInfo.pointCount << "\n";
    }
  }
}

} // namespace

std::unique_ptr<mlir::Pass> createMooreSummarizeCoveragePass() {
  return std::make_unique<MooreSummarizeCoveragePass>();
}

void registerMooreSummarizeCoveragePass() {
  mlir::PassRegistration<MooreSummarizeCoveragePass>();
}

} // namespace circt::pcov
