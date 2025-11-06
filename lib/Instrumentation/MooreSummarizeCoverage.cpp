//===- MooreSummarizeCoverage.cpp - Moore coverage summary reporting ------===//
//
// Part of the circt-cf project.
//
//===----------------------------------------------------------------------===//

#include "circt-cf/Instrumentation/Passes.h"

#include "circt/Dialect/Moore/MooreOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <string>
#include <utility>

using namespace mlir;
using namespace circt;

namespace circt::pcov {
#define GEN_PASS_DEF_MOORESUMMARIZECOVERAGE
#include "circt-cf/Instrumentation/InstrumentationPasses.h.inc"
} // namespace circt::pcov

namespace circt::pcov {
namespace {

struct ProcedureCoverageInfo {
  unsigned procIndex = 0;
  unsigned pathIdWidth = 0;
  std::string procName;
};

struct ModuleCoverageInfo {
  std::string moduleName;
  SmallVector<ProcedureCoverageInfo, 8> procedures;
  unsigned totalPathWidth = 0;
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

      if (auto symName =
              proc->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
        procInfo.procName = symName.getValue().str();

      moduleInfo.totalPathWidth += procInfo.pathIdWidth;
      moduleInfo.procedures.push_back(std::move(procInfo));
    }

    if (!moduleInfo.procedures.empty())
      modules.push_back(std::move(moduleInfo));
  }

  if (modules.empty()) {
    llvm::outs() << "[pcov] No instrumented Moore processes were found\n";
    return;
  }

  uint64_t totalWidth = 0;
  uint64_t totalProcedures = 0;
  for (const ModuleCoverageInfo &moduleInfo : modules) {
    totalWidth += moduleInfo.totalPathWidth;
    totalProcedures += moduleInfo.procedures.size();
  }

  llvm::outs() << "[pcov] Moore coverage summary: " << totalProcedures
               << " process(es) across " << modules.size()
               << " module(s), total path_id width = " << totalWidth << "\n";

  for (const ModuleCoverageInfo &moduleInfo : modules) {
    llvm::outs() << "[pcov]   module " << moduleInfo.moduleName << ": "
                 << moduleInfo.procedures.size()
                 << " process(es), accumulated path_id width = "
                 << moduleInfo.totalPathWidth << "\n";
    for (const ProcedureCoverageInfo &procInfo : moduleInfo.procedures) {
      llvm::outs() << "[pcov]     proc#" << procInfo.procIndex;
      if (!procInfo.procName.empty())
        llvm::outs() << " (" << procInfo.procName << ")";
      llvm::outs() << ": path_id width = " << procInfo.pathIdWidth << "\n";
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
