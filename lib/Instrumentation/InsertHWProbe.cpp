//===- InstrumentationPass.cpp - Basic instrumentation pass ---------------===//
//
// Part of the circt-cfa-trace project.
//
//===----------------------------------------------------------------------===//

#include "circt-cfa-trace/Instrumentation/Transforms/Passes.h"

#define GEN_PASS_DEF_INSERTHWPROBE
#include "circt-cfa-trace/Instrumentation/Transforms/InstrumentationPasses.h.inc"

#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassRegistry.h"

namespace circt::cfatrace {

namespace {
/// Simple pass that tags every `hw.module` with a marker attribute. This
/// demonstrates how custom instrumentation passes can reach into CIRCT IR.
class InsertHWProbePass
    : public impl::InsertHWProbeBase<InsertHWProbePass> {
public:
  using Base::Base;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    auto *context = module.getContext();
    auto unitAttr = mlir::UnitAttr::get(context);

    module.walk([&](hw::HWModuleOp op) {
      if (!op->hasAttr("hw.probes"))
        op->setAttr("hw.probes", unitAttr);
    });
  }
};
} // namespace

std::unique_ptr<mlir::Pass> createInsertHWProbePass() {
  return std::make_unique<InsertHWProbePass>();
}

void registerInsertHWProbePasses() {
  mlir::PassRegistration<InsertHWProbePass>();
}

} // namespace circt::cfatrace
