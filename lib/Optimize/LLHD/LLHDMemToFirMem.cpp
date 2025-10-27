//===- LLHDMemToFirMem.cpp - Convert LLHD Memories to FirMem -------------===//
//
// Part of the circt-cf project.
//
//===----------------------------------------------------------------------===//

#include "circt-cf/Optimize/LLHD/Passes.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llhd-mem-to-firmem"

using namespace mlir;
using namespace circt;
using namespace ::circt::llhd;
using namespace ::circt::hw;
using namespace ::circt::seq;

namespace circt::cfatrace::optimize::llhd {
namespace cllhd = ::circt::llhd;
namespace chw = ::circt::hw;
namespace cseq = ::circt::seq;
namespace ccomb = ::circt::comb;
#define GEN_PASS_DEF_LLHDMEMTOFIRMEMPASS
#include "circt-cf/Optimize/LLHD/Passes.h.inc"
} // namespace circt::cfatrace::optimize::llhd

namespace circt::cfatrace::optimize::llhd {
namespace {

struct PortInfo {
  cseq::FirRegOp regOp;
  chw::ArrayInjectOp arrayInject;
  ccomb::MuxOp innerMux;
  ccomb::MuxOp outerMux;
  Value clock;
  Value writeEnable;
  Value writeAddr;
  Value writeData;
  Value readAddr;
};

struct MultiPortMemPattern {
  cllhd::SignalOp signal;
  SmallVector<PortInfo> ports;
  uint64_t depth;
  uint64_t width;
};

class LLHDMemToFirMemPass
    : public impl::LLHDMemToFirMemPassBase<LLHDMemToFirMemPass> {
public:
  using impl::LLHDMemToFirMemPassBase<LLHDMemToFirMemPass>::Base::Base;

  void runOnOperation() override;

private:
  bool analyzeSignal(cllhd::SignalOp sig, MultiPortMemPattern &pattern);
  bool analyzePort(cllhd::DriveOp drv, PortInfo &port);
  void convertToFirMem(MultiPortMemPattern &pattern);
  std::optional<std::pair<uint64_t, uint64_t>> getArrayDimensions(Type type);

  SmallVector<Operation *> opsToErase;
};

std::optional<std::pair<uint64_t, uint64_t>>
LLHDMemToFirMemPass::getArrayDimensions(Type type) {
  if (auto refType = dyn_cast<cllhd::RefType>(type))
    type = refType.getNestedType();

  if (auto arrayType = dyn_cast<chw::ArrayType>(type)) {
    auto elemType = arrayType.getElementType();
    if (auto intType = dyn_cast<IntegerType>(elemType))
      return std::make_pair(arrayType.getNumElements(), intType.getWidth());
  }
  return std::nullopt;
}

bool LLHDMemToFirMemPass::analyzePort(cllhd::DriveOp drv, PortInfo &port) {
  auto regOp = drv.getValue().getDefiningOp<cseq::FirRegOp>();
  if (!regOp)
    return false;

  port.regOp = regOp;
  port.clock = regOp.getClk();

  Value nextValue = regOp.getNext();
  auto outerMux = nextValue.getDefiningOp<ccomb::MuxOp>();
  if (!outerMux)
    return false;

  port.outerMux = outerMux;

  Value writeEnable = outerMux.getCond();
  Value writePath = outerMux.getTrueValue();
  Value feedback = outerMux.getFalseValue();

  if (feedback != regOp.getResult())
    return false;

  chw::ArrayInjectOp arrayInject =
      writePath.getDefiningOp<chw::ArrayInjectOp>();

  if (!arrayInject) {
    if (auto innerMux = writePath.getDefiningOp<ccomb::MuxOp>()) {
      port.innerMux = innerMux;

      arrayInject = innerMux.getTrueValue().getDefiningOp<chw::ArrayInjectOp>();
      if (!arrayInject)
        arrayInject =
            innerMux.getFalseValue().getDefiningOp<chw::ArrayInjectOp>();
    }
  }

  if (!arrayInject)
    return false;

  port.arrayInject = arrayInject;
  port.writeAddr = arrayInject.getIndex();
  port.writeData = arrayInject.getElement();
  port.writeEnable = writeEnable;

  return true;
}

bool LLHDMemToFirMemPass::analyzeSignal(cllhd::SignalOp sig,
                                        MultiPortMemPattern &pattern) {
  auto dims = getArrayDimensions(sig.getType());
  if (!dims)
    return false;

  pattern.signal = sig;
  pattern.depth = dims->first;
  pattern.width = dims->second;

  SmallVector<cllhd::DriveOp> drives;
  for (Operation *user : sig.getResult().getUsers())
    if (auto drv = dyn_cast<cllhd::DriveOp>(user))
      drives.push_back(drv);

  if (drives.empty())
    return false;

  for (auto drv : drives) {
    PortInfo port;
    if (analyzePort(drv, port))
      pattern.ports.push_back(port);
  }

  if (pattern.ports.empty())
    return false;

  return true;
}

void LLHDMemToFirMemPass::convertToFirMem(MultiPortMemPattern &pattern) {
  auto sigName = pattern.signal.getName();

  LLVM_DEBUG({
    llvm::dbgs() << "[LLHDMemToFirMem] Converting signal '" << sigName << "' ("
                 << pattern.depth << "x" << pattern.width
                 << ") to seq.firmem with " << pattern.ports.size()
                 << " port(s)\n";
  });

  ImplicitLocOpBuilder builder(pattern.signal.getLoc(), pattern.signal);

  auto memType = FirMemType::get(builder.getContext(), pattern.depth,
                                 pattern.width, /*maskWidth=*/1);
  auto firMem = cseq::FirMemOp::create(
      builder, memType, /*readLatency=*/0, /*writeLatency=*/1,
      /*readUnderWrite=*/cseq::RUW::Undefined,
      /*writeUnderWrite=*/cseq::WUW::Undefined,
      /*name=*/builder.getStringAttr("mem"), /*innerSym=*/chw::InnerSymAttr{},
      /*init=*/cseq::FirMemInitAttr{}, /*prefix=*/StringAttr{},
      /*outputFile=*/Attribute{});

  LLVM_DEBUG(llvm::dbgs() << "  - Created seq.firmem<" << pattern.depth << " x "
                          << pattern.width << ">\n");

  cllhd::ProbeOp probe = nullptr;
  for (Operation *user : pattern.signal.getResult().getUsers()) {
    if (auto prb = dyn_cast<cllhd::ProbeOp>(user)) {
      probe = prb;
      break;
    }
  }

  if (!probe) {
    LLVM_DEBUG(llvm::dbgs() << "  - Warning: No probe found, skipping\n");
    return;
  }

  SmallVector<chw::ArrayGetOp> allArrayGets;
  for (Operation *user : probe.getResult().getUsers())
    if (auto arrayGet = dyn_cast<chw::ArrayGetOp>(user))
      allArrayGets.push_back(arrayGet);

  for (size_t i = 0; i < pattern.ports.size(); ++i) {
    auto &port = pattern.ports[i];
    builder.setInsertionPointAfter(firMem);

    Value mask;
    (void)cseq::FirMemWriteOp::create(builder, firMem, port.writeAddr,
                                      port.clock, port.writeEnable,
                                      port.writeData, mask);

    LLVM_DEBUG(llvm::dbgs() << "  - Created write port " << i << "\n");

    (void)cseq::FirMemReadOp::create(builder, firMem, port.writeAddr,
                                     port.clock, /*enable=*/Value());

    opsToErase.push_back(port.regOp);
    opsToErase.push_back(port.arrayInject);
    opsToErase.push_back(port.outerMux);
    if (port.innerMux)
      opsToErase.push_back(port.innerMux);
  }

  if (!pattern.ports.empty() && !allArrayGets.empty()) {
    for (auto arrayGet : allArrayGets) {
      Value readAddr = arrayGet.getIndex();
      Value readData =
          cseq::FirMemReadOp::create(builder, firMem, readAddr,
                                     pattern.ports[0].clock,
                                     /*enable=*/Value());

      LLVM_DEBUG(llvm::dbgs() << "  - Created read port for hw.array_get\n");

      arrayGet.getResult().replaceAllUsesWith(readData);
      opsToErase.push_back(arrayGet);
    }
  }

  opsToErase.push_back(probe);
  opsToErase.push_back(pattern.signal);

  for (Operation *user : pattern.signal.getResult().getUsers())
    if (auto drv = dyn_cast<cllhd::DriveOp>(user))
      opsToErase.push_back(drv);

  LLVM_DEBUG({
    llvm::dbgs() << "  - Marked " << opsToErase.size()
                 << " operations for removal (llhd.sig, llhd.prb, llhd.drv, "
                 << "seq.firreg, hw.array_inject, comb.mux, hw.array_get)\n";
  });
}

void LLHDMemToFirMemPass::runOnOperation() {
  auto module = getOperation();

  SmallVector<cllhd::SignalOp> signals;

  module.walk([&](cllhd::SignalOp sig) {
    auto sigType = sig.getResult().getType();
    if (auto refType = dyn_cast<cllhd::RefType>(sigType))
      if (isa<chw::ArrayType>(refType.getNestedType()))
        signals.push_back(sig);
  });

  for (auto sig : signals) {
    MultiPortMemPattern pattern;
    if (analyzeSignal(sig, pattern))
      convertToFirMem(pattern);
  }

  for (Operation *op : opsToErase) {
    if (!op->use_empty())
      op->dropAllUses();
    op->erase();
  }
  opsToErase.clear();
}

} // namespace

std::unique_ptr<mlir::Pass> createLLHDMemToFirMemPass() {
  return std::make_unique<LLHDMemToFirMemPass>();
}

void registerLLHDTransformPasses() {
  static bool initOnce = []() {
    mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
      return createLLHDMemToFirMemPass();
    });
    return true;
  }();
  (void)initOnce;
}

} // namespace circt::cfatrace::optimize::llhd
