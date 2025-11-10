#include "circt-cf/Instrumentation/MooreProcedureAnalysis.h"
#include "circt-cf/Instrumentation/Passes.h"

#include "circt/Dialect/Moore/MooreOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <cctype>
#include <string>

using namespace mlir;
using namespace circt;

namespace circt::pcov {
#define GEN_PASS_DEF_MOOREEXPORTPROCESSCFG
#include "circt-cf/Instrumentation/InstrumentationPasses.h.inc"
} // namespace circt::pcov

namespace circt::pcov {
namespace {

static std::string sanitizeForIdentifier(StringRef name, StringRef fallback) {
  if (name.empty())
    name = fallback;

  std::string result;
  result.reserve(name.size());
  for (char ch : name) {
    if (llvm::isAlnum(static_cast<unsigned char>(ch)) || ch == '_' ||
        ch == '-' || ch == '.')
      result.push_back(ch);
    else
      result.push_back('_');
  }
  if (result.empty())
    result = fallback.str();
  return result;
}

static std::string
getSymbolNameOrFallback(Operation *op, StringRef fallbackPrefix,
                        unsigned index = 0) {
  if (auto symAttr =
          op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
    if (!symAttr.getValue().empty())
      return symAttr.getValue().str();

  return (Twine(fallbackPrefix) + Twine(index)).str();
}

class MooreExportProcessCFGPass
    : public impl::MooreExportProcessCFGBase<MooreExportProcessCFGPass> {
public:
  MooreExportProcessCFGPass() = default;
  explicit MooreExportProcessCFGPass(StringRef directory) {
    outputDir = directory.str();
  }

  void runOnOperation() override;

private:
  LogicalResult emitProcedureCFG(moore::SVModuleOp module,
                                 moore::ProcedureOp proc,
                                 const MooreProcedureCFGAnalysis &analysis,
                                 unsigned procIndex,
                                 StringRef directory) const;
  void writeDotHeader(raw_ostream &os, StringRef graphName) const;
  void writeDotNodes(raw_ostream &os, llvm::ArrayRef<Block *> blockOrder,
                     const llvm::DenseMap<Block *, std::string> &blockNames,
                     const MooreProcedureCFGAnalysis &analysis) const;
  void writeDotEdges(raw_ostream &os, llvm::ArrayRef<Block *> blockOrder,
                     const llvm::DenseMap<Block *, std::string> &blockNames,
                     const MooreProcedureCFGAnalysis &analysis) const;
};

void MooreExportProcessCFGPass::writeDotHeader(raw_ostream &os,
                                               StringRef graphName) const {
  os << "digraph \"" << graphName << "\" {\n";
  os << "  node [shape=box];\n";
}

void MooreExportProcessCFGPass::writeDotNodes(
    raw_ostream &os, llvm::ArrayRef<Block *> blockOrder,
    const llvm::DenseMap<Block *, std::string> &blockNames,
    const MooreProcedureCFGAnalysis &analysis) const {
  for (Block *block : blockOrder) {
    auto it = blockNames.find(block);
    if (it == blockNames.end())
      continue;
    uint64_t paths =
        analysis.numPaths.lookup(block); // guaranteed by analysis result
    os << "  " << it->second << " [label=\"" << it->second << "\\npaths="
       << paths << "\"];\n";
  }
}

static std::string formatEdgeLabel(StringRef prefix, uint64_t weight) {
  if (weight == 0 && prefix.empty())
    return "";

  std::string label;
  if (!prefix.empty())
    label.append(prefix.str());

  if (weight != 0) {
    if (!label.empty())
      label.append("\\n");
    label.append("weight=");
    label.append(std::to_string(weight));
  }
  return label;
}

void MooreExportProcessCFGPass::writeDotEdges(
    raw_ostream &os, llvm::ArrayRef<Block *> blockOrder,
    const llvm::DenseMap<Block *, std::string> &blockNames,
    const MooreProcedureCFGAnalysis &analysis) const {
  for (Block *block : blockOrder) {
    auto nameIt = blockNames.find(block);
    if (nameIt == blockNames.end())
      continue;
    StringRef srcName = nameIt->second;
    Operation *terminator = block->getTerminator();

    if (auto cond = dyn_cast<cf::CondBranchOp>(terminator)) {
      Block *trueDest = cond.getTrueDest();
      Block *falseDest = cond.getFalseDest();
      if (auto found = blockNames.find(trueDest); found != blockNames.end()) {
        uint64_t weight =
            analysis.edgeWeights.lookup(BlockEdge{block, trueDest});
        std::string label = formatEdgeLabel("true", weight);
        if (!label.empty())
          os << "  " << srcName << " -> " << found->second << " [label=\""
             << label << "\"];\n";
        else
          os << "  " << srcName << " -> " << found->second << ";\n";
      }
      if (auto found = blockNames.find(falseDest); found != blockNames.end()) {
        uint64_t weight =
            analysis.edgeWeights.lookup(BlockEdge{block, falseDest});
        std::string label = formatEdgeLabel("false", weight);
        if (!label.empty())
          os << "  " << srcName << " -> " << found->second << " [label=\""
             << label << "\"];\n";
        else
          os << "  " << srcName << " -> " << found->second << ";\n";
      }
      continue;
    }

    for (Block *dest : block->getSuccessors()) {
      if (auto found = blockNames.find(dest); found != blockNames.end()) {
        uint64_t weight =
            analysis.edgeWeights.lookup(BlockEdge{block, dest});
        std::string label = formatEdgeLabel("", weight);
        if (!label.empty())
          os << "  " << srcName << " -> " << found->second << " [label=\""
             << label << "\"];\n";
        else
          os << "  " << srcName << " -> " << found->second << ";\n";
      }
    }
  }
}

LogicalResult MooreExportProcessCFGPass::emitProcedureCFG(
    moore::SVModuleOp module, moore::ProcedureOp proc,
    const MooreProcedureCFGAnalysis &analysis, unsigned procIndex,
    StringRef directory) const {
  std::string moduleSymbol =
      getSymbolNameOrFallback(module.getOperation(), "module");
  std::string procSymbol =
      getSymbolNameOrFallback(proc.getOperation(), "proc", procIndex);
  std::string moduleName = sanitizeForIdentifier(moduleSymbol, "module");
  std::string procName = sanitizeForIdentifier(procSymbol, "proc");
  std::string graphName = (Twine(moduleName) + "__" + procName).str();

  llvm::DenseMap<Block *, std::string> blockNames;
  llvm::SmallVector<Block *> blockOrder;
  for (Block &block : proc.getBody()) {
    Block *blockPtr = &block;
    if (!analysis.numPaths.contains(blockPtr))
      continue;
    blockOrder.push_back(blockPtr);
  }

  blockNames.reserve(blockOrder.size());
  mlir::AsmState asmState(module);
  unsigned fallbackIndex = 0;
  for (Block *blockPtr : blockOrder) {
    std::string buffer;
    {
      llvm::raw_string_ostream os(buffer);
      blockPtr->printAsOperand(os, asmState);
    }
    if (!buffer.empty() && buffer.front() == '^')
      buffer.erase(buffer.begin());
    if (buffer.empty())
      buffer = (Twine("bb") + Twine(fallbackIndex)).str();
    blockNames.try_emplace(blockPtr, buffer);
    ++fallbackIndex;
  }

#if 0
  llvm::errs() << "[pcov-cfg] module " << moduleSymbol << " proc#" << procIndex
               << " blocks=" << blockOrder.size() << "\n";
#endif

  uint64_t totalPaths = analysis.numPaths.lookup(analysis.entryBlock);
  auto renderGraph = [&](raw_ostream &os) {
    os << "// CFG for module `" << moduleSymbol << "`, procedure `"
       << procSymbol << "` (paths=" << totalPaths << ")\n";
    writeDotHeader(os, graphName);
    writeDotNodes(os, blockOrder, blockNames, analysis);
    writeDotEdges(os, blockOrder, blockNames, analysis);
    os << "}\n";
  };

  if (!directory.empty()) {
    SmallString<256> filePath(directory);
    llvm::sys::path::append(filePath, graphName + ".dot");

    if (auto parent = llvm::sys::path::parent_path(filePath);
        !parent.empty()) {
      if (std::error_code ec =
              llvm::sys::fs::create_directories(parent, /*IgnoreExisting=*/true)) {
        module.emitError("failed to create directory '")
            << parent << "': " << ec.message();
        return failure();
      }
    }

    std::error_code ec;
    llvm::raw_fd_ostream file(filePath, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      module.emitError("failed to open '")
          << filePath << "' for writing: " << ec.message();
      return failure();
    }
    renderGraph(file);
    file << '\n';
    return success();
  }

  raw_ostream &os = llvm::outs();
  renderGraph(os);
  os << "\n";
  return success();
}

void MooreExportProcessCFGPass::runOnOperation() {
  moore::SVModuleOp module = getOperation();
  std::string directory = outputDir;

  if (!directory.empty()) {
    SmallString<256> resolved(directory);
    llvm::sys::path::remove_dots(resolved, /*remove_dot_dot=*/true);
    directory = resolved.str().str();
    if (std::error_code ec = llvm::sys::fs::create_directories(
            directory, /*IgnoreExisting=*/true)) {
      module.emitError("failed to create directory '")
          << directory << "': " << ec.message();
      signalPassFailure();
      return;
    }
  }

  for (moore::ProcedureOp proc : module.getOps<moore::ProcedureOp>()) {
    if (!proc->hasAttr("pcov.coverage.instrumented"))
      continue;
    auto kindAttr = proc->getAttrOfType<StringAttr>("pcov.coverage.kind");
    if (!kindAttr || kindAttr.getValue() != "path")
      continue;

    IntegerAttr procIndexAttr =
        proc->getAttrOfType<IntegerAttr>("pcov.coverage.proc_index");
    if (!procIndexAttr)
      continue;
    unsigned procIndex = procIndexAttr.getInt();

    MooreProcedureAnalysisResult analysisResult =
        analyzeMooreProcedure(proc, /*emitDiagnostics=*/true);
    if (analysisResult.fatalError) {
      signalPassFailure();
      return;
    }
    if (!analysisResult.analysis)
      continue;
    if (failed(emitProcedureCFG(module, proc, *analysisResult.analysis,
                                procIndex, directory))) {
      signalPassFailure();
      return;
    }
  }
}

} // namespace

std::unique_ptr<mlir::Pass> createMooreExportProcessCFGPass() {
  return std::make_unique<MooreExportProcessCFGPass>();
}

std::unique_ptr<mlir::Pass>
createMooreExportProcessCFGPass(StringRef outputDir) {
  return std::make_unique<MooreExportProcessCFGPass>(outputDir);
}

void registerMooreExportProcessCFGPass() {
  mlir::PassRegistration<MooreExportProcessCFGPass>();
}

} // namespace circt::pcov
