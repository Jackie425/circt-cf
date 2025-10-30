#include "circt-cf/Instrumentation/Passes.h"

#include "circt/Dialect/Moore/MooreOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <cctype>
#include <string>

using namespace mlir;
using namespace circt;

namespace circt::svcf {
#define GEN_PASS_DEF_MOOREEXPORTPROCESSCFG
#include "circt-cf/Instrumentation/InstrumentationPasses.h.inc"
} // namespace circt::svcf

namespace circt::svcf {
namespace {

/// Sanitize strings that will be used in file names to avoid characters that
/// are problematic on common filesystems.
static std::string sanitizeForFileName(StringRef name, StringRef fallback) {
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

/// Retrieve the symbol name of an operation if it has one, otherwise return
/// the provided fallback identifier.
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
  using Base::Base;

  void runOnOperation() override;

private:
  LogicalResult emitProcedureCFG(moore::SVModuleOp module,
                                 moore::ProcedureOp proc,
                                 StringRef directory) const;
  void writeDotHeader(raw_ostream &os, StringRef graphName) const;
  void writeDotNodes(raw_ostream &os, llvm::ArrayRef<Block *> blockOrder,
                     const llvm::DenseMap<Block *, std::string> &blockNames)
      const;
  void writeDotEdges(raw_ostream &os, llvm::ArrayRef<Block *> blockOrder,
                     const llvm::DenseMap<Block *, std::string> &blockNames)
      const;
};

LogicalResult
MooreExportProcessCFGPass::emitProcedureCFG(moore::SVModuleOp module,
                                            moore::ProcedureOp proc,
                                            StringRef directory) const {
  std::string moduleName = sanitizeForFileName(
      getSymbolNameOrFallback(module.getOperation(), "module"), "module");
  std::string procName = sanitizeForFileName(
      getSymbolNameOrFallback(proc.getOperation(), "proc"), "proc");

  SmallString<256> filePath(directory);
  llvm::sys::path::append(filePath,
                          (Twine(moduleName) + "__" + procName + ".dot")
                              .str());

  llvm::StringRef parentDirRef = llvm::sys::path::parent_path(filePath);
  if (!parentDirRef.empty()) {
    std::error_code error =
        llvm::sys::fs::create_directories(parentDirRef, /*IgnoreExisting=*/true);
    if (error) {
      module.emitError("failed to create directory '")
          << parentDirRef << "': " << error.message();
      return failure();
    }
  }

  std::error_code openError;
  llvm::raw_fd_ostream file(filePath, openError, llvm::sys::fs::OF_Text);
  if (openError) {
    module.emitError("failed to open '")
        << filePath << "' for writing: " << openError.message();
    return failure();
  }

  llvm::DenseMap<Block *, std::string> blockNames;
  llvm::SmallVector<Block *> blockOrder;
  unsigned blockIndex = 0;
  for (Block &block : proc.getBody()) {
    std::string blockName = (Twine("bb") + Twine(blockIndex)).str();
    blockOrder.push_back(&block);
    blockNames.try_emplace(&block, std::move(blockName));
    ++blockIndex;
  }

  std::string graphName = (Twine(moduleName) + "__" + procName).str();
  writeDotHeader(file, graphName);
  writeDotNodes(file, blockOrder, blockNames);
  writeDotEdges(file, blockOrder, blockNames);
  file << "}\n";

  proc.emitRemark("wrote CFG to ") << filePath;
  return success();
}

void MooreExportProcessCFGPass::writeDotHeader(raw_ostream &os,
                                               StringRef graphName) const {
  os << "digraph \"" << graphName << "\" {\n";
  os << "  node [shape=box];\n";
}

void MooreExportProcessCFGPass::writeDotNodes(
    raw_ostream &os, llvm::ArrayRef<Block *> blockOrder,
    const llvm::DenseMap<Block *, std::string> &blockNames) const {
  for (Block *block : blockOrder) {
    auto it = blockNames.find(block);
    if (it == blockNames.end())
      continue;
    os << "  " << it->second << " [label=\"" << it->second << "\"];\n";
  }
}

void MooreExportProcessCFGPass::writeDotEdges(
    raw_ostream &os, llvm::ArrayRef<Block *> blockOrder,
    const llvm::DenseMap<Block *, std::string> &blockNames) const {
  for (Block *block : blockOrder) {
    auto nameIt = blockNames.find(block);
    if (nameIt == blockNames.end())
      continue;
    StringRef srcName = nameIt->second;
    Operation *terminator = block->getTerminator();

    if (auto cond = dyn_cast<cf::CondBranchOp>(terminator)) {
      Block *trueDest = cond.getTrueDest();
      Block *falseDest = cond.getFalseDest();
      if (auto found = blockNames.find(trueDest); found != blockNames.end())
        os << "  " << srcName << " -> " << found->second
           << " [label=\"true\"];\n";
      if (auto found = blockNames.find(falseDest); found != blockNames.end())
        os << "  " << srcName << " -> " << found->second
           << " [label=\"false\"];\n";
      continue;
    }

    for (Block *dest : block->getSuccessors()) {
      if (auto found = blockNames.find(dest); found != blockNames.end())
        os << "  " << srcName << " -> " << found->second << ";\n";
    }
  }
}

void MooreExportProcessCFGPass::runOnOperation() {
  moore::SVModuleOp module = getOperation();
  std::string directory = outputDir.getValue();
  if (directory.empty())
    directory = ".";

  llvm::SmallString<256> resolvedDir(directory);
  llvm::sys::path::remove_dots(resolvedDir, /*remove_dot_dot=*/true);

  for (moore::ProcedureOp proc : module.getOps<moore::ProcedureOp>()) {
    if (failed(emitProcedureCFG(module, proc, resolvedDir))) {
      signalPassFailure();
      return;
    }
  }
}

} // namespace

std::unique_ptr<mlir::Pass> createMooreExportProcessCFGPass() {
  return std::make_unique<MooreExportProcessCFGPass>();
}

void registerMooreExportProcessCFGPass() {
  mlir::PassRegistration<MooreExportProcessCFGPass>();
}

} // namespace circt::svcf
