import os
import lit.formats
from lit.llvm import llvm_config

config.name = "pcov"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.pcov_obj_root

llvm_config.use_default_substitutions()

llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.mlir_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.pcov_tools_dir,
                             append_path=True)

config.substitutions.append(("%pcov", "pcov"))
config.substitutions.append(("%pcov-opt", "pcov-opt"))
