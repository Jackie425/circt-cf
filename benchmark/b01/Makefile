# simple Makefile to test a CIRCT flow from SystemVerilog to MLIR to SystemVerilog
DESIGN ?= b01
BUILD_DIR = build
CFG_DIR = $(BUILD_DIR)/cfg
# Debug options for LLVM_DEBUG output
# Use --debug to see all debug output, or --debug-only=<type> for specific types

DEBUG_OPTS ?=
# Example: make run DEBUG_OPTS="--debug-only=llhd-lower-processes"
# Example: make run DEBUG_OPTS="--debug"

######################################################################
default: run

run:
	mkdir -p $(BUILD_DIR)
	circt-cf --ir-moore $(DESIGN).v -o build/$(DESIGN)_moore_cov.mlir \
         --mlir-print-ir-before=moore-instrument-coverage \
		 --mlir-print-ir-module-scope --mlir-disable-threading > build/$(DESIGN)_moore.mlir 2>&1
	circt-cf --mlir-pretty-debuginfo --verbose-pass-executions $(DEBUG_OPTS) --ir-hw $(DESIGN).v -o $(BUILD_DIR)/$(DESIGN)_hw.mlir
	circt-cf --mlir-pretty-debuginfo --verbose-pass-executions $(DEBUG_OPTS) --ir-sv $(DESIGN).v -o $(BUILD_DIR)/$(DESIGN)_sv.mlir
	circt-cf --mlir-pretty-debuginfo --verbose-pass-executions $(DEBUG_OPTS) --sv-sv $(DESIGN).v -o $(BUILD_DIR)/$(DESIGN).sv

run-split:
	mkdir -p $(BUILD_DIR)/split-output
	circt-cf --verbose-pass-executions $(DEBUG_OPTS) --sv-sv --split-verilog --split-verilog-dir=$(BUILD_DIR)/split-output $(DESIGN).v

graph:
	mkdir -p $(CFG_DIR)
	circt-opt --hw-print-instance-graph $(BUILD_DIR)/$(DESIGN)_hw.mlir > /dev/null 2> $(CFG_DIR)/$(DESIGN)_inst.dot
	dot -Tpng $(CFG_DIR)/$(DESIGN)_inst.dot -o $(CFG_DIR)/$(DESIGN)_inst.png
	circt-cf-opt --pass-pipeline='builtin.module(moore.module(moore-export-process-cfg{output-dir=$(CFG_DIR)}))' $(BUILD_DIR)/$(DESIGN)_moore.mlir > /dev/null 2>&1

clean:
	rm -rf $(BUILD_DIR)
