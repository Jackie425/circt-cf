# simple Makefile to test a CIRCT flow from SystemVerilog to MLIR to SystemVerilog
DESIGN ?= or1200
BUILD_DIR = build

# Debug options for LLVM_DEBUG output
# Use --debug to see all debug output, or --debug-only=<type> for specific types

DEBUG_OPTS ?=
# Example: make run DEBUG_OPTS="--debug-only=llhd-lower-processes"
# Example: make run DEBUG_OPTS="--debug"

######################################################################
default: run

run:
	mkdir -p $(BUILD_DIR)
	circt-cfa-trace --mlir-pretty-debuginfo --verbose-pass-executions $(DEBUG_OPTS) --ir-moore -C $(DESIGN).f -o $(BUILD_DIR)/$(DESIGN)_moore.mlir
	circt-cfa-trace --mlir-pretty-debuginfo --verbose-pass-executions $(DEBUG_OPTS) --ir-hw -C $(DESIGN).f -o $(BUILD_DIR)/$(DESIGN)_hw.mlir
	circt-cfa-trace --mlir-pretty-debuginfo --verbose-pass-executions $(DEBUG_OPTS) --ir-sv -C $(DESIGN).f -o $(BUILD_DIR)/$(DESIGN)_sv.mlir
	circt-cfa-trace --mlir-pretty-debuginfo --verbose-pass-executions $(DEBUG_OPTS) --sv-sv -C $(DESIGN).f -o $(BUILD_DIR)/$(DESIGN).sv

run-split:
	mkdir -p $(BUILD_DIR)/split-output
	circt-cfa-trace --verbose-pass-executions $(DEBUG_OPTS) --sv-sv --split-verilog --split-verilog-dir=$(BUILD_DIR)/split-output -C $(DESIGN).f

graph:
	circt-opt --hw-print-instance-graph $(BUILD_DIR)/$(DESIGN)_hw.mlir 2> $(BUILD_DIR)/$(DESIGN)_hw.dot
	dot -Tpng $(BUILD_DIR)/$(DESIGN)_hw.dot -o $(BUILD_DIR)/$(DESIGN)_hw.png

clean:
	rm -rf $(BUILD_DIR)