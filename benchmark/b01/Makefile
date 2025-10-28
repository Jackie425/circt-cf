# simple Makefile to test a CIRCT flow from SystemVerilog to MLIR to SystemVerilog
DESIGN ?= b01
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
	circt-cf --mlir-pretty-debuginfo --verbose-pass-executions $(DEBUG_OPTS) --ir-moore $(DESIGN).v -o $(BUILD_DIR)/$(DESIGN)_moore.mlir
	circt-cf --mlir-pretty-debuginfo --verbose-pass-executions $(DEBUG_OPTS) --ir-hw $(DESIGN).v -o $(BUILD_DIR)/$(DESIGN)_hw.mlir
	circt-cf --mlir-pretty-debuginfo --verbose-pass-executions $(DEBUG_OPTS) --ir-sv $(DESIGN).v -o $(BUILD_DIR)/$(DESIGN)_sv.mlir
	circt-cf --mlir-pretty-debuginfo --verbose-pass-executions $(DEBUG_OPTS) --sv-sv $(DESIGN).v -o $(BUILD_DIR)/$(DESIGN).sv

run-split:
	mkdir -p $(BUILD_DIR)/split-output
	circt-cf --verbose-pass-executions $(DEBUG_OPTS) --sv-sv --split-verilog --split-verilog-dir=$(BUILD_DIR)/split-output $(DESIGN).v

graph:
	circt-opt --hw-print-instance-graph $(BUILD_DIR)/$(DESIGN)_hw.mlir 2> $(BUILD_DIR)/$(DESIGN)_hw.dot
	dot -Tpng $(BUILD_DIR)/$(DESIGN)_hw.dot -o $(BUILD_DIR)/$(DESIGN)_hw.png

clean:
	rm -rf $(BUILD_DIR)