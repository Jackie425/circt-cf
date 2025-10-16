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
	circt-cfa-trace --verbose-pass-executions $(DEBUG_OPTS) --ir-moore -C $(DESIGN).f -o $(BUILD_DIR)/$(DESIGN)_moore.mlir
	circt-cfa-trace --mlir-pretty-debuginfo --verbose-pass-executions $(DEBUG_OPTS) --ir-hw -C $(DESIGN).f -o $(BUILD_DIR)/$(DESIGN)_hw.mlir
	circt-opt $(DEBUG_OPTS) \
		'--pass-pipeline=builtin.module(lower-seq-firmem,lower-seq-to-sv{disable-mem-randomization disable-reg-randomization},hw-memory-sim{disable-mem-randomization disable-reg-randomization},hw.module(lower-hw-to-sv),test-apply-lowering-options{options=disallowLocalVariables},hw.module(hw-cleanup),export-split-verilog{dir-name=$(BUILD_DIR)/export-or1200-sv})' \
		$(BUILD_DIR)/$(DESIGN)_hw.mlir -o $(BUILD_DIR)/$(DESIGN)_sv.mlir

graph:
	circt-opt --hw-print-instance-graph $(BUILD_DIR)/$(DESIGN)_hw.mlir 2> $(BUILD_DIR)/$(DESIGN)_hw.dot
	dot -Tpng $(BUILD_DIR)/$(DESIGN)_hw.dot -o $(BUILD_DIR)/$(DESIGN)_hw.png

clean:
	rm -rf $(BUILD_DIR)