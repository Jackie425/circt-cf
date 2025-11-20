# Simple entry points to exercise the pcov pipeline on mor1kx.
DESIGN ?= mor1kx
BUILD_DIR := build
CFG_DIR := $(BUILD_DIR)/cfg

PCOV ?= pcov
PCOV_OPT ?= pcov-opt

DEBUG_OPTS ?=
SYNTH_DEF ?= -DSYNTHESIS
CFG_ARGS := --emit-moore-cfg --emit-moore-cfg-dir=$(CFG_DIR)

######################################################################
.PHONY: default run run-split clean

default: run

run:
	mkdir -p $(BUILD_DIR) $(CFG_DIR)
	$(PCOV) $(SYNTH_DEF) --ir-moore $(DEBUG_OPTS) $(CFG_ARGS) \
	  --mlir-print-ir-before=moore-instrument-coverage \
	  -C $(DESIGN).f -o $(BUILD_DIR)/$(DESIGN)_moore_pcov.mlir 2> $(BUILD_DIR)/$(DESIGN)_moore.mlir
	$(PCOV) $(SYNTH_DEF) --mlir-pretty-debuginfo --verbose-pass-executions \
	  $(DEBUG_OPTS) $(CFG_ARGS) --ir-hw -C $(DESIGN).f \
	  -o $(BUILD_DIR)/$(DESIGN)_hw.mlir
	$(PCOV) $(SYNTH_DEF) --mlir-pretty-debuginfo --verbose-pass-executions \
	  $(DEBUG_OPTS) $(CFG_ARGS) --ir-sv -C $(DESIGN).f \
	  -o $(BUILD_DIR)/$(DESIGN)_sv.mlir
	$(PCOV) $(SYNTH_DEF) --mlir-pretty-debuginfo --verbose-pass-executions \
	  $(DEBUG_OPTS) $(CFG_ARGS) --sv-sv -C $(DESIGN).f \
	  -o $(BUILD_DIR)/$(DESIGN).sv

run-split:
	mkdir -p $(BUILD_DIR)/split-output $(CFG_DIR)
	$(PCOV) --verbose-pass-executions $(SYNTH_DEF) $(DEBUG_OPTS) $(CFG_ARGS) --sv-sv \
	  --split-verilog --split-verilog-dir=$(BUILD_DIR)/split-output \
	  -C $(DESIGN).f

clean:
	rm -rf $(BUILD_DIR)
