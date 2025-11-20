# Simple entry points to exercise the pcov pipeline on or1200.
DESIGN ?= or1200
BUILD_DIR := build
CFG_DIR := $(BUILD_DIR)/cfg
SYNTH_DIR = $(BUILD_DIR)/synth

PCOV ?= pcov
PCOV_OPT ?= pcov-opt

DEBUG_OPTS ?=

CFG_ARGS := --emit-moore-cfg --emit-moore-cfg-dir=$(CFG_DIR)

######################################################################
.PHONY: default run run-base run-split clean

default: run

run:
	mkdir -p $(BUILD_DIR) $(CFG_DIR)
	$(PCOV) --ir-moore $(DEBUG_OPTS) $(CFG_ARGS) \
	  --mlir-print-ir-before=moore-instrument-coverage \
	  -C $(DESIGN).f -o $(BUILD_DIR)/$(DESIGN)_moore_pcov.mlir 2> $(BUILD_DIR)/$(DESIGN)_moore.mlir
	$(PCOV) --mlir-pretty-debuginfo --verbose-pass-executions \
	  $(DEBUG_OPTS) $(CFG_ARGS) --ir-hw -C $(DESIGN).f \
	  -o $(BUILD_DIR)/$(DESIGN)_hw.mlir
	$(PCOV) --mlir-pretty-debuginfo --verbose-pass-executions \
	  $(DEBUG_OPTS) $(CFG_ARGS) --ir-sv -C $(DESIGN).f \
	  -o $(BUILD_DIR)/$(DESIGN)_sv.mlir
	$(PCOV) --mlir-pretty-debuginfo --verbose-pass-executions \
	  $(DEBUG_OPTS) $(CFG_ARGS) --sv-sv -C $(DESIGN).f \
	  -o $(BUILD_DIR)/$(DESIGN).sv

run-base:
	mkdir -p $(BUILD_DIR)
	$(PCOV) --mlir-pretty-debuginfo --verbose-pass-executions \
	  $(DEBUG_OPTS) --disable-pcov-instrumentation --sv-sv -C $(DESIGN).f \
	  -o $(BUILD_DIR)/$(DESIGN)_base.sv

run-split:
	mkdir -p $(BUILD_DIR)/split-output $(CFG_DIR)
	$(PCOV) --verbose-pass-executions $(DEBUG_OPTS) $(CFG_ARGS) --sv-sv \
	  --split-verilog --split-verilog-dir=$(BUILD_DIR)/split-output \
	  -C $(DESIGN).f

synth: run run-base
	mkdir -p $(SYNTH_DIR)
	yosys -Q -s synth.ys

clean:
	rm -rf $(BUILD_DIR)
