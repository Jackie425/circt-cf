# Simple entry points to exercise the CIRCT-CF pipeline on b06.
DESIGN ?= b06
BUILD_DIR := build
CFG_DIR := $(BUILD_DIR)/cfg
SYNTH_DIR = $(BUILD_DIR)/synth

CIRCT_CF ?= circt-cf
CIRCT_CF_OPT ?= circt-cf-opt

DEBUG_OPTS ?=

CFG_ARGS := --emit-moore-cfg --emit-moore-cfg-dir=$(CFG_DIR)

######################################################################
.PHONY: default run run-split clean

default: run

run:
	mkdir -p $(BUILD_DIR) $(CFG_DIR)
	$(CIRCT_CF) --ir-moore $(DEBUG_OPTS) $(CFG_ARGS) \
	  --mlir-print-ir-before=moore-instrument-coverage \
	  $(DESIGN).v -o $(BUILD_DIR)/$(DESIGN)_moore_pcov.mlir 2> $(BUILD_DIR)/$(DESIGN)_moore.mlir
	$(CIRCT_CF) --mlir-pretty-debuginfo --verbose-pass-executions \
	  $(DEBUG_OPTS) $(CFG_ARGS) --ir-hw $(DESIGN).v \
	  -o $(BUILD_DIR)/$(DESIGN)_hw.mlir
	$(CIRCT_CF) --mlir-pretty-debuginfo --verbose-pass-executions \
	  $(DEBUG_OPTS) $(CFG_ARGS) --ir-sv $(DESIGN).v \
	  -o $(BUILD_DIR)/$(DESIGN)_sv.mlir
	$(CIRCT_CF) --mlir-pretty-debuginfo --verbose-pass-executions \
	  $(DEBUG_OPTS) $(CFG_ARGS) --sv-sv $(DESIGN).v \
	  -o $(BUILD_DIR)/$(DESIGN).sv

run-split:
	mkdir -p $(BUILD_DIR)/split-output $(CFG_DIR)
	$(CIRCT_CF) --verbose-pass-executions $(DEBUG_OPTS) $(CFG_ARGS) --sv-sv \
	  --split-verilog --split-verilog-dir=$(BUILD_DIR)/split-output \
	  $(DESIGN).v

synth: run
	mkdir -p $(SYNTH_DIR)
	yosys -Q -s synth.ys

clean:
	rm -rf $(BUILD_DIR)
