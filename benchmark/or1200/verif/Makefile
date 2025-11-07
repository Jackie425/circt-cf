################################################################################
# OR1200 Verilator Simulation
################################################################################

.DEFAULT_GOAL := run
.SUFFIXES:
.PHONY: all run sim info dis wave clean \
        build-% format-help

PROGS := program coremark
PROG ?= program

ifeq (,$(filter $(PROG),$(PROGS)))
$(error Unknown PROG '$(PROG)'. Choose from: $(PROGS))
endif

TRACE ?= 0

################################################################################
# Toolchain configuration
################################################################################

VERILATOR ?= verilator
GTKWAVE   ?= gtkwave

################################################################################
# Directory layout
################################################################################

ROOT_DIR       := $(CURDIR)
OR1K_DIR       := or1k-am
BUILD_DIR      := build
OBJ_DIR        := $(BUILD_DIR)/obj_dir
TB_DIR         := tb

PROG_DIR       := $(OR1K_DIR)/$(PROG)
PROG_BUILD_DIR := $(PROG_DIR)/build
PROG_ELF       := $(PROG_BUILD_DIR)/$(PROG).elf
PROG_ELF_ABS   := $(abspath $(PROG_ELF))
PROG_DIS       := $(PROG_BUILD_DIR)/$(PROG).dis

TRACE_FILE     := $(BUILD_DIR)/or1200_trace.vcd

RTL_SOURCES := ../build/or1200.sv $(TB_DIR)/or1200_tb_top.sv
CPP_SOURCES := sim-main.cpp
SIM_EXE     := $(OBJ_DIR)/Vor1200_tb_top

################################################################################
# Verilator flags
################################################################################

VERILATOR_FLAGS := \
	-cc --exe \
	-O3 \
	-x-assign fast \
	-x-initial fast \
	-Wall \
	-Wno-UNOPTFLAT \
	-Wno-BLKANDNBLK \
	-Wno-CASEINCOMPLETE \
	-Wno-CMPCONST \
	-Wno-WIDTH \
	-Wno-WIDTHCONCAT \
	-Wno-UNSIGNED \
	-Wno-LITENDIAN \
	-Wno-UNUSEDSIGNAL \
	-Wno-DECLFILENAME \
	-Wno-PINCONNECTEMPTY \
	-Wno-SYNCASYNCNET \
	--build -j 0 \
	--top-module or1200_tb_top \
	-Mdir $(OBJ_DIR) \
	-CFLAGS -I$(ROOT_DIR) \
	-CFLAGS -DPROGRAM_ELF=\"$(PROG_ELF_ABS)\"

ifeq ($(TRACE),1)
VERILATOR_FLAGS += --trace --trace-depth 10 --trace-structs --trace-max-array 1024
VERILATOR_FLAGS += -CFLAGS -DTRACE_ENABLED
endif

################################################################################
# User-facing targets
################################################################################

all: run

run: $(SIM_EXE)
	@printf "Running %s (trace=%s)\n" "$(PROG)" "$(TRACE)"
	@cd $(OBJ_DIR) && ./Vor1200_tb_top
ifeq ($(TRACE),1)
	@if [ -f $(OBJ_DIR)/or1200_trace.vcd ]; then \
		mv $(OBJ_DIR)/or1200_trace.vcd $(BUILD_DIR)/; \
	fi
endif

sim: $(SIM_EXE)

info:
	@printf "Program      : %s\n" "$(PROG)"
	@printf "Trace        : %s\n" "$(TRACE)"
	@printf "ELF          : %s\n" "$(PROG_ELF)"
	@printf "Simulator    : %s\n" "$(SIM_EXE)"

dis: $(PROG_DIS)
	@less $(PROG_DIS)

wave:
	@if [ ! -f $(TRACE_FILE) ]; then \
		printf "Trace file not found: %s\n" "$(TRACE_FILE)" >&2; \
		printf "Run with TRACE=1 first.\n" >&2; \
		exit 1; \
	fi
	@$(GTKWAVE) $(TRACE_FILE) &

clean:
	@$(MAKE) -C $(OR1K_DIR)/program clean
	@$(MAKE) -C $(OR1K_DIR)/coremark clean
	@rm -rf $(BUILD_DIR)

################################################################################
# Build graph
################################################################################

$(SIM_EXE): $(RTL_SOURCES) $(CPP_SOURCES) $(PROG_ELF) | $(OBJ_DIR)
	@$(VERILATOR) $(VERILATOR_FLAGS) $(RTL_SOURCES) $(abspath $(CPP_SOURCES))

$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(PROG_ELF) $(PROG_DIS): | $(PROG_DIR)
	@$(MAKE) -C $(PROG_DIR)

$(PROG_DIR):
	@true

build-%:
	@$(MAKE) -C $(OR1K_DIR)/$*

################################################################################
# Auxiliary targets
################################################################################

format-help:
	@printf "Targets:\n"
	@printf "  run (default)  - build SW ELF, build Verilator model, run\n"
	@printf "  sim            - build Verilator model\n"
	@printf "  info           - print resolved paths\n"
	@printf "  dis            - open disassembly for current program\n"
	@printf "  wave           - open GTKWave on latest trace\n"
	@printf "  clean*         - remove build products\n"
