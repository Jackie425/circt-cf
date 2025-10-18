################################################################################
# OR1200 Verilator Simulation - Top Level Makefile
# 
# This Makefile coordinates the OR1K program compilation and Verilator simulation
#
# Directory Structure:
#   or1k-am/          - OR1K program compilation
#   tb/               - Testbench files
#   build/            - All build artifacts
#   build/obj_dir/    - Verilator generated files
#
# Usage:
#   make PROG=program              - Build and run with simple program (default)
#   make PROG=coremark             - Build and run with CoreMark benchmark
#   make PROG=coremark TRACE=0     - Run CoreMark without trace (faster, smaller)
#   make build-program             - Only compile program
#   make build-coremark            - Only compile CoreMark
#   make sim                       - Only build Verilator simulator
#   make run PROG=program          - Run simulation with specified program
#   make wave                      - Open waveform viewer
#   make clean                     - Clean all build artifacts
#   make clean-all                 - Clean everything including Verilator build
#   make help                      - Show help
#
# Variables:
#   PROG               - Program to run: 'program' or 'coremark' (default: program)
#   TRACE              - Enable waveform trace: 1=yes, 0=no (default: 1)
################################################################################

# Tools
VERILATOR = verilator
GTKWAVE = gtkwave

# Program selection (default: program)
PROG ?= program

# Trace control (default: 1 = enable trace, 0 = disable trace)
TRACE ?= 1

# Validate PROG variable
ifeq ($(filter $(PROG),program coremark),)
    $(error Invalid PROG value: $(PROG). Must be 'program' or 'coremark')
endif

# Directories
OR1K_DIR = or1k-am
TB_DIR = tb
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj_dir
SCRIPTS_DIR = scripts

# Program-specific directories
PROGRAM_DIR = $(OR1K_DIR)/program
COREMARK_DIR = $(OR1K_DIR)/coremark

# Source files
RTL_SOURCES = ../build/or1200.sv $(TB_DIR)/or1200_tb_top.sv
CPP_SOURCES = sim-main.cpp

# Scripts
HEX_CONVERTER = $(SCRIPTS_DIR)/hex_to_readmemh.py

# Program-specific files (based on PROG variable)
ifeq ($(PROG),program)
    PROG_BUILD_DIR = $(PROGRAM_DIR)/build
    PROG_IMEM_HEX = $(PROG_BUILD_DIR)/program_imem.hex
    PROG_DMEM_HEX = $(PROG_BUILD_DIR)/program_dmem.hex
    PROG_DIS_FILE = $(PROG_BUILD_DIR)/program.dis
    PROG_NAME = Simple Program
else ifeq ($(PROG),coremark)
    PROG_BUILD_DIR = $(COREMARK_DIR)/build
    PROG_IMEM_HEX = $(PROG_BUILD_DIR)/coremark_imem.hex
    PROG_DMEM_HEX = $(PROG_BUILD_DIR)/coremark_dmem.hex
    PROG_DIS_FILE = $(PROG_BUILD_DIR)/coremark.dis
    PROG_NAME = CoreMark Benchmark
endif

# Output files in top-level build directory
BUILD_IMEM_HEX = $(BUILD_DIR)/$(PROG)_imem.hex
BUILD_DMEM_HEX = $(BUILD_DIR)/$(PROG)_dmem.hex
TRACE_FILE = $(BUILD_DIR)/or1200_trace.vcd
SIM_EXE = $(OBJ_DIR)/Vor1200_tb_top

# Verilator flags
VERILATOR_FLAGS = -cc --exe
VERILATOR_FLAGS += -x-assign fast
VERILATOR_FLAGS += -Wall
VERILATOR_FLAGS += -Wno-UNOPTFLAT -Wno-BLKANDNBLK -Wno-CASEINCOMPLETE -Wno-CMPCONST
VERILATOR_FLAGS += -Wno-WIDTH -Wno-WIDTHCONCAT -Wno-UNSIGNED -Wno-LITENDIAN
VERILATOR_FLAGS += -Wno-UNUSEDSIGNAL -Wno-DECLFILENAME -Wno-PINCONNECTEMPTY
VERILATOR_FLAGS += -Wno-SYNCASYNCNET

# Add trace flags if TRACE=1
ifeq ($(TRACE),1)
VERILATOR_FLAGS += --trace --trace-depth 10 --trace-structs --trace-max-array 1024
VERILATOR_FLAGS += -CFLAGS -DTRACE_ENABLED
endif

VERILATOR_FLAGS += --build -j 0
VERILATOR_FLAGS += --top-module or1200_tb_top
VERILATOR_FLAGS += -Mdir $(OBJ_DIR)
VERILATOR_FLAGS += -CFLAGS -I$(shell pwd)
VERILATOR_FLAGS += -DPROGRAM_IMEM_HEX=\"../$(PROG)_imem.hex\"
VERILATOR_FLAGS += -DPROGRAM_DMEM_HEX=\"../$(PROG)_dmem.hex\"

################################################################################
# Targets
################################################################################

.PHONY: default all build-program build-coremark sim run wave dis clean clean-sim clean-all help info

default: all

# Build everything and run with selected program
all: build-$(PROG) sim run

# Compile simple program
build-program:
	@echo "========================================"
	@echo "Compiling Simple Program"
	@echo "========================================"
	@$(MAKE) -C $(PROGRAM_DIR)
	@echo ""

# Compile CoreMark
build-coremark:
	@echo "========================================"
	@echo "Compiling CoreMark Benchmark"
	@echo "========================================"
	@$(MAKE) -C $(COREMARK_DIR)
	@echo ""

# Build Verilator simulator
sim: $(SIM_EXE)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Info about current configuration
info:
	@echo "========================================"
	@echo "Current Configuration"
	@echo "========================================"
	@echo "Selected Program:  $(PROG)"
	@echo "Program Name:      $(PROG_NAME)"
	@echo "Trace Enabled:     $(TRACE) (0=no VCD, 1=generate VCD)"
	@echo "Source IMEM HEX:   $(PROG_IMEM_HEX)"
	@echo "Source DMEM HEX:   $(PROG_DMEM_HEX)"
	@echo "Target IMEM HEX:   $(BUILD_IMEM_HEX)"
	@echo "Target DMEM HEX:   $(BUILD_DMEM_HEX)"
	@echo "Disassembly:       $(PROG_DIS_FILE)"
	@echo "========================================"

# Convert hex files to $readmemh format and copy to build directory
$(BUILD_IMEM_HEX): $(PROG_IMEM_HEX) | $(BUILD_DIR)
	@echo "Converting IMEM hex file to $readmemh format..."
	@python3 $(HEX_CONVERTER) $(PROG_IMEM_HEX) $(BUILD_IMEM_HEX)
	@echo ""

$(BUILD_DMEM_HEX): $(PROG_DMEM_HEX) | $(BUILD_DIR)
	@echo "Converting DMEM hex file to $readmemh format..."
	@if [ -s $(PROG_DMEM_HEX) ]; then \
		python3 $(HEX_CONVERTER) $(PROG_DMEM_HEX) $(BUILD_DMEM_HEX); \
	else \
		echo "DMEM hex file is empty, creating empty readmemh file..."; \
		echo "@00000000" > $(BUILD_DMEM_HEX); \
	fi
	@echo ""

$(SIM_EXE): $(RTL_SOURCES) $(CPP_SOURCES) $(BUILD_IMEM_HEX) $(BUILD_DMEM_HEX) | $(BUILD_DIR)
	@echo "========================================"
	@echo "Building Verilator Simulator"
	@echo "========================================"
	@echo "Program:     $(PROG_NAME) ($(PROG))"
	@echo "RTL sources: $(RTL_SOURCES)"
	@echo "C++ sources: $(CPP_SOURCES)"
	@echo ""
	@echo "Running Verilator compilation..."
	$(VERILATOR) $(VERILATOR_FLAGS) $(RTL_SOURCES) $(shell pwd)/$(CPP_SOURCES)
	@echo ""


# Run simulation
run: $(SIM_EXE)
	@echo "========================================"
	@echo "Running Simulation"
	@echo "========================================"
	@echo "Program:         $(PROG_NAME) ($(PROG))"
	@echo "Source HEX:      $(PROG_INTEL_HEX)"
	@echo "Simulator HEX:   $(PROGRAM_IMEM_HEX)"
	@echo ""
	@cd $(OBJ_DIR) && ./Vor1200_tb_top
	@echo ""
	@if [ -f $(OBJ_DIR)/or1200_trace.vcd ]; then \
		echo "Moving VCD trace to build directory..."; \
		mv $(OBJ_DIR)/or1200_trace.vcd $(BUILD_DIR)/; \
	fi
	@echo ""
	@echo "========================================"
	@echo "Simulation Complete!"
	@echo "========================================"
	@echo "Program:       $(PROG_NAME) ($(PROG))"
	@echo ""
	@echo "Outputs:"
	@echo "  Disassembly: $(PROG_DIS_FILE)"
	@echo "  Waveform:    $(TRACE_FILE)"
	@echo ""
	@echo "Next steps:"
	@echo "  make wave PROG=$(PROG)    - View waveform"
	@echo "  make dis PROG=$(PROG)     - View disassembly"
	@echo "========================================"

# Show disassembly
dis:
	@if [ ! -f $(PROG_DIS_FILE) ]; then \
		echo "Generating disassembly for $(PROG)..."; \
		$(MAKE) -C $(PROG_BUILD_DIR)/.. dis; \
	fi
	@echo "Disassembly for $(PROG_NAME):"
	@echo "File: $(PROG_DIS_FILE)"
	@echo ""
	@less $(PROG_DIS_FILE)

# Open waveform viewer
wave: $(TRACE_FILE)
	@echo "Opening waveform viewer..."
	$(GTKWAVE) $(TRACE_FILE) &

# Clean program build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@$(MAKE) -C $(PROGRAM_DIR) clean
	@$(MAKE) -C $(COREMARK_DIR) clean
	@rm -f $(BUILD_DIR)/*_imem.hex $(TRACE_FILE)
	@echo "Done."

# Clean Verilator build only
clean-sim:
	@echo "Cleaning Verilator build..."
	@rm -rf $(OBJ_DIR)

# Clean everything
clean-all: clean clean-sim
	@echo "Cleaning all build artifacts..."
	@rm -rf $(BUILD_DIR)
	@echo "Done."

# Help
help:
	@echo "========================================"
	@echo "OR1200 Verilator Simulation Makefile"
	@echo "========================================"
	@echo ""
	@echo "PROGRAM SELECTION:"
	@echo "  Use PROG=<name> to select program"
	@echo "  Available programs:"
	@echo "    program   - Simple test program (default)"
	@echo "    coremark  - CoreMark benchmark"
	@echo ""
	@echo "TRACE CONTROL:"
	@echo "  Use TRACE=<0|1> to enable/disable waveform generation"
	@echo "    TRACE=1   - Enable trace (default, slower, large files)"
	@echo "    TRACE=0   - Disable trace (faster, no VCD file)"
	@echo ""
	@echo "EXAMPLES:"
	@echo "  make PROG=program                - Build and run simple program with trace"
	@echo "  make PROG=coremark TRACE=0       - Run CoreMark without trace (recommended)"
	@echo "  make PROG=coremark TRACE=1       - Run CoreMark with trace (very large file)"
	@echo "  make build-coremark              - Only compile CoreMark"
	@echo "  make run PROG=coremark TRACE=0   - Run simulation without trace"
	@echo ""
	@echo "MAIN TARGETS:"
	@echo "  make [PROG=<name>] [TRACE=<0|1>] - Build and run (default: program, trace=1)"
	@echo "  make all PROG=<name> TRACE=<0|1> - Same as make"
	@echo ""
	@echo "BUILD TARGETS:"
	@echo "  make build-program     - Compile simple program"
	@echo "  make build-coremark    - Compile CoreMark benchmark"
	@echo "  make sim PROG=<name> TRACE=<0|1> - Build Verilator simulator"
	@echo ""
	@echo "RUN TARGETS:"
	@echo "  make run PROG=<name>   - Run simulation with program"
	@echo ""
	@echo "UTILITY TARGETS:"
	@echo "  make info PROG=<name>  - Show current configuration"
	@echo "  make wave              - Open waveform in GTKWave (if TRACE=1)"
	@echo "  make dis PROG=<name>   - View program disassembly"
	@echo ""
	@echo "CLEAN TARGETS:"
	@echo "  make clean             - Clean program artifacts"
	@echo "  make clean-sim         - Clean Verilator build"
	@echo "  make clean-all         - Clean everything"
	@echo ""
	@echo "DIRECTORY STRUCTURE:"
	@echo "  or1k-am/program/       - Simple program source"
	@echo "  or1k-am/coremark/      - CoreMark source"
	@echo "  tb/                    - Testbench files"
	@echo "  build/                 - Build artifacts"
	@echo ""
	@echo "FILES:"
	@echo "  Programs:"
	@echo "    or1k-am/program/program.c"
	@echo "    or1k-am/coremark/*.c"
	@echo "  Testbench: tb/or1200_tb_top.sv"
	@echo "  RTL:       or1200.sv"
	@echo ""
	@echo "========================================"