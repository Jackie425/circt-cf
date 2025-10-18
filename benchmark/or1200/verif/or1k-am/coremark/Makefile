################################################################################
# CoreMark for OR1200 Compilation Makefile
# 
# This Makefile compiles CoreMark benchmark for the OR1200 processor
################################################################################

# Tools
OR1K_GCC = or1k-elf-gcc
OR1K_OBJCOPY = or1k-elf-objcopy
OR1K_OBJDUMP = or1k-elf-objdump
OR1K_SIZE = or1k-elf-size

# Directories
BUILD_DIR = build
PARENT_DIR = ..

# Source files
SRCS = core_list_join.c \
       core_main.c \
       core_matrix.c \
       core_state.c \
       core_util.c \
       core_portme.c

# Startup code (assembly)
STARTUP_SRC = $(PARENT_DIR)/crt0.S

# Linker script
LINKER_SCRIPT = $(PARENT_DIR)/or1200.ld

# Output files
PROGRAM_NAME = coremark
ELF_FILE = $(BUILD_DIR)/$(PROGRAM_NAME).elf
IMEM_HEX_FILE = $(BUILD_DIR)/$(PROGRAM_NAME)_imem.hex
DMEM_HEX_FILE = $(BUILD_DIR)/$(PROGRAM_NAME)_dmem.hex
DIS_FILE = $(BUILD_DIR)/$(PROGRAM_NAME).dis

# Compiler flags
CFLAGS = -nostartfiles -nostdlib -nodefaultlibs
CFLAGS += -ffreestanding -fno-builtin
CFLAGS += -O2
CFLAGS += -Wl,-T,$(LINKER_SCRIPT)

# CoreMark configuration flags
CFLAGS += -DPERFORMANCE_RUN=1
CFLAGS += -DITERATIONS=1

################################################################################
# Targets
################################################################################

.PHONY: all clean help dis

all: $(IMEM_HEX_FILE) $(DMEM_HEX_FILE)
	@echo ""
	@echo "==========================================="
	@echo "CoreMark build complete!"
	@echo "==========================================="
	@echo "ELF file:  $(ELF_FILE)"
	@echo "IMEM hex:  $(IMEM_HEX_FILE)"
	@echo "DMEM hex:  $(DMEM_HEX_FILE)"
	@echo ""

# Create build directory
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Compile CoreMark to ELF
$(ELF_FILE): $(SRCS) $(STARTUP_SRC) $(LINKER_SCRIPT) | $(BUILD_DIR)
	@echo "==========================================="
	@echo "Compiling CoreMark for OR1200..."
	@echo "==========================================="
	@echo "Startup: $(STARTUP_SRC)"
	@echo "Sources: $(SRCS)"
	@echo "Linker:  $(LINKER_SCRIPT)"
	@echo "Output:  $@"
	@echo ""
	$(OR1K_GCC) $(CFLAGS) $(STARTUP_SRC) $(SRCS) -o $@
	@echo ""
	@echo "Program size:"
	@$(OR1K_SIZE) $@
	@echo ""

# Generate hex files from ELF - separate IMEM and DMEM
$(IMEM_HEX_FILE): $(ELF_FILE)
	@echo "==========================================="
	@echo "Generating instruction memory hex file..."
	@echo "==========================================="
	$(OR1K_OBJCOPY) -O verilog --only-section=.text $< $@
	@echo "Generated: $@"
	@echo ""

$(DMEM_HEX_FILE): $(ELF_FILE)
	@echo "==========================================="
	@echo "Generating data memory hex file..."
	@echo "==========================================="
	$(OR1K_OBJCOPY) -O verilog --only-section=.data --only-section=.bss $< $@
	@echo "Generated: $@"
	@echo ""

# Generate disassembly
$(DIS_FILE): $(ELF_FILE)
	@echo "==========================================="
	@echo "Generating disassembly..."
	@echo "==========================================="
	$(OR1K_OBJDUMP) -D $< > $@
	@echo "Generated: $@"
	@echo ""
	@echo "First 50 lines:"
	@head -50 $@
	@echo ""

# Compile and show disassembly
dis: $(DIS_FILE)

# Clean build artifacts
clean:
	@echo "Cleaning CoreMark build artifacts..."
	@rm -rf $(BUILD_DIR)

# Help
help:
	@echo "CoreMark for OR1200 Compilation Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make        - Compile CoreMark to hex"
	@echo "  make dis    - Also generate disassembly"
	@echo "  make clean  - Remove build artifacts"
	@echo "  make help   - Show this help"
	@echo ""
	@echo "Configuration:"
	@echo "  Performance run with 1000 iterations"
	@echo "  Optimization level: -O2"
	@echo "  Target: OR1200 processor (OR1K architecture)"
	@echo ""
	@echo "Output files:"
	@echo "  $(ELF_FILE)"
	@echo "  $(IMEM_HEX_FILE) - instruction memory"
	@echo "  $(DMEM_HEX_FILE) - data memory"
	@echo "  $(DIS_FILE) (optional, use 'make dis')"
