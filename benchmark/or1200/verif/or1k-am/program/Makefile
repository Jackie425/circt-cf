################################################################################
# OR1K Program Compilation Makefile
# 
# This Makefile compiles C programs for the OR1200 processor
################################################################################

# Tools
OR1K_GCC = or1k-elf-gcc
OR1K_OBJCOPY = or1k-elf-objcopy
OR1K_OBJDUMP = or1k-elf-objdump
OR1K_SIZE = or1k-elf-size

# Directories
BUILD_DIR = build

# Files
PROGRAM_SRC = program.c
STARTUP_SRC = ../crt0.S
LINKER_SCRIPT = ../or1200.ld

# Output files
PROGRAM_NAME = program
ELF_FILE = $(BUILD_DIR)/$(PROGRAM_NAME).elf
IMEM_HEX_FILE = $(BUILD_DIR)/$(PROGRAM_NAME)_imem.hex
DMEM_HEX_FILE = $(BUILD_DIR)/$(PROGRAM_NAME)_dmem.hex
DIS_FILE = $(BUILD_DIR)/$(PROGRAM_NAME).dis

# Compiler flags
CFLAGS = -nostartfiles -nostdlib -nodefaultlibs
CFLAGS += -ffreestanding -fno-builtin
CFLAGS += -O1
CFLAGS += -Wl,-T,$(LINKER_SCRIPT)

################################################################################
# Targets
################################################################################

.PHONY: all clean help

all: $(IMEM_HEX_FILE) $(DMEM_HEX_FILE)

# Create build directory
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Compile C to ELF
$(ELF_FILE): $(PROGRAM_SRC) $(STARTUP_SRC) $(LINKER_SCRIPT) | $(BUILD_DIR)
	@echo "==========================================="
	@echo "Compiling OR1K program..."
	@echo "==========================================="
	@echo "Startup: $(STARTUP_SRC)"
	@echo "Source:  $(PROGRAM_SRC)"
	@echo "Linker:  $(LINKER_SCRIPT)"
	@echo "Output:  $@"
	@echo ""
	$(OR1K_GCC) $(CFLAGS) $(STARTUP_SRC) $< -o $@
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
	@echo "First 30 lines:"
	@head -30 $@
	@echo ""

# Compile and show disassembly
dis: $(DIS_FILE)

# Clean build artifacts
clean:
	@echo "Cleaning OR1K build artifacts..."
	@rm -rf $(BUILD_DIR)

# Help
help:
	@echo "OR1K Program Compilation Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make        - Compile program to hex"
	@echo "  make dis    - Also generate disassembly"
	@echo "  make clean  - Remove build artifacts"
	@echo "  make help   - Show this help"
	@echo ""
	@echo "Files:"
	@echo "  Source:  $(PROGRAM_SRC)"
	@echo "  Linker:  $(LINKER_SCRIPT)"
	@echo "  Output:  $(HEX_FILE)"
