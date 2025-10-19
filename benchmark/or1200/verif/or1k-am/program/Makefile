# SPDX-License-Identifier: Apache-2.0

.DEFAULT_GOAL := all
.SUFFIXES:
.PHONY: all clean dis

OR1K_GCC     ?= or1k-elf-gcc
OR1K_OBJCOPY ?= or1k-elf-objcopy
OR1K_OBJDUMP ?= or1k-elf-objdump
OR1K_SIZE    ?= or1k-elf-size

BUILD_DIR      := build
PROGRAM_NAME   := program
PROGRAM_SRC    := program.c
STARTUP_SRC    := ../crt0.S
LINKER_SCRIPT  := ../or1200.ld

ELF_FILE       := $(BUILD_DIR)/$(PROGRAM_NAME).elf
IMEM_HEX_FILE  := $(BUILD_DIR)/$(PROGRAM_NAME)_imem.hex
DMEM_HEX_FILE  := $(BUILD_DIR)/$(PROGRAM_NAME)_dmem.hex
DIS_FILE       := $(BUILD_DIR)/$(PROGRAM_NAME).dis

CFLAGS  := -nostartfiles -nostdlib -nodefaultlibs -ffreestanding -fno-builtin
CFLAGS  += -O2
LDFLAGS := -Wl,-T,$(LINKER_SCRIPT)

all: $(IMEM_HEX_FILE) $(DMEM_HEX_FILE)

$(BUILD_DIR):
	@mkdir -p $@

$(ELF_FILE): $(PROGRAM_SRC) $(STARTUP_SRC) $(LINKER_SCRIPT) | $(BUILD_DIR)
	$(OR1K_GCC) $(CFLAGS) $(STARTUP_SRC) $(PROGRAM_SRC) -o $@ $(LDFLAGS)
	$(OR1K_SIZE) $@

$(IMEM_HEX_FILE): $(ELF_FILE)
	$(OR1K_OBJCOPY) -O verilog --only-section=.text $< $@

$(DMEM_HEX_FILE): $(ELF_FILE)
	$(OR1K_OBJCOPY) -O verilog --only-section=.data --only-section=.bss $< $@

$(DIS_FILE): $(ELF_FILE)
	$(OR1K_OBJDUMP) -D $< > $@

dis: $(DIS_FILE)
	@less $<

clean:
	@rm -rf $(BUILD_DIR)

