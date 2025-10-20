# SPDX-License-Identifier: Apache-2.0

.DEFAULT_GOAL := all
.SUFFIXES:
.PHONY: all clean dis

OR1K_GCC     ?= or1k-elf-gcc
OR1K_OBJDUMP ?= or1k-elf-objdump
OR1K_SIZE    ?= or1k-elf-size

BUILD_DIR      := build
PROGRAM_NAME   := coremark
SRCS := core_list_join.c \
        core_main.c \
        core_matrix.c \
        core_state.c \
        core_util.c \
        core_portme.c
STARTUP_SRC    := ../crt0.S
LINKER_SCRIPT  := ../or1200.ld

ELF_FILE       := $(BUILD_DIR)/$(PROGRAM_NAME).elf
DIS_FILE       := $(BUILD_DIR)/$(PROGRAM_NAME).dis

CFLAGS  := -nostartfiles -nostdlib -nodefaultlibs -ffreestanding -fno-builtin
CFLAGS  += -O2
CFLAGS  += -DPERFORMANCE_RUN=1 -DITERATIONS=10 -DTOTAL_DATA_SIZE=2000
LDFLAGS := -Wl,-T,$(LINKER_SCRIPT)

all: $(ELF_FILE)

$(BUILD_DIR):
	@mkdir -p $@

$(ELF_FILE): $(SRCS) $(STARTUP_SRC) $(LINKER_SCRIPT) | $(BUILD_DIR)
	$(OR1K_GCC) $(CFLAGS) $(STARTUP_SRC) $(SRCS) -o $@ $(LDFLAGS)
	$(OR1K_SIZE) $@

$(DIS_FILE): $(ELF_FILE)
	$(OR1K_OBJDUMP) -D $< > $@

dis: $(DIS_FILE)
	@less $<

clean:
	@rm -rf $(BUILD_DIR)
