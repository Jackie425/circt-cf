# OR1200 Verilator Verification Environment

Functional verification testbench for CIRCT-generated OR1200 SystemVerilog using Verilator simulation.

## Overview

This verification environment validates the correctness of CIRCT source-to-source transformed OR1200 RTL by running real software workloads on a cycle-accurate Verilator model.

**Key Features:**
- Harvard architecture with separate instruction/data memories (64KB each)
- Automated software compilation and simulation
- Program exit detection via magic values
- Optional VCD waveform generation
- Includes simple test and CoreMark benchmark

## Quick Start

### Prerequisites
- Verilator (v5.0+)
- `or1k-elf-gcc` cross-compiler toolchain
- Python 3 (for hex file conversion)
- GTKWave (optional, for waveform viewing)

### Basic Usage

```bash
# Quick test with simple program (~6K cycles)
make

# Run CoreMark benchmark without trace (~250M cycles, faster)
make PROG=coremark TRACE=0

# Run with waveform generation (slower, large VCD file)
make PROG=coremark TRACE=1

# View waveform
make wave
```

## Directory Structure

```
verif/or1200/
├── Makefile              # Top-level build orchestration
├── sim-main.cpp          # Verilator C++ testbench driver
├── tb/
│   └── or1200_tb_top.sv  # SystemVerilog testbench wrapper
├── or1k-am/              # Software compilation environment
│   ├── or1200.ld         # Linker script (Harvard architecture)
│   ├── crt0.S            # Startup code (reset vector @ 0x100)
│   ├── program/          # Simple test program
│   │   ├── program.c     # Basic arithmetic test
│   │   └── Makefile
│   └── coremark/         # CoreMark embedded benchmark
│       └── Makefile
├── scripts/
│   └── hex_to_readmemh.py  # Convert objcopy output to Verilog format
└── build/                # Generated files (Verilator output, VCD, hex)
```

**RTL Source:** `../../benchmark/or1200/build/or1200.sv` (CIRCT-generated)

## Memory Architecture

OR1200 uses **Harvard architecture** with separate instruction and data memory spaces:

```
IMEM (Instruction Memory):  0x00000000 - 0x0000FFFF (64KB)
  - .text section starts at 0x100 (CPU reset vector)
  - .rodata (read-only data)

DMEM (Data Memory):         0x00000000 - 0x0000FFFF (64KB)
  - .data section starts at 0x0
  - .bss (uninitialized data)
  - Stack grows from 0x10000 downward
```

## Software Compilation Flow

1. **Compile**: `or1k-elf-gcc` → ELF executable
2. **Extract Sections**: `objcopy` → separate IMEM/DMEM raw binary + Intel hex
3. **Convert Format**: `hex_to_readmemh.py` → Verilog `$readmemh` compatible
4. **Load**: Testbench initializes memories from hex files

### Simple Test Program
Located in `or1k-am/program/program.c`:
- Basic arithmetic operations (add, subtract, multiply)
- Store results to memory addresses 0x0-0x6
- Loop 100 iterations
- **Runtime:** ~6,200 cycles (~62 μs @ 100MHz)

### CoreMark Benchmark
Located in `or1k-am/coremark/`:
- Standard embedded performance benchmark
- Tests: list processing, matrix operations, state machines, CRC
- **Configuration:** ITERATIONS=1, PERFORMANCE_RUN=1
- **Runtime:** ~250M cycles (~2.5 seconds @ 100MHz)

## Exit Detection Mechanism

Programs signal completion by writing magic values to address `0xFFFF0000`:

| Value        | Meaning        | Exit Code |
|--------------|----------------|-----------|
| `0xDEADBEEF` | Success        | 0         |
| `0xABADBABE` | Abort/Failure  | 1         |

Example in C:
```c
volatile unsigned int *exit_signal = (volatile unsigned int *)0xFFFF0000;
*exit_signal = 0xDEADBEEF;  // Signal success
```


## Makefile Targets

| Target           | Description                                      |
|------------------|--------------------------------------------------|
| `make`           | Build and run simple program (default)           |
| `make PROG=coremark` | Build and run CoreMark benchmark             |
| `make TRACE=0`   | Disable VCD waveform (10x faster simulation)     |
| `make build-program` | Compile simple program only                  |
| `make build-coremark` | Compile CoreMark only                       |
| `make sim`       | Build Verilator simulator only                   |
| `make wave`      | Open VCD in GTKWave                              |
| `make dis`       | Show disassembly of current program              |
| `make clean`     | Clean build artifacts                            |
| `make clean-all` | Clean everything including Verilator build       |
| `make info`      | Show current configuration                       |
| `make help`      | Display help message                             |

## Trace Control

The `TRACE` variable controls VCD waveform generation:

```bash
# With trace (default) - slower, generates large VCD file
make PROG=coremark TRACE=1

# Without trace - 10x faster, no VCD file
make PROG=coremark TRACE=0
```

**Performance Impact:**
- With trace: ~1-2M cycles/second
- Without trace: ~10-20M cycles/second

## Simulation Details

### Clock and Reset
- **Clock Period:** 10ns (100 MHz)
- **Reset Duration:** 5 clock cycles
- **Reset Vector:** 0x100 (where `.text` section begins)

### Testbench Monitoring
The C++ driver (`sim-main.cpp`) monitors:
- Data Wishbone writes to `0xFFFF0000` for exit signals
- Progress updates every 1,000 cycles
- Maximum simulation time: 500M cycles (safety timeout)

### Wishbone Bus Protocol
OR1200 uses Wishbone B3 Classic with single-cycle memory responses:
- **IWB (Instruction Wishbone):** Read-only, fetches from IMEM
- **DWB (Data Wishbone):** Read/write, accesses DMEM

## Debugging

### View Waveform
```bash
make PROG=program TRACE=1
make wave  # Opens GTKWave
```

**Key Signals:**
- `or1200_tb_top.iwb_*` - Instruction bus transactions
- `or1200_tb_top.dwb_*` - Data bus transactions
- `or1200_tb_top.or1200_inst.or1200_cpu.*` - CPU internals

### View Disassembly
```bash
make dis PROG=program
# or
make dis PROG=coremark
```

### Check Program Memory Layout
```bash
or1k-elf-objdump -h or1k-am/coremark/build/coremark.elf
or1k-elf-nm or1k-am/coremark/build/coremark.elf | sort
```

## Integration with CIRCT Pipeline

This verification environment is part of the end-to-end CIRCT transformation flow:

```
1. Original RTL       → benchmark/or1200/or1200/rtl/*.v
2. CIRCT Transform    → circt-cfa-trace (Moore → HW → SV dialects)
3. Generated SV       → benchmark/or1200/build/or1200.sv
4. Verification       → verif/or1200/ (this directory)
```

**Typical Workflow:**
```bash
# Step 1: Generate instrumented SV from original RTL
cd ~/circt-cfa-trace/benchmark/or1200
make run

# Step 2: Verify functional correctness
cd ~/circt-cfa-trace/verif/or1200
make PROG=coremark TRACE=0
```

## Expected Results

### Simple Program
```
Starting OR1200 simulation...
[50] Reset released, processor starting...
========================================
PROGRAM COMPLETED SUCCESSFULLY
========================================
Exit signal detected at time 62070 (cycle 6207)
Total cycles: 6207
```

### CoreMark (1 iteration)
```
Starting OR1200 simulation...
[50] Reset released, processor starting...
========================================
PROGRAM COMPLETED SUCCESSFULLY
========================================
Exit signal detected at time ~2500000000 (cycle ~250000000)
Total cycles: ~250000000
CoreMark Score: ~0.4 CoreMark/MHz @ 100MHz
```

## Troubleshooting

**Issue:** `No rule to make target '../../benchmark/or1200/build/or1200.sv'`
- **Solution:** Run `make run` in `benchmark/or1200/` first to generate SV

**Issue:** `or1k-elf-gcc: command not found`
- **Solution:** Install OpenRISC toolchain or add to PATH

**Issue:** Simulation timeout after 500M cycles
- **Solution:** Check if program has infinite loop, review disassembly

**Issue:** VCD file too large (>10GB)
- **Solution:** Use `TRACE=0` or reduce `--trace-depth` in Makefile

## Performance Notes

- **CoreMark with TRACE=1:** ~2-3 hours real time for 250M cycles
- **CoreMark with TRACE=0:** ~15-20 minutes real time for 250M cycles
- **VCD file size (TRACE=1):** ~100MB per 1M cycles

For CI/CD pipelines, use `TRACE=0` for faster validation.

## References

- [CIRCT Project](https://circt.llvm.org/)
- [OpenRISC 1000 Architecture](https://openrisc.io/architecture)
- [Verilator Documentation](https://verilator.org/guide/latest/)
- [CoreMark Benchmark](https://www.eembc.org/coremark/)
- [Wishbone B3 Specification](https://opencores.org/howto/wishbone)
