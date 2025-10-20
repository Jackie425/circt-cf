# OR1200 Verilator Verification Environment

Cycle-accurate Verilator test bench for the CIRCT-generated OR1200 core.  
It compiles bare‑metal OR1K programs, loads them into a simple Harvard memory model, and traps software termination via a Wishbone write.

---

## Requirements

- Verilator (v5.x recommended)
- OR1K cross toolchain (`or1k-elf-gcc`, `or1k-elf-objcopy`, `or1k-elf-objdump`)
- GTKWave (optional, for viewing traces)

Ensure all tools are on your `PATH` before running `make`.

---

## Directory Layout

```
verif/
├── Makefile                  # Top-level build & run orchestration
├── sim-main.cpp              # Verilator C++ harness (exit detection, tracing)
├── tb/
│   └── or1200_tb_top.sv      # SystemVerilog test bench (64 KiB IMEM/DMEM model)
├── or1k-am/
│   ├── crt0.S                # Startup + exit/abort handoff
│   ├── or1200.ld             # Harvard-layout linker script (IMEM/DMEM @ 0)
│   ├── program/              # Simple regression program
│   └── coremark/             # CoreMark port with OR1K configuration
└── build/                    # Verilator obj_dir/ and optional waveform dumps
```

---

## Quick Start

```bash
# Build and run the default program (writes test data, exits cleanly)
make

# Run the CoreMark benchmark without waveform dumping
make PROG=coremark TRACE=0

# Build the simulator executable without running it
make sim

# Clean generated program artifacts and simulator build
make clean        # or clean-sim / clean-all
```

- `TRACE=1` (default) enables VCD dumping; `TRACE=0` disables tracing for speed.
- The simulator binary is produced under `build/obj_dir/Vor1200_tb_top`.
- Software ELFs live under `or1k-am/<prog>/build/<prog>.elf` and are loaded directly by the simulator.

---

## Execution Flow

1. `or1k-elf-gcc` compiles the selected program (`program` or `coremark`) using
   the provided `crt0.S` and `or1200.ld`, producing a standalone ELF binary.
2. Verilator builds and links `sim-main.cpp` with the OR1200 test bench and generated RTL.
3. During reset, the C++ harness parses the ELF, loads executable segments into IMEM, and
   writes writable segments into DMEM (zero-filling `.bss`).
4. The harness toggles the clock, watches for Wishbone writes to `0xFFFF0000`, and exits with:
   - `0xDEADBEEF` → program success (return code 0)
   - `0xABADBABE` → abort/failure (return code 1)

`crt0.S`’s `_exit` writes the success magic when `main` returns `0`; non-zero return values are routed through `abort`.

---

## Useful Targets & Flags

| Target / Flag           | Description                                                  |
|-------------------------|--------------------------------------------------------------|
| `run` (default)         | Build software ELF, build Verilator model, run once         |
| `sim`                   | Build the Verilator executable without running it           |
| `info`                  | Print resolved paths and currently selected program         |
| `dis`                   | Show disassembly via `less`                                 |
| `wave`                  | Launch GTKWave on the latest trace (requires `TRACE=1`)      |
| `clean`, `clean-sim`, `clean-all` | Remove software, simulator, or all build products |
| `TRACE=0/1`             | Disable/enable VCD dumping                                   |
| `PROG=program/coremark` | Select the software payload                                  |

To enable detailed bus logging during a manual run:

```bash
cd build/obj_dir
./Vor1200_tb_top +tb_verbose
```

---

## Test Bench Notes

- IMEM/DMEM depth: 64 KiB each (word-accessed via Wishbone).
- No caches or wait-state modelling; every transaction completes in a single cycle.
- Write strobes and partial writes are supported on DMEM.
- Instruction fetch outside the 0x00000000–0x0000FFFF range defaults to `l.nop`.

---

## Debug Tips

- Use `make dis PROG=<prog>` to inspect the generated assembly; check the tail of `main` to confirm it returns via `_exit`.
- If the simulator reports an abort exit, verify your program does not return non-zero or explicitly call `abort()`.
- When experimenting with different programs, ensure their data pointers do not alias the null address—`crt0.S` and the linker script expect IMEM/DMEM to start at 0, but the compiler must not see it as a null pointer.

---

With these pieces, you can compile new OR1K workloads, run them under Verilator, and capture waveforms or disassembly as needed.
