# pcov

Prototype CIRCT-based control-flow analysis and trace instrumentation tool that
demonstrates how to build and ship custom passes out-of-tree. The repository
mirrors the high-level layout used by CIRCT with dedicated directories
for libraries, tools, CMake helper
modules, and regression tests.

## Prerequisites

This project depends on an existing LLVM/MLIR/CIRCT build and Ninja on `PATH`.
Define two variables (adjust as needed):

```bash
export CIRCT_DIR=/path/to/circt      # CIRCT repository root
export PCOV_DIR=/path/to/pcov        # this repository root
```

## Configure & Build

Configure the project against the in-tree build of LLVM/MLIR/CIRCT:

```bash
cmake -S "$PCOV_DIR" -B "$PCOV_DIR/build" -G Ninja \
  -DLLVM_DIR="$CIRCT_DIR/llvm/build/lib/cmake/llvm" \
  -DMLIR_DIR="$CIRCT_DIR/llvm/build/lib/cmake/mlir" \
  -DCIRCT_DIR="$CIRCT_DIR/build/lib/cmake/circt"
```

Then build the libraries and tool:

```bash
ninja -C "$PCOV_DIR/build"
```

## Tests

The repository comes with a small lit suite that exercises the instrumentation
pass. Run it after each build:

```bash
ninja -C "$PCOV_DIR/build" check-pcov
```

## Running the Tool

The main executable lands in `$PCOV_DIR/build/bin/pcov`.
Invoke it on MLIR files containing `hw.module` ops to stamp them with the
`hw.probe` attribute:

```bash
"$PCOV_DIR/build/bin/pcov" \
  "$PCOV_DIR/test/instrumentation/insert-probe.mlir"
```

Pass `--emit-bytecode` to emit MLIR bytecode instead of textual MLIR, or redirect
stdout to capture the instrumented module.

## Extending

- Add new passes under `lib/Instrumentation/` with associated TableGen entries.
- Register additional command-line flags or pipeline wiring in
  `tools/pcov/pcov.cpp`.
- Add regression tests in `test/` and hook them into the lit suite.
