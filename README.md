# circt-cfa-trace

Prototype CIRCT-based control-flow analysis and trace instrumentation tool that
demonstrates how to build and ship custom passes out-of-tree. The repository
mirrors the high-level layout of `circt-cf-trace` with dedicated directories
for libraries, tools, CMake helper
modules, and regression tests.

## Prerequisites

This project depends on an existing LLVM/MLIR/CIRCT build. The commands below
assume you already built CIRCT from the `~/circt-cf-trace` checkout provided in
the environment and that Ninja is available on `PATH`.

If you use a different checkout or install prefix, adjust the paths in the
configuration examples accordingly.

## Configure & Build

Configure the project against the in-tree build of LLVM/MLIR/CIRCT:

```bash
cmake -S ~/circt-cfa-trace -B ~/circt-cfa-trace/build -G Ninja \
  -DLLVM_DIR=~/circt/llvm/build/lib/cmake/llvm \
  -DMLIR_DIR=~/circt/llvm/build/lib/cmake/mlir \
  -DCIRCT_DIR=~/circt/build/lib/cmake/circt
```

Then build the libraries and tool:

```bash
ninja -C ~/circt-cfa-trace/build
```

## Tests

The repository comes with a small lit suite that exercises the instrumentation
pass. Run it after each build:

```bash
ninja -C ~/circt-cfa-trace/build check-circt-cfa-trace
```

## Running the Tool

The main executable lands in `~/circt-cfa-trace/build/bin/circt-cfa-trace`.
Invoke it on MLIR files containing `hw.module` ops to stamp them with the
`hw.probe` attribute:

```bash
~/circt-cfa-trace/build/bin/circt-cfa-trace \
  ~/circt-cfa-trace/test/instrumentation/insert-probe.mlir
```

Pass `--emit-bytecode` to emit MLIR bytecode instead of textual MLIR, or redirect
stdout to capture the instrumented module.

## Extending

- Add new passes under `lib/Instrumentation/` with associated TableGen entries.
- Register additional command-line flags or pipeline wiring in
  `tools/circt-cfa-trace/circt-cfa-trace.cpp`.
- Add regression tests in `test/` and hook them into the lit suite.
