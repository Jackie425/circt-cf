# circt-cf

Prototype CIRCT-based control-flow analysis and trace instrumentation tool that
demonstrates how to build and ship custom passes out-of-tree. The repository
mirrors the high-level layout used by CIRCT with dedicated directories
for libraries, tools, CMake helper
modules, and regression tests.

## Prerequisites

This project depends on an existing LLVM/MLIR/CIRCT build. The commands below
assume you already built CIRCT from the `~/circt-cf` checkout provided in
the environment and that Ninja is available on `PATH`.

If you use a different checkout or install prefix, adjust the paths in the
configuration examples accordingly.

## Configure & Build

Configure the project against the in-tree build of LLVM/MLIR/CIRCT:

```bash
cmake -S ~/circt-cf -B ~/circt-cf/build -G Ninja \
  -DLLVM_DIR=~/circt/llvm/build/lib/cmake/llvm \
  -DMLIR_DIR=~/circt/llvm/build/lib/cmake/mlir \
  -DCIRCT_DIR=~/circt/build/lib/cmake/circt
```

Then build the libraries and tool:

```bash
ninja -C ~/circt-cf/build
```

## Tests

The repository comes with a small lit suite that exercises the instrumentation
pass. Run it after each build:

```bash
ninja -C ~/circt-cf/build check-circt-cf
```

## Running the Tool

The main executable lands in `~/circt-cf/build/bin/circt-cf`.
Invoke it on MLIR files containing `hw.module` ops to stamp them with the
`hw.probe` attribute:

```bash
~/circt-cf/build/bin/circt-cf \
  ~/circt-cf/test/instrumentation/insert-probe.mlir
```

Pass `--emit-bytecode` to emit MLIR bytecode instead of textual MLIR, or redirect
stdout to capture the instrumented module.

## Extending

- Add new passes under `lib/Instrumentation/` with associated TableGen entries.
- Register additional command-line flags or pipeline wiring in
  `tools/circt-cf/circt-cf.cpp`.
- Add regression tests in `test/` and hook them into the lit suite.
