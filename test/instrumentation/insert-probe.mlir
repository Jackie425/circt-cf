// RUN: %circt-cf-opt --insert-hw-probes %s | FileCheck %s

hw.module @Foo() {
  hw.output
}

hw.module @Bar(in %in: i1, out out: i1) {
  %0 = hw.wire %in : i1
  hw.output %0 : i1
}

// CHECK: hw.module @Foo() attributes {hw.probes}
// CHECK: hw.module @Bar(in %in : i1, out out : i1) attributes {hw.probes}
