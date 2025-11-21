// Include directories
+incdir+cva6/core/include/
+incdir+cva6/vendor/pulp-platform/common_cells/include/
+incdir+cva6/vendor/pulp-platform/common_cells/src/
+incdir+cva6/vendor/pulp-platform/axi/include/
+incdir+cva6/common/local/util/
+incdir+cva6/core/cache_subsystem/hpdcache/rtl/include/

// Defines
+define+SYNTHESIS
+define+COMMON_CELLS_ASSERTS_OFF

// FPGA memories (needed by cache/frontend SRAMs)
cva6/vendor/pulp-platform/fpga-support/rtl/SyncDpRam.sv
cva6/vendor/pulp-platform/fpga-support/rtl/AsyncDpRam.sv
cva6/vendor/pulp-platform/fpga-support/rtl/AsyncThreePortRam.sv
cva6/vendor/pulp-platform/fpga-support/rtl/SyncThreePortRam.sv
cva6/vendor/pulp-platform/fpga-support/rtl/SyncDpRam_ind_r_w.sv

// Floating point unit
cva6/core/cvfpu/src/fpnew_pkg.sv
cva6/core/cvfpu/src/fpnew_cast_multi.sv
cva6/core/cvfpu/src/fpnew_classifier.sv
cva6/core/cvfpu/vendor/opene906/E906_RTL_FACTORY/gen_rtl/clk/rtl/gated_clk_cell.v
cva6/core/cvfpu/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_ctrl.v
cva6/core/cvfpu/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_ff1.v
cva6/core/cvfpu/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_pack_single.v
cva6/core/cvfpu/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_prepare.v
cva6/core/cvfpu/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_round_single.v
cva6/core/cvfpu/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_special.v
cva6/core/cvfpu/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_srt_single.v
cva6/core/cvfpu/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_top.v
cva6/core/cvfpu/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fpu/rtl/pa_fpu_dp.v
cva6/core/cvfpu/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fpu/rtl/pa_fpu_frbus.v
cva6/core/cvfpu/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fpu/rtl/pa_fpu_src_type.v
cva6/core/cvfpu/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_ctrl.v
cva6/core/cvfpu/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_double.v
cva6/core/cvfpu/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_ff1.v
cva6/core/cvfpu/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_pack.v
cva6/core/cvfpu/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_prepare.v
cva6/core/cvfpu/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_round.v
cva6/core/cvfpu/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_scalar_dp.v
cva6/core/cvfpu/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_srt_radix16_bound_table.v
cva6/core/cvfpu/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_srt_radix16_with_sqrt.v
cva6/core/cvfpu/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_srt.v
cva6/core/cvfpu/vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_top.v
cva6/core/cvfpu/src/fpnew_divsqrt_th_32.sv
cva6/core/cvfpu/src/fpnew_divsqrt_th_64_multi.sv
cva6/core/cvfpu/src/fpnew_divsqrt_multi.sv
cva6/core/cvfpu/src/fpnew_fma_multi.sv
cva6/core/cvfpu/src/fpnew_fma.sv
cva6/core/cvfpu/src/fpnew_noncomp.sv
cva6/core/cvfpu/src/fpnew_opgroup_block.sv
cva6/core/cvfpu/src/fpnew_opgroup_fmt_slice.sv
cva6/core/cvfpu/src/fpnew_opgroup_multifmt_slice.sv
cva6/core/cvfpu/src/fpnew_rounding.sv
cva6/core/cvfpu/src/fpnew_top.sv
cva6/core/cvfpu/src/fpu_div_sqrt_mvp/hdl/defs_div_sqrt_mvp.sv
cva6/core/cvfpu/src/fpu_div_sqrt_mvp/hdl/control_mvp.sv
cva6/core/cvfpu/src/fpu_div_sqrt_mvp/hdl/div_sqrt_top_mvp.sv
cva6/core/cvfpu/src/fpu_div_sqrt_mvp/hdl/iteration_div_sqrt_mvp.sv
cva6/core/cvfpu/src/fpu_div_sqrt_mvp/hdl/norm_div_sqrt_mvp.sv
cva6/core/cvfpu/src/fpu_div_sqrt_mvp/hdl/nrbd_nrsc_mvp.sv
cva6/core/cvfpu/src/fpu_div_sqrt_mvp/hdl/preprocess_mvp.sv

cva6/core/include/config_pkg.sv
cva6/core/include/cv32a6_ima_sv32_fpga_config_pkg.sv
cva6/core/include/riscv_pkg.sv
// Note: depends on fpnew_pkg, above
cva6/core/include/ariane_pkg.sv
cva6/vendor/pulp-platform/axi/src/axi_pkg.sv

// Packages
cva6/core/include/wt_cache_pkg.sv
cva6/core/include/std_cache_pkg.sv
cva6/core/include/instr_tracer_pkg.sv
cva6/core/include/build_config_pkg.sv
cva6/core/include/aes_pkg.sv
cva6/core/include/triggers_pkg.sv

// CVXIF core-side shims (no example coprocessor)
cva6/core/cvxif_compressed_if_driver.sv
cva6/core/cvxif_issue_register_commit_if_driver.sv
cva6/core/cvxif_fu.sv

// Common Cells
cva6/vendor/pulp-platform/common_cells/src/cf_math_pkg.sv
cva6/vendor/pulp-platform/common_cells/src/fifo_v3.sv
cva6/vendor/pulp-platform/common_cells/src/lfsr.sv
cva6/vendor/pulp-platform/common_cells/src/lfsr_8bit.sv
cva6/vendor/pulp-platform/common_cells/src/stream_arbiter.sv
cva6/vendor/pulp-platform/common_cells/src/stream_arbiter_flushable.sv
cva6/vendor/pulp-platform/common_cells/src/stream_mux.sv
cva6/vendor/pulp-platform/common_cells/src/stream_demux.sv
cva6/vendor/pulp-platform/common_cells/src/lzc.sv
cva6/vendor/pulp-platform/common_cells/src/rr_arb_tree.sv
cva6/vendor/pulp-platform/common_cells/src/shift_reg.sv
cva6/vendor/pulp-platform/common_cells/src/unread.sv
cva6/vendor/pulp-platform/common_cells/src/popcount.sv
cva6/vendor/pulp-platform/common_cells/src/exp_backoff.sv

// Common Cells for example coprocessor
cva6/vendor/pulp-platform/common_cells/src/counter.sv
cva6/vendor/pulp-platform/common_cells/src/delta_counter.sv

// Top-level source files (not necessarily instantiated at the top of the cva6).
cva6/core/cva6.sv
cva6/core/cva6_rvfi_probes.sv
cva6/core/alu.sv
cva6/core/alu_wrapper.sv
cva6/core/aes.sv
// Note: depends on fpnew_pkg, above
cva6/core/fpu_wrap.sv
cva6/core/branch_unit.sv
cva6/core/compressed_decoder.sv
cva6/core/macro_decoder.sv
cva6/core/controller.sv
cva6/core/zcmt_decoder.sv
cva6/core/csr_buffer.sv
cva6/core/csr_regfile.sv
cva6/core/trigger_module.sv
overrides/decoder_pcov.sv
cva6/core/ex_stage.sv
cva6/core/instr_realign.sv
cva6/core/id_stage.sv
cva6/core/issue_read_operands.sv
cva6/core/issue_stage.sv
cva6/core/load_unit.sv
cva6/core/load_store_unit.sv
cva6/core/lsu_bypass.sv
cva6/core/mult.sv
cva6/core/multiplier.sv
cva6/core/serdiv.sv
cva6/core/perf_counters.sv
cva6/core/ariane_regfile_ff.sv
cva6/core/ariane_regfile_fpga.sv
// NOTE: scoreboard.sv modified for DSIM (unchanged for other simulators)
cva6/core/scoreboard.sv
cva6/core/raw_checker.sv
cva6/core/store_buffer.sv
cva6/core/amo_buffer.sv
cva6/core/store_unit.sv
cva6/core/commit_stage.sv
cva6/core/axi_shim.sv
cva6/core/cva6_accel_first_pass_decoder_stub.sv
cva6/core/acc_dispatcher.sv
cva6/core/cva6_fifo_v3.sv

// What is "frontend"?
cva6/core/frontend/btb.sv
cva6/core/frontend/bht.sv
cva6/core/frontend/bht2lvl.sv
cva6/core/frontend/ras.sv
cva6/core/frontend/instr_scan.sv
cva6/core/frontend/instr_queue.sv
cva6/core/frontend/frontend.sv

// Cache subsystem
cva6/core/cache_subsystem/wt_dcache_ctrl.sv
cva6/core/cache_subsystem/wt_dcache_mem.sv
cva6/core/cache_subsystem/wt_dcache_missunit.sv
cva6/core/cache_subsystem/wt_dcache_wbuffer.sv
cva6/core/cache_subsystem/wt_dcache.sv
cva6/core/cache_subsystem/cva6_icache.sv
cva6/core/cache_subsystem/wt_cache_subsystem.sv
cva6/core/cache_subsystem/wt_axi_adapter.sv
cva6/core/cache_subsystem/tag_cmp.sv
cva6/core/cache_subsystem/axi_adapter.sv
cva6/core/cache_subsystem/miss_handler.sv
cva6/core/cache_subsystem/cache_ctrl.sv
cva6/core/cache_subsystem/cva6_icache_axi_wrapper.sv
cva6/core/cache_subsystem/std_cache_subsystem.sv
cva6/core/cache_subsystem/std_nbdcache.sv

// Physical Memory Protection
// NOTE: pmp.sv modified for DSIM (unchanged for other simulators)
cva6/core/pmp/src/pmp.sv
cva6/core/pmp/src/pmp_entry.sv
cva6/core/pmp/src/pmp_data_if.sv

// Tracer (behavioral code, not RTL)
cva6/common/local/util/instr_tracer.sv
cva6/common/local/util/tc_sram_wrapper.sv
cva6/common/local/util/tc_sram_wrapper_cache_techno.sv
cva6/vendor/pulp-platform/tech_cells_generic/src/rtl/tc_sram.sv
cva6/common/local/util/sram.sv
cva6/common/local/util/sram_cache.sv

// MMU 
cva6/core/cva6_mmu/cva6_mmu.sv
cva6/core/cva6_mmu/cva6_ptw.sv
cva6/core/cva6_mmu/cva6_tlb.sv
cva6/core/cva6_mmu/cva6_shared_tlb.sv

// end of manifest
