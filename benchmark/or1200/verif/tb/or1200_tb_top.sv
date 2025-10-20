// OR1200 Testbench Top Module
// This wraps the OR1200 and provides simple memory models

module or1200_tb_top (
    input logic clk,
    input logic rst
);

    localparam bit TB_VERBOSE_DEFAULT = 1'b0;
    bit verbose_enable;

    initial begin
        verbose_enable = TB_VERBOSE_DEFAULT;
        if ($test$plusargs("tb_verbose"))
            verbose_enable = 1'b1;
    end
    // Internal signals
    logic [19:0] pic_ints;
    logic [1:0]  clmode;
    
    // Instruction Wishbone signals
    logic        iwb_cyc, iwb_stb, iwb_we, iwb_ack, iwb_err, iwb_rty;
    logic [31:0] iwb_adr, iwb_dat_i, iwb_dat_o;
    logic [3:0]  iwb_sel;
    logic [2:0]  iwb_cti;
    logic [1:0]  iwb_bte;
    
    // Data Wishbone signals
    // Mark as public for Verilator C++ access (program exit detection)
    logic        dwb_cyc /* verilator public */;
    logic        dwb_stb /* verilator public */;
    logic        dwb_we  /* verilator public */;
    logic        dwb_ack, dwb_err, dwb_rty;
    logic [31:0] dwb_adr /* verilator public */;
    logic [31:0] dwb_dat_i;
    logic [31:0] dwb_dat_o /* verilator public */;
    logic [3:0]  dwb_sel;
    logic [2:0]  dwb_cti;
    logic [1:0]  dwb_bte;
    
    // Debug signals
    logic        dbg_stall, dbg_ewt, dbg_stb, dbg_we, dbg_ack;
    logic [31:0] dbg_adr, dbg_dat_i, dbg_dat_o;
    logic [3:0]  dbg_lss;
    logic [1:0]  dbg_is;
    logic [10:0] dbg_wp;
    logic        dbg_bp;
    
    // Power management
    logic        pm_cpustall;
    logic [3:0]  pm_clksd;
    logic        pm_dc_gate, pm_ic_gate, pm_dmmu_gate, pm_immu_gate;
    logic        pm_tt_gate, pm_cpu_gate, pm_wakeup, pm_lvolt;
    logic        sig_tick;

    // Initialize control signals
    initial begin
        pic_ints = 20'h0;
        clmode = 2'b00;
        dbg_stall = 1'b0;
        dbg_ewt = 1'b0;
        dbg_stb = 1'b0;
        dbg_we = 1'b0;
        dbg_adr = 32'h0;
        dbg_dat_i = 32'h0;
        pm_cpustall = 1'b0;
    end

    // Simple instruction memory (initialized with NOP instructions)
    logic [31:0] imem [0:16383] /* verilator public_flat */;  // 64KB = 16384 words
    
    initial begin
        // Initialize all memory with NOP instructions (l.nop 0x0)
        // OpenRISC NOP encoding: 0x15000000
        for (int i = 0; i < 16384; i++) begin
            imem[i] = 32'h15000000;
        end
    end
    
    // Instruction Wishbone slave (memory model)
    wire        iwb_req = iwb_cyc && iwb_stb;
    wire [15:0] iwb_addr_hi = iwb_adr[31:16];
    wire [13:0] iwb_word_idx = iwb_adr[15:2];
    logic [31:0] imem_read_data;

    always_comb begin
        if (iwb_addr_hi == 16'h0000) begin
            imem_read_data = imem[iwb_word_idx];
        end else begin
            imem_read_data = 32'h15000000;  // default NOP on out-of-range fetch
        end
    end

    assign iwb_dat_i = imem_read_data;

    always_comb begin
        iwb_ack = iwb_req;
        iwb_err = 1'b0;
        iwb_rty = 1'b0;
    end
    
    // Simple data memory
    logic [31:0] dmem [0:16383] /* verilator public_flat */;  // 64KB = 16384 words
    
    initial begin
        // Initialize all data memory to zero
        for (int i = 0; i < 16384; i++) begin
            dmem[i] = 32'h00000000;
        end
    end
    // (second zeroing pass removed; initial block above already handles init)
    
    // Data Wishbone slave (memory model)
    wire        dwb_req = dwb_cyc && dwb_stb;
    wire [15:0] dwb_addr_hi = dwb_adr[31:16];
    wire [13:0] dwb_word_idx = dwb_adr[15:2];
    logic [31:0] dmem_read_data;
    logic [31:0] dmem_write_data;
    wire         dmem_addr_valid = (dwb_addr_hi == 16'h0000);

    always_comb begin
        if (dmem_addr_valid) begin
            dmem_read_data = dmem[dwb_word_idx];
        end else begin
            dmem_read_data = 32'h00000000;
        end

        dmem_write_data = dmem_read_data;
        if (dwb_sel[0]) dmem_write_data[7:0]   = dwb_dat_o[7:0];
        if (dwb_sel[1]) dmem_write_data[15:8]  = dwb_dat_o[15:8];
        if (dwb_sel[2]) dmem_write_data[23:16] = dwb_dat_o[23:16];
        if (dwb_sel[3]) dmem_write_data[31:24] = dwb_dat_o[31:24];
    end

    assign dwb_dat_i = dmem_read_data;

    always_comb begin
        dwb_ack = dwb_req;
        dwb_err = 1'b0;
        dwb_rty = 1'b0;
    end

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            // nothing to do, memories already initialised
        end else if (dwb_req && dwb_we && dmem_addr_valid) begin
            dmem[dwb_word_idx] <= dmem_write_data;
        end
    end

    // OR1200 processor instance
    or1200_top or1200_inst (
        .clk_i(clk),
        .rst_i(rst),
        .pic_ints_i(pic_ints),
        .clmode_i(clmode),
        
        // Instruction Wishbone
        .iwb_clk_i(clk),
        .iwb_rst_i(rst),
        .iwb_ack_i(iwb_ack),
        .iwb_err_i(iwb_err),
        .iwb_rty_i(iwb_rty),
        .iwb_dat_i(iwb_dat_i),
        .iwb_cyc_o(iwb_cyc),
        .iwb_adr_o(iwb_adr),
        .iwb_stb_o(iwb_stb),
        .iwb_we_o(iwb_we),
        .iwb_sel_o(iwb_sel),
        .iwb_dat_o(iwb_dat_o),
        .iwb_cti_o(iwb_cti),
        .iwb_bte_o(iwb_bte),
        
        // Data Wishbone
        .dwb_clk_i(clk),
        .dwb_rst_i(rst),
        .dwb_ack_i(dwb_ack),
        .dwb_err_i(dwb_err),
        .dwb_rty_i(dwb_rty),
        .dwb_dat_i(dwb_dat_i),
        .dwb_cyc_o(dwb_cyc),
        .dwb_adr_o(dwb_adr),
        .dwb_stb_o(dwb_stb),
        .dwb_we_o(dwb_we),
        .dwb_sel_o(dwb_sel),
        .dwb_dat_o(dwb_dat_o),
        .dwb_cti_o(dwb_cti),
        .dwb_bte_o(dwb_bte),
        
        // Debug interface
        .dbg_stall_i(dbg_stall),
        .dbg_ewt_i(dbg_ewt),
        .dbg_lss_o(dbg_lss),
        .dbg_is_o(dbg_is),
        .dbg_wp_o(dbg_wp),
        .dbg_bp_o(dbg_bp),
        .dbg_stb_i(dbg_stb),
        .dbg_we_i(dbg_we),
        .dbg_adr_i(dbg_adr),
        .dbg_dat_i(dbg_dat_i),
        .dbg_dat_o(dbg_dat_o),
        .dbg_ack_o(dbg_ack),
        
        // Power management
        .pm_cpustall_i(pm_cpustall),
        .pm_clksd_o(pm_clksd),
        .pm_dc_gate_o(pm_dc_gate),
        .pm_ic_gate_o(pm_ic_gate),
        .pm_dmmu_gate_o(pm_dmmu_gate),
        .pm_immu_gate_o(pm_immu_gate),
        .pm_tt_gate_o(pm_tt_gate),
        .pm_cpu_gate_o(pm_cpu_gate),
        .pm_wakeup_o(pm_wakeup),
        .pm_lvolt_o(pm_lvolt),
        .sig_tick(sig_tick)
    );

    // Monitor for debugging
    int cycle_count = 0;
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            cycle_count <= 0;
        end else begin
            cycle_count <= cycle_count + 1;
            
            // Print instruction fetches
            if (verbose_enable && iwb_cyc && iwb_stb && iwb_ack) begin
                  $display("[%0t] Cycle %0d: IFETCH addr=0x%08h data=0x%08h", 
                          $time, cycle_count, iwb_adr, iwb_dat_i);
            end
            
            // Print data accesses
            if (verbose_enable && dwb_cyc && dwb_stb && dwb_ack) begin
                 if (dwb_we)
                     $display("[%0t] Cycle %0d: DWRITE addr=0x%08h data=0x%08h sel=0x%h", 
                             $time, cycle_count, dwb_adr, dwb_dat_o, dwb_sel);
                else
                    $display("[%0t] Cycle %0d: DREAD  addr=0x%08h data=0x%08h", 
                             $time, cycle_count, dwb_adr, dwb_dat_i);
            end
        end
    end

endmodule
