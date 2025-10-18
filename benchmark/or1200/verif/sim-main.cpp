#include "Vor1200_tb_top.h"
#include "Vor1200_tb_top_or1200_tb_top.h"
#include "verilated.h"

#ifdef TRACE_ENABLED
#include "verilated_vcd_c.h"
#endif

// Program exit detection magic values
#define EXIT_MAGIC_ADDR    0xFFFF0000
#define EXIT_SUCCESS_VALUE 0xDEADBEEF
#define EXIT_ABORT_VALUE   0xABADBABE

int main(int argc, char** argv) {

    Verilated::commandArgs(argc, argv);

    Vor1200_tb_top* top = new Vor1200_tb_top;

#ifdef TRACE_ENABLED
    Verilated::traceEverOn(true);
    VerilatedVcdC* tfp = new VerilatedVcdC;
    top->trace(tfp, 99);
    tfp->open("or1200_trace.vcd");  // Will be created in current directory (build/)
    printf("Trace enabled: waveform will be saved to or1200_trace.vcd\n");
#else
    printf("Trace disabled: running without waveform generation (faster)\n");
#endif

    top->clk = 0;
    top->rst = 1;

    vluint64_t main_time = 0;
    const vluint64_t max_sim_time = 5000000000;  // Maximum simulation time (safety limit)
    
    bool program_finished = false;
    int exit_code = 0;
    vluint64_t finish_time = 0;

    printf("Starting OR1200 simulation...\n");
    printf("Reset will be released at time 50\n");
    printf("Monitoring for program exit signal at address 0x%08X\n", EXIT_MAGIC_ADDR);
    printf("  - 0x%08X = Normal exit\n", EXIT_SUCCESS_VALUE);
    printf("  - 0x%08X = Abort\n", EXIT_ABORT_VALUE);

    while (!Verilated::gotFinish() && main_time < max_sim_time && !program_finished) {
        // Toggle clock every 5 time units (10ns period = 100MHz)
        if ((main_time % 10) == 5) {
            top->clk = 0;
        }
        else if ((main_time % 10) == 0) {
            top->clk = 1;

            // Release reset after 50 time units (5 clock cycles)
            if (main_time == 50) {
                top->rst = 0;
                printf("[%lu] Reset released, processor starting...\n", main_time);
            }
            
            // Print progress every 1000 cycles
            if (main_time > 0 && (main_time % 10000) == 0) {
                printf("[%lu] Simulation running... (cycle %lu)\n", 
                       main_time, main_time / 10);
            }
            
            // Check for program exit signal on data bus write
            // Access public signals marked in testbench
            if (top->or1200_tb_top->dwb_cyc && top->or1200_tb_top->dwb_stb && top->or1200_tb_top->dwb_we) {
                
                uint32_t addr = top->or1200_tb_top->dwb_adr;
                uint32_t data = top->or1200_tb_top->dwb_dat_o;
                
                // Check if writing to exit signal address
                if (addr == EXIT_MAGIC_ADDR) {
                    if (data == EXIT_SUCCESS_VALUE) {
                        printf("\n");
                        printf("========================================\n");
                        printf("PROGRAM COMPLETED SUCCESSFULLY\n");
                        printf("========================================\n");
                        printf("Exit signal detected at time %lu (cycle %lu)\n", 
                               main_time, main_time / 10);
                        printf("Exit value: 0x%08X\n", data);
                        program_finished = true;
                        exit_code = 0;
                        finish_time = main_time;
                    } else if (data == EXIT_ABORT_VALUE) {
                        printf("\n");
                        printf("========================================\n");
                        printf("PROGRAM ABORTED\n");
                        printf("========================================\n");
                        printf("Abort signal detected at time %lu (cycle %lu)\n", 
                               main_time, main_time / 10);
                        printf("Abort value: 0x%08X\n", data);
                        program_finished = true;
                        exit_code = 1;
                        finish_time = main_time;
                    }
                }
            }
        }
        
        top->eval();
#ifdef TRACE_ENABLED
        tfp->dump(main_time);
#endif
        main_time++;
    }

    // Print summary
    printf("\n");
    if (program_finished) {
        printf("========================================\n");
        printf("Simulation Summary\n");
        printf("========================================\n");
        printf("Status:      %s\n", exit_code == 0 ? "SUCCESS" : "ABORTED");
        printf("End time:    %lu ns (cycle %lu)\n", finish_time, finish_time / 10);
        printf("Total cycles: %lu\n", finish_time / 10);
    } else if (main_time >= max_sim_time) {
        printf("========================================\n");
        printf("WARNING: Simulation timeout\n");
        printf("========================================\n");
        printf("Program did not exit within %lu cycles\n", max_sim_time / 10);
        printf("This may indicate an infinite loop or\n");
        printf("the program is still running normally.\n");
        exit_code = 2;
    } else {
        printf("Simulation stopped at time %lu\n", main_time);
    }
    
    printf("========================================\n");
#ifdef TRACE_ENABLED
    printf("Waveform saved to: or1200_trace.vcd\n");
    printf("View with: gtkwave or1200_trace.vcd\n");
#else
    printf("No waveform generated (TRACE=0)\n");
    printf("To enable trace: make TRACE=1 ...\n");
#endif
    printf("========================================\n");

#ifdef TRACE_ENABLED
    tfp->close();
    delete tfp;
#endif
    delete top;

    return exit_code;
}