#include <cstdint>
#include <cstdio>
#include <memory>
#include <inttypes.h>

#include "Vor1200_tb_top.h"
#include "Vor1200_tb_top_or1200_tb_top.h"
#include "verilated.h"

#ifdef TRACE_ENABLED
#include "verilated_vcd_c.h"
#endif

namespace {

constexpr std::uint32_t kExitMagicAddr    = 0xFFFF0000u;
constexpr std::uint32_t kExitSuccessValue = 0xDEADBEEFu;
constexpr std::uint32_t kExitAbortValue   = 0xABADBABEu;

constexpr vluint64_t kHalfPeriodPs      = 5;      // 10 ns full period => 100 MHz
constexpr vluint64_t kResetReleaseTime  = 50;     // 5 cycles
constexpr vluint64_t kProgressInterval  = 1'000'000'000ULL;  // every 100M cycles
constexpr vluint64_t kMaxSimTime        = 500'000'000'000'000ULL;

}  // namespace

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);

    auto top = std::make_unique<Vor1200_tb_top>();
#ifdef TRACE_ENABLED
    Verilated::traceEverOn(true);
    auto trace = std::make_unique<VerilatedVcdC>();
    top->trace(trace.get(), 99);
    trace->open("or1200_trace.vcd");
#endif

    top->clk = 0;
    top->rst = 1;

    bool program_finished = false;
    int exit_code = 0;
    vluint64_t finish_time = 0;

    vluint64_t main_time = 0;
    std::puts("Starting OR1200 simulation");

    while (!Verilated::gotFinish() && main_time < kMaxSimTime && !program_finished) {
        if ((main_time % (2 * kHalfPeriodPs)) == kHalfPeriodPs) {
            top->clk = 0;
        } else if ((main_time % (2 * kHalfPeriodPs)) == 0) {
            top->clk = 1;

            if (main_time == kResetReleaseTime) {
                top->rst = 0;
                std::puts("Reset released");
            }

            if (main_time != 0 && (main_time % kProgressInterval) == 0) {
                std::printf("[%" PRIu64 "] cycle=%" PRIu64 "\n", main_time,
                            main_time / (2 * kHalfPeriodPs));
            }

            if (top->or1200_tb_top->dwb_cyc && top->or1200_tb_top->dwb_stb &&
                top->or1200_tb_top->dwb_we) {
                const std::uint32_t addr = top->or1200_tb_top->dwb_adr;
                const std::uint32_t data = top->or1200_tb_top->dwb_dat_o;
                if (addr == kExitMagicAddr) {
                    if (data == kExitSuccessValue) {
                        program_finished = true;
                        finish_time = main_time;
                        exit_code = 0;
                    } else if (data == kExitAbortValue) {
                        program_finished = true;
                        finish_time = main_time;
                        exit_code = 1;
                    }
                }
            }
        }

        top->eval();
#ifdef TRACE_ENABLED
        trace->dump(main_time);
#endif
        ++main_time;
    }

    const auto cycles = finish_time / (2 * kHalfPeriodPs);

    if (program_finished) {
        std::printf("Program exited (%s) at time %" PRIu64 " (%" PRIu64 " cycles)\n",
                    exit_code == 0 ? "success" : "abort", finish_time, cycles);
    } else if (main_time >= kMaxSimTime) {
        std::printf("Simulation reached time limit (%" PRIu64 " cycles)\n",
                    kMaxSimTime / (2 * kHalfPeriodPs));
        exit_code = 2;
    } else {
        std::printf("Simulation stopped at time %" PRIu64 "\n", main_time);
    }

#ifdef TRACE_ENABLED
    trace->close();
#endif
    return exit_code;
}
