#include <cstdint>
#include <cstdio>
#include <inttypes.h>
#include <fstream>
#include <memory>
#include <vector>
#include <elf.h>

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
constexpr vluint64_t kCovLogIntervalCycles = 1000ULL;

#ifndef PROGRAM_ELF
#error "PROGRAM_ELF must be defined to point at the software ELF image"
#endif

#define STRINGIFY_IMPL(x) #x
#define STRINGIFY(x) STRINGIFY_IMPL(x)

std::uint16_t ReadBe16(const std::vector<std::uint8_t> &buffer, std::size_t offset) {
    return static_cast<std::uint16_t>(static_cast<std::uint16_t>(buffer[offset]) << 8U |
                                      static_cast<std::uint16_t>(buffer[offset + 1]));
}

std::uint32_t ReadBe32(const std::vector<std::uint8_t> &buffer, std::size_t offset) {
    return (static_cast<std::uint32_t>(buffer[offset]) << 24U) |
           (static_cast<std::uint32_t>(buffer[offset + 1]) << 16U) |
           (static_cast<std::uint32_t>(buffer[offset + 2]) << 8U) |
           static_cast<std::uint32_t>(buffer[offset + 3]);
}

template <std::size_t N>
bool WriteSegment(VlUnpacked<IData, N> &mem, std::uint32_t base_addr,
                  const std::uint8_t *data, std::uint32_t file_size,
                  std::uint32_t mem_size) {
    if (mem_size == 0) {
        return true;
    }

    const std::uint32_t available_bytes =
        static_cast<std::uint32_t>(N) * static_cast<std::uint32_t>(sizeof(mem[0]));
    if (base_addr + mem_size > available_bytes) {
        return false;
    }

    for (std::uint32_t offset = 0; offset < mem_size; ++offset) {
        const std::uint32_t addr = base_addr + offset;
        const std::uint32_t word_index = addr / 4U;
        const std::uint32_t byte_index = addr % 4U;
        auto word = mem[word_index];
        const std::uint32_t shift = (3U - byte_index) * 8U;
        const bool has_data = offset < file_size && data != nullptr;
        const std::uint32_t byte_value =
            has_data ? static_cast<std::uint32_t>(data[offset]) : 0U;
        word &= ~(0xFFu << shift);
        word |= (byte_value << shift);
        mem[word_index] = word;
    }

    return true;
}

bool LoadProgramElf(Vor1200_tb_top *top) {
    const char *const elf_path = STRINGIFY(PROGRAM_ELF);
    std::ifstream elf_stream(elf_path, std::ios::binary | std::ios::ate);
    if (!elf_stream) {
        std::fprintf(stderr, "Failed to open ELF image: %s\n", elf_path);
        return false;
    }

    const std::streamsize file_size = elf_stream.tellg();
    if (file_size <= 0) {
        std::fprintf(stderr, "ELF image appears empty: %s\n", elf_path);
        return false;
    }
    std::vector<std::uint8_t> buffer(static_cast<std::size_t>(file_size));
    elf_stream.seekg(0, std::ios::beg);
    elf_stream.read(reinterpret_cast<char *>(buffer.data()), buffer.size());
    if (!elf_stream) {
        std::fprintf(stderr, "Failed to read ELF image: %s\n", elf_path);
        return false;
    }

    if (buffer.size() < sizeof(Elf32_Ehdr)) {
        std::fprintf(stderr, "ELF header truncated: %s\n", elf_path);
        return false;
    }

    const auto *ident = buffer.data();
    if (!(ident[0] == 0x7F && ident[1] == 'E' && ident[2] == 'L' && ident[3] == 'F')) {
        std::fprintf(stderr, "ELF magic mismatch: %s\n", elf_path);
        return false;
    }
    if (ident[EI_CLASS] != ELFCLASS32) {
        std::fprintf(stderr, "Unsupported ELF class (expected 32-bit): %s\n", elf_path);
        return false;
    }
    if (ident[EI_DATA] != ELFDATA2MSB) {
        std::fprintf(stderr, "Unsupported ELF endianness (expected big-endian): %s\n",
                     elf_path);
        return false;
    }

    const std::uint16_t ph_entry_size = ReadBe16(buffer, 42);
    const std::uint16_t ph_count = ReadBe16(buffer, 44);
    const std::uint32_t ph_offset = ReadBe32(buffer, 28);

    if (ph_entry_size != sizeof(Elf32_Phdr)) {
        std::fprintf(stderr, "Unexpected program header size (%u): %s\n",
                     ph_entry_size, elf_path);
        return false;
    }
    const std::uint64_t ph_table_end =
        static_cast<std::uint64_t>(ph_offset) +
        static_cast<std::uint64_t>(ph_entry_size) * ph_count;
    if (ph_table_end > buffer.size()) {
        std::fprintf(stderr, "Program header table truncated: %s\n", elf_path);
        return false;
    }

    std::uint32_t imem_bytes = 0;
    std::uint32_t dmem_bytes = 0;
    bool segment_loaded = false;

    for (std::uint16_t i = 0; i < ph_count; ++i) {
        const std::size_t entry_offset = ph_offset + i * ph_entry_size;
        const std::uint32_t type = ReadBe32(buffer, entry_offset + 0);
        if (type != PT_LOAD) {
            continue;
        }
        const std::uint32_t seg_offset = ReadBe32(buffer, entry_offset + 4);
        const std::uint32_t vaddr = ReadBe32(buffer, entry_offset + 8);
        const std::uint32_t filesz = ReadBe32(buffer, entry_offset + 16);
        const std::uint32_t memsz = ReadBe32(buffer, entry_offset + 20);
        const std::uint32_t flags = ReadBe32(buffer, entry_offset + 24);

        if (seg_offset + filesz > buffer.size()) {
            std::fprintf(stderr, "Segment %u exceeds ELF bounds: %s\n", i, elf_path);
            return false;
        }

        const std::uint8_t *segment_data =
            filesz > 0 ? &buffer[seg_offset] : nullptr;
        const bool to_dmem = (flags & PF_W) != 0U;
        const std::uint32_t span = (memsz >= filesz) ? memsz : filesz;

        if (to_dmem) {
            if (!WriteSegment(top->or1200_tb_top->dmem, vaddr, segment_data, filesz,
                              span)) {
                std::fprintf(stderr,
                             "DMEM segment (vaddr=0x%08X, size=%u) out of range: %s\n",
                             vaddr, span, elf_path);
                return false;
            }
            dmem_bytes += span;
        } else {
            if (!WriteSegment(top->or1200_tb_top->imem, vaddr, segment_data, filesz,
                              span)) {
                std::fprintf(stderr,
                             "IMEM segment (vaddr=0x%08X, size=%u) out of range: %s\n",
                             vaddr, span, elf_path);
                return false;
            }
            imem_bytes += span;
        }

        segment_loaded = true;
    }

    if (!segment_loaded) {
        std::fprintf(stderr, "No loadable segments present in ELF: %s\n", elf_path);
        return false;
    }

    std::printf("Loaded program ELF: %s (IMEM %u bytes, DMEM %u bytes)\n",
                elf_path, imem_bytes, dmem_bytes);
    return true;
}

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

    top->eval();  // run initial blocks so that memories are initialised
    if (!LoadProgramElf(top.get())) {
        return 1;
    }

    bool program_finished = false;
    int exit_code = 0;
    vluint64_t finish_time = 0;

    vluint64_t main_time = 0;
    vluint64_t cycle_count = 0;
    std::uint64_t instr_count = 0;
    std::FILE *covsum_log = std::fopen("covsum_log.csv", "w");
    if (covsum_log)
        std::fprintf(covsum_log, "cycle,instructions,covsum\n");
    std::puts("Starting OR1200 simulation");

    while (!Verilated::gotFinish() && main_time < kMaxSimTime && !program_finished) {
        if ((main_time % (2 * kHalfPeriodPs)) == kHalfPeriodPs) {
            top->clk = 0;
        } else if ((main_time % (2 * kHalfPeriodPs)) == 0) {
            top->clk = 1;
            ++cycle_count;

            if (main_time == kResetReleaseTime) {
                top->rst = 0;
                std::puts("Reset released");
            }

            if (main_time != 0 && (main_time % kProgressInterval) == 0) {
                std::printf("[%" PRIu64 "] cycle=%" PRIu64 "\n", main_time,
                            main_time / (2 * kHalfPeriodPs));
            }

            if (top->or1200_tb_top->iwb_cyc && top->or1200_tb_top->iwb_stb &&
                top->or1200_tb_top->iwb_ack) {
                ++instr_count;
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

            if (covsum_log && (cycle_count % kCovLogIntervalCycles) == 0 &&
                !top->rst) {
                std::fprintf(covsum_log, "%" PRIu64 ",%" PRIu64 ",%u\n",
                             cycle_count, instr_count,
                             static_cast<unsigned>(
                                 top->or1200_tb_top->pcov_covsum));
                std::fflush(covsum_log);
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
    if (covsum_log)
        std::fclose(covsum_log);
    return exit_code;
}
