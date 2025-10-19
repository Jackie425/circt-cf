/*
 * OR1200 Linker Script - Harvard Architecture
 * 
 * Memory layout for Harvard Architecture (separate instruction and data memory):
 *   IMEM: 0x00000000 - 0x00003FFF (16KB) - Instruction memory
 *   DMEM: 0x00000000 - 0x00003FFF (16KB) - Data memory (separate address space)
 * 
 * Note: In Harvard architecture, instruction and data use separate physical memories.
 * Both can use the same address range because they're accessed via different buses.
 */

MEMORY
{
    IMEM : ORIGIN = 0x00000000, LENGTH = 64K
    DMEM : ORIGIN = 0x00000000, LENGTH = 64K
}

SECTIONS
{
    /* Code section - goes to instruction memory */
    .text 0x100 :
    {
        *(.text.start)
        *(.text)
        *(.text*)
        *(.rodata)
        *(.rodata*)
    } > IMEM
    
    /* Initialized data - goes to data memory */
    .data 0x0 :
    {
        *(.data)
        *(.data*)
    } > DMEM
    
    /* Uninitialized data - goes to data memory */
    .bss :
    {
        *(.bss)
        *(.bss*)
        *(COMMON)
    } > DMEM
    
    /* End of data/bss */
    _end = .;
    
    /* Stack grows downward from top of DMEM */
    _stack_top = ORIGIN(DMEM) + LENGTH(DMEM);  /* 0x10000 */
}

/* Define entry point */
ENTRY(_start)
