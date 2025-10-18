/*
 * crt0.S - C Runtime Startup for OR1K (OpenRISC 1000)
 * 
 * This file provides the startup code that:
 * 1. Initializes the stack pointer
 * 2. Calls main()
 * 3. Handles program exit with a halt signal
 */

.section .text.start
.global _start

_start:
    /* Initialize stack pointer to top of RAM (64KB = 0x10000) */
    l.movhi r1, 0x1
    l.ori   r1, r1, 0x0000
    
    /* Clear frame pointer */
    l.or    r2, r0, r0
    
    /* Call main function */
    l.jal   main
    l.nop
    
    /* main() returned - handle exit */
    /* Save return value from main (r11 contains return value) */
    l.or    r3, r11, r11
    
_exit:
    /* Write exit magic value to special address to signal program end */
    /* This can be detected by the testbench */
    
    /* Write 0xDEADBEEF to address 0xFFFF0000 to signal completion */
    l.movhi r4, 0xFFFF          /* r4 = 0xFFFF0000 */
    l.movhi r5, 0xDEAD          /* r5 = 0xDEAD0000 */
    l.ori   r5, r5, 0xBEEF      /* r5 = 0xDEADBEEF */
    l.sw    0(r4), r5           /* mem[0xFFFF0000] = 0xDEADBEEF */
    
_halt:
    /* Infinite halt loop */
    /* Use special NOP that simulators can detect */
    l.nop   0x1                 /* NOP with immediate = exit/halt indicator */
    l.j     _halt
    l.nop

/* Define abort function for baremetal (required by some libraries) */
.global abort
.type abort, @function
abort:
    /* Write abort magic value */
    l.movhi r4, 0xFFFF          /* r4 = 0xFFFF0000 */
    l.movhi r5, 0xABAD          /* r5 = 0xABAD0000 */
    l.ori   r5, r5, 0xBABE      /* r5 = 0xABADBABE */
    l.sw    0(r4), r5           /* mem[0xFFFF0000] = 0xABADBABE (abort signal) */
    
    /* Halt */
    l.nop   0x1
    l.j     abort
    l.nop
