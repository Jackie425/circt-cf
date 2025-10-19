.section .text.start
.global _start, _exit, exit, abort

_start:
    l.movhi r1, hi(_stack_top)
    l.ori   r1, r1, lo(_stack_top)
    l.or    r2, r0, r0

    l.jal   main
    l.nop

    l.or    r3, r11, r11

_exit:
    l.sfne  r3, r0              /* r3 = return status */
    l.bf    abort               /* non-zero -> abort path */
    l.nop

    /* Write 0xDEADBEEF to address 0xFFFF0000 to signal completion */
    l.movhi r4, 0xFFFF          /* r4 = 0xFFFF0000 */
    l.movhi r5, 0xDEAD          /* r5 = 0xDEAD0000 */
    l.ori   r5, r5, 0xBEEF      /* r5 = 0xDEADBEEF */
    l.sw    0(r4), r5           /* mem[0xFFFF0000] = 0xDEADBEEF */

_halt:
    /* Infinite halt loop */
    l.nop   0x1                 /* NOP with immediate = halt hint */
    l.j     _halt
    l.nop

abort:
    /* Write abort magic value */
    l.movhi r4, 0xFFFF          /* r4 = 0xFFFF0000 */
    l.movhi r5, 0xABAD          /* r5 = 0xABAD0000 */
    l.ori   r5, r5, 0xBABE      /* r5 = 0xABADBABE */
    l.sw    0(r4), r5           /* mem[0xFFFF0000] = 0xABADBABE */

    /* Halt */
    l.nop   0x1
    l.j     abort
    l.nop

exit:
    l.j     _exit
    l.nop
