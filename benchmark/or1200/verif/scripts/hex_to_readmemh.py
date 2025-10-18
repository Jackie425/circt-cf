#!/usr/bin/env python3
"""
hex_to_readmemh.py

Convert Intel HEX format (with byte addresses) to plain hex format 
suitable for Verilog $readmemh system task (with word addresses).

Input format:
    @00000100              <- byte address marker
    18 20 00 00 A8 21 ...  <- space-separated hex bytes

Output format:
    @00000040              <- word address marker (byte_addr / 4)
    18200000               <- 32-bit words (no spaces)
    A8211000
    ...

Author: Generated for OR1200 Verilator testbench
Date: 2025-10-18
"""

import sys
import os

def convert_intel_hex_to_readmemh(input_file, output_file):
    
    print(f"=== Intel HEX to $readmemh Converter ===")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print()
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Parse address offset
    start_byte_addr = 0
    words = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check for address marker
        if line.startswith('@'):
            start_byte_addr = int(line[1:], 16)
            print(f"Start byte address: 0x{start_byte_addr:08x} ({start_byte_addr} bytes)")
            continue
        
        # Remove spaces and parse hex bytes
        bytes_str = line.replace(' ', '')
        
        # Convert to 32-bit words (big-endian)
        for i in range(0, len(bytes_str), 8):  # 8 hex chars = 4 bytes = 32 bits
            if i + 8 <= len(bytes_str):
                word = bytes_str[i:i+8]
                words.append(word)
    
    # Calculate word address (byte address / 4)
    start_word_addr = start_byte_addr // 4
    print(f"Start word address: 0x{start_word_addr:08x} ({start_word_addr} words)")
    print(f"Total words: {len(words)}")
    print()
    
    # Write to output
    with open(output_file, 'w') as f:
        # Write address marker for word address
        f.write(f"@{start_word_addr:08x}\n")
        
        # Write words
        for word in words:
            f.write(f"{word}\n")
    
    print(f"✓ Successfully converted to: {output_file}")
    print(f"\nFirst 5 words:")
    for i, word in enumerate(words[:5]):
        word_addr = start_word_addr + i
        byte_addr = word_addr * 4
        print(f"  imem[0x{word_addr:04x}] = 0x{word} (byte addr: 0x{byte_addr:08x})")
    
    if len(words) > 5:
        print(f"  ... and {len(words) - 5} more words")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_intel_hex> <output_readmemh_hex>")
        print()
        print("Example:")
        print(f"  {sys.argv[0]} build/program.hex build/program.readmemh.hex")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    try:
        convert_intel_hex_to_readmemh(input_file, output_file)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
