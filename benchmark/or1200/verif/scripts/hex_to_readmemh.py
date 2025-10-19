#!/usr/bin/env python3
"""Convert objcopy Verilog hex output to $readmemh format."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="intel/objcopy style hex file")
    parser.add_argument("output", type=Path, help="$readmemh compatible file")
    return parser.parse_args()


def collect_words(source: Path) -> dict[int, str]:
    words: dict[int, str] = {}
    byte_addr = 0
    pending_bytes: list[str] = []

    def flush_pending() -> None:
        nonlocal byte_addr, pending_bytes
        if not pending_bytes:
            return
        while len(pending_bytes) < 4:
            pending_bytes.append("00")
        word = "".join(pending_bytes).upper()
        word_addr = byte_addr // 4
        words[word_addr] = word
        byte_addr += 4
        pending_bytes.clear()

    for raw_line in source.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("@"):
            flush_pending()
            byte_addr = int(line[1:], 16)
            pending_bytes.clear()
            continue

        bytes_in_line = line.split()
        if not bytes_in_line:
            continue

        pending_bytes.extend(bytes_in_line)
        while len(pending_bytes) >= 4:
            word_bytes = pending_bytes[:4]
            pending_bytes = pending_bytes[4:]
            word = "".join(word_bytes).upper()
            if len(word) != 8:
                raise ValueError(f"invalid word width at byte address 0x{byte_addr:08X}")
            word_addr = byte_addr // 4
            words[word_addr] = word
            byte_addr += 4

    flush_pending()

    return words


def write_readmemh(words: dict[int, str], destination: Path) -> None:
    with destination.open("w", encoding="ascii") as fh:
        previous_addr: int | None = None
        for word_addr in sorted(words):
            if previous_addr is None or word_addr != previous_addr + 1:
                fh.write(f"@{word_addr:08X}\n")
            fh.write(f"{words[word_addr]}\n")
            previous_addr = word_addr


def main() -> None:
    args = parse_args()
    words = collect_words(args.input)
    write_readmemh(words, args.output)


if __name__ == "__main__":
    main()
