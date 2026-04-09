#!/usr/bin/env python3
"""
RAM scanner for MMBN6 battle state discovery.

Workflow:
  1. Load a save state into a battle
  2. Take a baseline RAM snapshot
  3. Perform an action in-game (move, attack, open chips, beast out, etc.)
  4. Take another snapshot and diff

Run interactively: press keys to scan, move in-game between scans.

Usage:
  python scripts/ram_scan.py
  python scripts/ram_scan.py --state 1
  python scripts/ram_scan.py --region 0x0203A900 --size 512
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from src.env.mgba_core import MgbaCore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROM_PATH = str(PROJECT_ROOT / 'roms' / 'Mega Man Battle Network 6 - Cybeast Gregar (USA).gba')
SAVE_PATH = str(PROJECT_ROOT / 'roms' / 'Mega Man Battle Network 6 - Cybeast Gregar (USA).sav')

WRAM_START = 0x02000000
WRAM_SIZE = 0x40000

BATTLE_REGION_START = 0x0203A900
BATTLE_REGION_SIZE = 0x300

KEY_A = 1 << 0
KEY_B = 1 << 1
KEY_START = 1 << 3
KEY_RIGHT = 1 << 4
KEY_LEFT = 1 << 5
KEY_UP = 1 << 6
KEY_DOWN = 1 << 7
KEY_R = 1 << 8
KEY_L = 1 << 9

KEY_MAP = {
    'a': KEY_A, 'b': KEY_B, 's': KEY_START,
    'u': KEY_UP, 'd': KEY_DOWN, 'l': KEY_LEFT, 'r': KEY_RIGHT,
    'lb': KEY_L, 'rb': KEY_R,
}


def read_region(core: MgbaCore, start: int, size: int) -> bytes:
    data = bytearray(size)
    for i in range(size):
        data[i] = core.read8(start + i)
    return bytes(data)


def diff_snapshots(before: bytes, after: bytes, base_addr: int) -> list[dict]:
    changes = []
    for i in range(len(before)):
        if before[i] != after[i]:
            changes.append({
                'addr': base_addr + i,
                'offset': i,
                'before': before[i],
                'after': after[i],
                'delta': after[i] - before[i],
            })
    return changes


def print_changes(changes: list[dict], label: str = "") -> None:
    if label:
        print(f"\n=== {label} === ({len(changes)} changes)")
    if not changes:
        print("  No changes detected")
        return
    for c in changes[:50]:
        print(f"  0x{c['addr']:08X} (+0x{c['offset']:03X}): {c['before']:3d} -> {c['after']:3d}  (delta {c['delta']:+d})")
    if len(changes) > 50:
        print(f"  ... and {len(changes) - 50} more")


def print_known_addresses(core: MgbaCore) -> None:
    player_hp = core.read16(0x0203A9D4)
    enemy_hp = core.read16(0x0203AAAC)
    beast_byte = core.read8(0x0203A9F0)
    print(f"\n  Known: Player HP={player_hp}, Enemy HP={enemy_hp}, Beast=0x{beast_byte:02X}")


def press_key(core: MgbaCore, keys: int, hold_frames: int = 4) -> None:
    core.set_keys(keys)
    for _ in range(hold_frames):
        core.run_frame()
    core.set_keys(0)
    for _ in range(4):
        core.run_frame()


def run_frames(core: MgbaCore, n: int = 30) -> None:
    for _ in range(n):
        core.run_frame()


def narrow_search(history: list[tuple[str, list[dict]]]) -> dict[int, list]:
    addr_events = {}
    for label, changes in history:
        for c in changes:
            addr = c['addr']
            if addr not in addr_events:
                addr_events[addr] = []
            addr_events[addr].append({'label': label, 'before': c['before'], 'after': c['after']})
    return addr_events


def main():
    parser = argparse.ArgumentParser(description="MMBN6 RAM Scanner")
    parser.add_argument("--state", type=str, default="1")
    parser.add_argument("--region", type=lambda x: int(x, 16), default=BATTLE_REGION_START)
    parser.add_argument("--size", type=int, default=BATTLE_REGION_SIZE)
    args = parser.parse_args()

    save_path = SAVE_PATH if Path(SAVE_PATH).exists() else None
    core = MgbaCore(ROM_PATH, save_path=save_path)

    if args.state.isdigit():
        core.load_state_slot(int(args.state))
    else:
        core.load_state(args.state)

    run_frames(core, 30)

    region_start = args.region
    region_size = args.size
    print(f"Scanning region 0x{region_start:08X} - 0x{region_start + region_size:08X} ({region_size} bytes)")
    print_known_addresses(core)

    history: list[tuple[str, list[dict]]] = []
    snapshot = read_region(core, region_start, region_size)

    print("\nCommands:")
    print("  snap [label]     - take snapshot and diff against previous")
    print("  press <key>      - press a button (a/b/s/u/d/l/r/lb/rb)")
    print("  wait [frames]    - advance N frames (default 60)")
    print("  known            - print known address values")
    print("  narrow           - show addresses that changed in multiple scans")
    print("  dump [addr] [n]  - dump N bytes at hex address")
    print("  watch <addr>     - continuously read an address")
    print("  find <value>     - find all bytes in region matching value")
    print("  q                - quit")

    while True:
        try:
            raw = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not raw:
            continue

        parts = raw.split()
        cmd = parts[0].lower()

        if cmd == 'q':
            break

        elif cmd == 'snap':
            label = parts[1] if len(parts) > 1 else f"snap_{len(history)}"
            new_snapshot = read_region(core, region_start, region_size)
            changes = diff_snapshots(snapshot, new_snapshot, region_start)
            print_changes(changes, label)
            print_known_addresses(core)
            history.append((label, changes))
            snapshot = new_snapshot

        elif cmd == 'press':
            if len(parts) < 2 or parts[1] not in KEY_MAP:
                print(f"  Keys: {', '.join(KEY_MAP.keys())}")
                continue
            key_name = parts[1]
            hold = int(parts[2]) if len(parts) > 2 else 4
            press_key(core, KEY_MAP[key_name], hold)
            print(f"  Pressed {key_name} for {hold} frames")
            print_known_addresses(core)

        elif cmd == 'wait':
            n = int(parts[1]) if len(parts) > 1 else 60
            run_frames(core, n)
            print(f"  Advanced {n} frames")
            print_known_addresses(core)

        elif cmd == 'known':
            print_known_addresses(core)

        elif cmd == 'narrow':
            addr_events = narrow_search(history)
            multi = {a: evts for a, evts in addr_events.items() if len(evts) >= 2}
            if not multi:
                print("  No addresses changed in multiple scans yet")
            else:
                print(f"\n  Addresses changed in 2+ scans ({len(multi)} found):")
                for addr in sorted(multi.keys()):
                    evts = multi[addr]
                    summary = ', '.join(f"{e['label']}: {e['before']}->{e['after']}" for e in evts[:5])
                    print(f"    0x{addr:08X}: {summary}")

        elif cmd == 'dump':
            addr = int(parts[1], 16) if len(parts) > 1 else region_start
            n = int(parts[2]) if len(parts) > 2 else 64
            data = read_region(core, addr, n)
            for row_start in range(0, n, 16):
                hex_str = ' '.join(f'{data[row_start + i]:02X}' for i in range(min(16, n - row_start)))
                ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in data[row_start:row_start + 16])
                print(f"  0x{addr + row_start:08X}: {hex_str:<48s} {ascii_str}")

        elif cmd == 'watch':
            if len(parts) < 2:
                print("  Usage: watch 0x0203A9D4")
                continue
            addr = int(parts[1], 16)
            print(f"  Watching 0x{addr:08X} (press Ctrl+C to stop)")
            prev = core.read16(addr)
            try:
                while True:
                    run_frames(core, 4)
                    val = core.read16(addr)
                    if val != prev:
                        print(f"    0x{addr:08X}: {prev} -> {val} (delta {val - prev:+d})")
                        prev = val
            except KeyboardInterrupt:
                print("  Stopped watching")

        elif cmd == 'find':
            if len(parts) < 2:
                print("  Usage: find 100  (find bytes with value 100)")
                continue
            target = int(parts[1])
            data = read_region(core, region_start, region_size)
            matches = [i for i, b in enumerate(data) if b == target]
            print(f"  Found {len(matches)} bytes with value {target}:")
            for m in matches[:20]:
                print(f"    0x{region_start + m:08X} (+0x{m:03X})")
            if len(matches) > 20:
                print(f"    ... and {len(matches) - 20} more")

        else:
            print(f"  Unknown command: {cmd}")

    core.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
