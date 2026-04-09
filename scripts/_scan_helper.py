#!/usr/bin/env python3
import sys

from src.env.mgba_core import MgbaCore

KEY_A = 1 << 0
KEY_B = 1 << 1
KEY_START = 1 << 3
KEY_RIGHT = 1 << 4
KEY_LEFT = 1 << 5
KEY_UP = 1 << 6
KEY_DOWN = 1 << 7
KEY_R = 1 << 8
KEY_L = 1 << 9

def log(msg=""):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()

def read_region(core, start, size):
    return bytes(core.read8(start + i) for i in range(size))

def diff(before, after, base):
    return [(base + i, i, before[i], after[i]) for i in range(len(before)) if before[i] != after[i]]

def press(core, keys, hold=8, wait=30):
    core.set_keys(keys)
    for _ in range(hold):
        core.run_frame()
    core.set_keys(0)
    for _ in range(wait):
        core.run_frame()

def run(core, n=30):
    for _ in range(n):
        core.run_frame()

def show_changes(changes, label, limit=40):
    log(f"\n--- {label} --- ({len(changes)} changes)")
    if not changes:
        log("  No changes")
    for addr, off, bef, aft in changes[:limit]:
        log(f"  0x{addr:08X} (+0x{off:03X}): {bef:3d} -> {aft:3d}  (delta {aft-bef:+d})")
    if len(changes) > limit:
        log(f"  ... and {len(changes)-limit} more")

def hp(core):
    return core.read16(0x0203A9D4), core.read16(0x0203AAAC)

def dump_hex(core, start, size, label):
    log(f"\n--- {label} ---")
    data = read_region(core, start, size)
    for row in range(0, size, 16):
        end = min(16, size - row)
        hex_str = ' '.join(f'{data[row+i]:02X}' for i in range(end))
        log(f"  0x{start+row:08X}: {hex_str}")

def main():
    ROM = 'roms/Mega Man Battle Network 6 - Cybeast Gregar (USA).gba'
    SAV = 'roms/Mega Man Battle Network 6 - Cybeast Gregar (USA).sav'

    core = MgbaCore(ROM, save_path=SAV)

    # Wider scan region
    REGION = 0x02030000
    SIZE = 0x10000

    core.load_state_slot(1)
    run(core, 60)

    php, ehp = hp(core)
    log(f"=== MMBN6 RAM Scanner ===")
    log(f"Scan: 0x{REGION:08X} - 0x{REGION+SIZE:08X} ({SIZE} bytes)")
    log(f"Player HP: {php}, Enemy HP: {ehp}")

    # Baseline after 60 idle frames for stability
    snap_a = read_region(core, REGION, SIZE)

    # More idle to filter out timer noise
    run(core, 60)
    snap_b = read_region(core, REGION, SIZE)
    noise_addrs = {REGION + i for i in range(SIZE) if snap_a[i] != snap_b[i]}
    log(f"\nNoise addresses (change on idle): {len(noise_addrs)}")

    def filtered_diff(before, after):
        return [(REGION + i, i, before[i], after[i])
                for i in range(SIZE)
                if before[i] != after[i] and (REGION + i) not in noise_addrs]

    # Test: Move RIGHT
    core.load_state_slot(1); run(core, 60)
    snap_a = read_region(core, REGION, SIZE)
    press(core, KEY_RIGHT, hold=10, wait=60)
    snap_b = read_region(core, REGION, SIZE)
    show_changes(filtered_diff(snap_a, snap_b), "MOVE RIGHT")

    # Move LEFT from that position
    snap_a = snap_b
    press(core, KEY_LEFT, hold=10, wait=60)
    snap_b = read_region(core, REGION, SIZE)
    show_changes(filtered_diff(snap_a, snap_b), "MOVE LEFT (back)")

    # Move UP from fresh state
    core.load_state_slot(1); run(core, 60)
    snap_a = read_region(core, REGION, SIZE)
    press(core, KEY_UP, hold=10, wait=60)
    snap_b = read_region(core, REGION, SIZE)
    show_changes(filtered_diff(snap_a, snap_b), "MOVE UP")

    # Move DOWN from that
    snap_a = snap_b
    press(core, KEY_DOWN, hold=10, wait=60)
    snap_b = read_region(core, REGION, SIZE)
    show_changes(filtered_diff(snap_a, snap_b), "MOVE DOWN (back)")

    # Buster: hold A, wait long enough for animation
    core.load_state_slot(1); run(core, 60)
    snap_a = read_region(core, REGION, SIZE)
    press(core, KEY_A, hold=10, wait=120)
    snap_b = read_region(core, REGION, SIZE)
    show_changes(filtered_diff(snap_a, snap_b), "PRESS A (buster)")
    log(f"  HP after: Player={hp(core)[0]} Enemy={hp(core)[1]}")

    # Chip select with L
    core.load_state_slot(1); run(core, 60)
    snap_a = read_region(core, REGION, SIZE)
    press(core, KEY_L, hold=10, wait=120)
    snap_b = read_region(core, REGION, SIZE)
    show_changes(filtered_diff(snap_a, snap_b), "PRESS L (chip select)")

    # START pause
    core.load_state_slot(1); run(core, 60)
    snap_a = read_region(core, REGION, SIZE)
    press(core, KEY_START, hold=10, wait=60)
    snap_b = read_region(core, REGION, SIZE)
    show_changes(filtered_diff(snap_a, snap_b), "PRESS START (pause)")

    # Hex dump of known player/enemy blocks
    core.load_state_slot(1); run(core, 60)
    dump_hex(core, 0x0203A9C0, 128, "PLAYER DATA BLOCK")
    dump_hex(core, 0x0203AA80, 128, "ENEMY DATA BLOCK")

    # Wider dump around known HP addresses
    dump_hex(core, 0x02034000, 64, "0x02034000 area")
    dump_hex(core, 0x02038000, 64, "0x02038000 area")

    core.close()
    log("\nDone.")


if __name__ == "__main__":
    main()
