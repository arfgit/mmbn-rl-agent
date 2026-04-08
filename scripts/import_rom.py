#!/usr/bin/env python3
"""Import a MMBN ROM into gym-retro."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Import MMBN ROM into gym-retro")
    parser.add_argument("rom_path", type=str, help="Path to the .gba ROM file")
    args = parser.parse_args()

    rom = Path(args.rom_path)
    if not rom.exists():
        print(f"Error: ROM file not found: {rom}")
        sys.exit(1)

    if rom.suffix.lower() != ".gba":
        print(f"Error: Expected a .gba file, got {rom.suffix}")
        sys.exit(1)

    # Copy ROM to local roms/ directory for reference
    roms_dir = Path("roms")
    roms_dir.mkdir(exist_ok=True)
    local_copy = roms_dir / rom.name
    shutil.copy2(rom, local_copy)
    print(f"ROM copied to {local_copy}")

    # Import into gym-retro
    print("Importing ROM into gym-retro...")
    result = subprocess.run(
        [sys.executable, "-m", "retro.import", str(roms_dir)],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print("ROM imported successfully.")
        if result.stdout:
            print(result.stdout)
    else:
        print(f"Import failed: {result.stderr}")
        sys.exit(1)


if __name__ == "__main__":
    main()
