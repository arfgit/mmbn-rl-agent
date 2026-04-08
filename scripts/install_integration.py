#!/usr/bin/env python3
import shutil
import sys
from pathlib import Path

import stable_retro


def main():
    game_name = "MegaManBattleNetwork6CybeastGregar-GBA"
    game_dir = Path(stable_retro.data.path()) / "stable" / game_name
    game_dir.mkdir(parents=True, exist_ok=True)

    integration_dir = Path(__file__).resolve().parent.parent / "integration"
    if not integration_dir.exists():
        print(f"Error: integration directory not found at {integration_dir}")
        sys.exit(1)

    for f in integration_dir.iterdir():
        shutil.copy2(f, game_dir / f.name)
        print(f"Copied {f.name}")

    rom_src = Path(__file__).resolve().parent.parent / "roms"
    for gba in rom_src.glob("*.gba"):
        shutil.copy2(gba, game_dir / "rom.gba")
        print(f"Copied ROM: {gba.name}")
        break

    state_src = rom_src / "mega-man-battle-network-6.12355.sps"
    if state_src.exists():
        shutil.copy2(state_src, game_dir / "battle.state")
        print("Copied save state as battle.state")

    print(f"Integration installed to {game_dir}")


if __name__ == "__main__":
    main()
