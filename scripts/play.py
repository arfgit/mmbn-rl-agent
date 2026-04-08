#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROM_PATH = PROJECT_ROOT / 'roms' / 'Mega Man Battle Network 6 - Cybeast Gregar (USA).gba'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', type=str, default=None)
    parser.add_argument('--scale', type=int, default=4)
    args = parser.parse_args()

    if not ROM_PATH.exists():
        print(f'ROM not found: {ROM_PATH}')
        sys.exit(1)

    cmd = ['mgba', f'-{args.scale}', str(ROM_PATH)]
    if args.state:
        state_path = PROJECT_ROOT / 'roms' / f'{args.state}.state'
        if state_path.exists():
            cmd.extend(['-t', str(state_path)])
        else:
            print(f'State file not found: {state_path}')
            sys.exit(1)

    print('Launching mGBA...')
    print(f'ROM: {ROM_PATH.name}')
    if args.state:
        print(f'State: {args.state}')
    print(f'Scale: {args.scale}x')
    print()
    print('mGBA Controls:')
    print('  Arrow keys  — D-pad')
    print('  Z           — B button')
    print('  X           — A button')
    print('  A / S       — L / R triggers')
    print('  Enter       — Start')
    print('  Backspace   — Select')
    print('  F1-F9       — Quick save/load states')
    print('  Shift+F1-F9 — Save to slot')
    print()

    subprocess.run(cmd)


if __name__ == '__main__':
    main()
