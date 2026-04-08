#!/usr/bin/env python3
import time
from pathlib import Path

import numpy as np
from PIL import Image

from src.env import MmbnEnv
from src.env.mmbn_env import ACTION_NAMES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROM_PATH = str(PROJECT_ROOT / 'roms' / 'Mega Man Battle Network 6 - Cybeast Gregar (USA).gba')
SAVE_PATH = str(PROJECT_ROOT / 'roms' / 'Mega Man Battle Network 6 - Cybeast Gregar (USA).sav')
OUT_DIR = PROJECT_ROOT / 'assets'


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    save_path = SAVE_PATH if Path(SAVE_PATH).exists() else None
    env = MmbnEnv(rom_path=ROM_PATH, save_path=save_path, state_path='1', render_mode='rgb_array')
    obs, info = env.reset()

    frames = []
    total_frames = 150
    print(f'Capturing {total_frames} frames...')

    for i in range(total_frames):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        screen = env.render()
        if screen is not None:
            img = Image.fromarray(screen)
            img = img.resize((480, 320), Image.NEAREST)
            frames.append(img)

        if terminated or truncated:
            obs, info = env.reset()

        if (i + 1) % 30 == 0:
            print(f'  {i + 1}/{total_frames} - HP: {info.get("player_hp", 0)}/{info.get("enemy_hp", 0)}')

    env.close()

    gif_path = OUT_DIR / 'demo.gif'
    print(f'Saving GIF to {gif_path}...')
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=66,
        loop=0,
        optimize=True,
    )
    print(f'Done! {gif_path} ({gif_path.stat().st_size // 1024}KB)')


if __name__ == '__main__':
    main()
