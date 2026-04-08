#!/usr/bin/env python3
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.env import MmbnEnv
from src.env.mmbn_env import ACTION_NAMES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROM_PATH = str(PROJECT_ROOT / 'roms' / 'Mega Man Battle Network 6 - Cybeast Gregar (USA).gba')
SAVE_PATH = str(PROJECT_ROOT / 'roms' / 'Mega Man Battle Network 6 - Cybeast Gregar (USA).sav')
OUT_DIR = PROJECT_ROOT / 'assets'

PANEL_W = 220
SCALE = 2
GAME_W = 240 * SCALE
GAME_H = 160 * SCALE
TOTAL_W = GAME_W + PANEL_W
TOTAL_H = GAME_H


def draw_panel(draw, info, step, episode_reward, action_name, recent_actions):
    x = GAME_W
    bg_color = (15, 15, 22)
    draw.rectangle([x, 0, TOTAL_W, TOTAL_H], fill=bg_color)

    y = 10
    draw.text((x + 8, y), "MMBN AGENT", fill=(0, 255, 180))
    y += 20

    draw.text((x + 8, y), "BATTLE", fill=(255, 220, 100))
    y += 16
    p_hp = info.get('player_hp', 0)
    e_hp = info.get('enemy_hp', 0)
    draw.text((x + 8, y), f"Player HP  {p_hp}", fill=(200, 200, 200))
    y += 14
    draw.text((x + 8, y), f"Enemy  HP  {e_hp}", fill=(200, 200, 200))
    y += 14
    draw.text((x + 8, y), f"Dmg Dealt  {info.get('damage_dealt', 0):.0f}", fill=(200, 200, 200))
    y += 14
    draw.text((x + 8, y), f"Dmg Taken  {info.get('damage_taken', 0):.0f}", fill=(200, 200, 200))
    y += 22

    draw.text((x + 8, y), f"EPISODE", fill=(160, 180, 220))
    y += 16
    draw.text((x + 8, y), f"Step    {step}", fill=(200, 200, 200))
    y += 14
    draw.text((x + 8, y), f"Reward  {episode_reward:.2f}", fill=(200, 200, 200))
    y += 22

    draw.text((x + 8, y), f"ACTION", fill=(255, 255, 0))
    y += 16
    draw.text((x + 8, y), f"> {action_name}", fill=(255, 255, 0))
    y += 18

    for a in reversed(recent_actions[-5:]):
        draw.text((x + 8, y), f"  {a}", fill=(140, 140, 160))
        y += 13


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    save_path = SAVE_PATH if Path(SAVE_PATH).exists() else None
    env = MmbnEnv(rom_path=ROM_PATH, save_path=save_path, state_path='1', render_mode='rgb_array')
    obs, info = env.reset()

    frames = []
    recent_actions = []
    total_frames = 200
    episode_reward = 0.0
    print(f'Capturing {total_frames} frames with dashboard...')

    for i in range(total_frames):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        action_name = ACTION_NAMES[action]
        recent_actions.append(action_name)
        if len(recent_actions) > 20:
            recent_actions = recent_actions[-10:]

        screen = env.render()
        if screen is not None:
            game_img = Image.fromarray(screen).resize((GAME_W, GAME_H), Image.NEAREST)
            composite = Image.new('RGB', (TOTAL_W, TOTAL_H), (15, 15, 22))
            composite.paste(game_img, (0, 0))
            draw = ImageDraw.Draw(composite)
            draw_panel(draw, info, i + 1, episode_reward, action_name, recent_actions)
            frames.append(composite)

        if terminated or truncated:
            obs, info = env.reset()
            episode_reward = 0.0
            recent_actions = []

        if (i + 1) % 50 == 0:
            print(f'  {i + 1}/{total_frames}')

    env.close()

    gif_path = OUT_DIR / 'demo.gif'
    print(f'Saving GIF to {gif_path}...')
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
        optimize=True,
    )
    print(f'Done! {gif_path} ({gif_path.stat().st_size // 1024}KB)')


if __name__ == '__main__':
    main()
