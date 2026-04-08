#!/usr/bin/env python3
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pyglet
from pyglet.window import key as keycodes

from src.env import MmbnEnv, TrainingProgress, EpisodeStats
from src.env.mmbn_env import ACTION_NAMES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROM_PATH = str(PROJECT_ROOT / 'roms' / 'Mega Man Battle Network 6 - Cybeast Gregar (USA).gba')
SAVE_PATH = str(PROJECT_ROOT / 'roms' / 'Mega Man Battle Network 6 - Cybeast Gregar (USA).sav')
LOG_DIR = PROJECT_ROOT / 'logs'

PANEL_W = 300


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--state', type=str, default='1')
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    progress = TrainingProgress.load(str(LOG_DIR / 'progress.json'))

    save_path = SAVE_PATH if Path(SAVE_PATH).exists() else None

    env = MmbnEnv(
        rom_path=ROM_PATH,
        save_path=save_path,
        state_path=args.state,
        render_mode='rgb_array',
    )
    obs, info = env.reset()

    game_w = 240 * args.scale
    game_h = 160 * args.scale
    win_w = game_w + PANEL_W
    win_h = game_h

    window = pyglet.window.Window(win_w, win_h, caption='MMBN Agent Dashboard')

    state = {
        'obs': obs,
        'info': info,
        'paused': False,
        'step_count': 0,
        'episode': progress.total_episodes + 1,
        'episode_reward': 0.0,
        'episode_damage_dealt': 0.0,
        'episode_damage_taken': 0.0,
        'actions_hist': [0] * len(ACTION_NAMES),
        'fps_actual': 0.0,
        'frame_times': [],
        'session_start': time.time(),
        'session_steps': 0,
        'action_log': [],
    }

    panel_bg = pyglet.shapes.Rectangle(game_w, 0, PANEL_W, win_h, color=(20, 20, 30))

    title_label = pyglet.text.Label(
        'MMBN AGENT', font_name='Courier', font_size=14, bold=True,
        x=game_w + 10, y=win_h - 20, color=(0, 255, 180, 255),
    )

    stats_label = pyglet.text.Label(
        '', font_name='Courier', font_size=10,
        x=game_w + 10, y=win_h - 50, anchor_y='top',
        color=(200, 200, 200, 255), multiline=True, width=PANEL_W - 20,
    )

    action_header = pyglet.text.Label(
        'CURRENT ACTION', font_name='Courier', font_size=9,
        x=game_w + PANEL_W // 2, y=160, anchor_x='center',
        color=(120, 120, 140, 255),
    )

    action_label = pyglet.text.Label(
        '', font_name='Courier', font_size=18, bold=True,
        x=game_w + PANEL_W // 2, y=135, anchor_x='center',
        color=(255, 255, 0, 255),
    )

    history_label = pyglet.text.Label(
        '', font_name='Courier', font_size=9,
        x=game_w + 10, y=115, anchor_y='top',
        color=(180, 180, 200, 255), multiline=True, width=PANEL_W - 20,
    )

    status_label = pyglet.text.Label(
        '', font_name='Courier', font_size=10,
        x=game_w + 10, y=20,
        color=(100, 255, 100, 255),
    )

    def save_session_log():
        log_file = LOG_DIR / 'session_log.jsonl'
        entry = {
            'timestamp': datetime.now().isoformat(),
            'episode': state['episode'],
            'steps': state['step_count'],
            'reward': state['episode_reward'],
            'session_steps': state['session_steps'],
            'session_duration': time.time() - state['session_start'],
            'action_distribution': {
                ACTION_NAMES[i]: state['actions_hist'][i]
                for i in range(len(ACTION_NAMES)) if state['actions_hist'][i] > 0
            },
        }
        with open(log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == keycodes.ESCAPE:
            save_session_log()
            progress.save(str(LOG_DIR / 'progress.json'))
            window.close()
        elif symbol == keycodes.P:
            state['paused'] = not state['paused']
        elif symbol == keycodes.R:
            ep_stats = EpisodeStats(
                damage_dealt=state['episode_damage_dealt'],
                damage_taken=state['episode_damage_taken'],
                reward_total=state['episode_reward'],
                steps=state['step_count'],
            )
            progress.record(ep_stats)

            obs, info = env.reset()
            state['obs'] = obs
            state['info'] = info
            state['step_count'] = 0
            state['episode'] += 1
            state['episode_reward'] = 0.0
            state['episode_damage_dealt'] = 0.0
            state['episode_damage_taken'] = 0.0
            state['actions_hist'] = [0] * len(ACTION_NAMES)

    @window.event
    def on_draw():
        window.clear()

        frame = env.render_bgra()
        if frame is not None:
            frame_flipped = np.ascontiguousarray(np.flip(frame, axis=0))
            h, w = frame_flipped.shape[:2]
            img = pyglet.image.ImageData(w, h, 'BGRA', frame_flipped.tobytes())
            texture = img.get_texture()
            texture.width = game_w
            texture.height = game_h
            texture.blit(0, 0)

        panel_bg.draw()
        title_label.draw()

        top5 = sorted(range(len(state['actions_hist'])),
                       key=lambda i: state['actions_hist'][i], reverse=True)[:5]
        top5_str = '\n'.join(
            f"  {ACTION_NAMES[i]:>8}: {state['actions_hist'][i]}"
            for i in top5
        )

        elapsed = time.time() - state['session_start']
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)

        stats_label.text = (
            f"Episode:    {state['episode']}\n"
            f"Step:       {state['step_count']}\n"
            f"Frame:      {state['info'].get('frame', 0)}\n"
            f"Reward:     {state['episode_reward']:.2f}\n"
            f"FPS:        {state['fps_actual']:.0f}\n"
            f"Session:    {mins}m {secs}s\n"
            f"Total Steps:{state['session_steps']}\n"
            f"\n--- All Time ---\n"
            f"Episodes:   {progress.total_episodes}\n"
            f"Wins:       {progress.wins}\n"
            f"Win Rate:   {progress.win_rate:.1%}\n"
            f"Best Streak:{progress.best_win_streak}\n"
            f"Best Reward:{progress.best_reward:.1f}\n"
            f"\n--- Actions ---\n{top5_str}"
        )
        stats_label.draw()

        action_header.draw()
        action_label.text = state['info'].get('action_name', 'NOOP')
        action_label.draw()

        last8 = state['action_log'][-8:]
        if last8:
            history_label.text = '\n'.join(f"  {a}" for a in reversed(last8))
            history_label.draw()

        status_label.text = 'PAUSED [P]' if state['paused'] else 'RUNNING | P=Pause R=Reset'
        status_label.color = (255, 100, 100, 255) if state['paused'] else (100, 255, 100, 255)
        status_label.draw()

    def update(dt):
        if state['paused']:
            return

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        state['obs'] = obs
        state['info'] = info
        state['step_count'] += 1
        state['session_steps'] += 1
        state['episode_reward'] += reward
        state['actions_hist'][action] += 1
        state['action_log'].append(ACTION_NAMES[action])
        if len(state['action_log']) > 100:
            state['action_log'] = state['action_log'][-50:]

        now = time.time()
        state['frame_times'].append(now)
        state['frame_times'] = [t for t in state['frame_times'] if now - t < 1.0]
        state['fps_actual'] = len(state['frame_times'])

        if state['session_steps'] % 100 == 0:
            save_session_log()

        if terminated or truncated:
            ep_stats = EpisodeStats(
                damage_dealt=state['episode_damage_dealt'],
                damage_taken=state['episode_damage_taken'],
                reward_total=state['episode_reward'],
                steps=state['step_count'],
            )
            progress.record(ep_stats)

            if state['episode'] % 10 == 0:
                progress.save(str(LOG_DIR / 'progress.json'))

            obs, info = env.reset()
            state['obs'] = obs
            state['info'] = info
            state['step_count'] = 0
            state['episode'] += 1
            state['episode_reward'] = 0.0
            state['episode_damage_dealt'] = 0.0
            state['episode_damage_taken'] = 0.0
            state['actions_hist'] = [0] * len(ACTION_NAMES)

    pyglet.clock.schedule_interval(update, 1.0 / args.fps)

    print(f'Agent Dashboard | State: slot {args.state}')
    print(f'  P     — Pause/Resume')
    print(f'  R     — Reset episode')
    print(f'  Esc   — Save & Quit')
    print()

    pyglet.app.run()

    progress.save(str(LOG_DIR / 'progress.json'))
    env.close()


if __name__ == '__main__':
    main()
