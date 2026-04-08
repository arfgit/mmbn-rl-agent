import gymnasium as gym
import numpy as np
import cv2
from gymnasium import spaces

from src.env.mgba_core import MgbaCore, GBA_W, GBA_H, KEY_NAMES

ADDR_PLAYER_HP = 0x0203A9D4
ADDR_ENEMY_HP = 0x0203AAAC

ACTIONS = [
    0,                  # 0: nothing
    (1 << 0),           # 1: A
    (1 << 1),           # 2: B
    (1 << 3),           # 3: START
    (1 << 6),           # 4: UP
    (1 << 7),           # 5: DOWN
    (1 << 5),           # 6: LEFT
    (1 << 4),           # 7: RIGHT
    (1 << 0) | (1 << 6),  # 8: A+UP
    (1 << 0) | (1 << 7),  # 9: A+DOWN
    (1 << 0) | (1 << 5),  # 10: A+LEFT
    (1 << 0) | (1 << 4),  # 11: A+RIGHT
    (1 << 1) | (1 << 6),  # 12: B+UP
    (1 << 1) | (1 << 7),  # 13: B+DOWN
    (1 << 1) | (1 << 5),  # 14: B+LEFT
    (1 << 1) | (1 << 4),  # 15: B+RIGHT
    (1 << 8),           # 16: L
    (1 << 9),           # 17: R
]

ACTION_NAMES = [
    'NOOP', 'A', 'B', 'START',
    'UP', 'DOWN', 'LEFT', 'RIGHT',
    'A+UP', 'A+DOWN', 'A+LEFT', 'A+RIGHT',
    'B+UP', 'B+DOWN', 'B+LEFT', 'B+RIGHT',
    'L', 'R',
]


def _keys_to_names(keys: int) -> str:
    pressed = [KEY_NAMES[i] for i in range(10) if keys & (1 << i)]
    return '+'.join(pressed) if pressed else 'NOOP'


class MmbnEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(
        self,
        rom_path: str,
        save_path: str | None = None,
        state_path: str | None = None,
        render_mode: str | None = None,
        frame_skip: int = 4,
        frame_size: tuple[int, int] = (84, 84),
        max_episode_steps: int = 18000,
    ):
        super().__init__()
        self.rom_path = rom_path
        self.save_path = save_path
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.frame_size = frame_size
        self.max_episode_steps = max_episode_steps

        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(frame_size[1], frame_size[0], 1),
            dtype=np.uint8,
        )

        self._core = MgbaCore(rom_path, save_path=save_path)
        self._state_slot = None
        self._state_path = None
        if state_path:
            if state_path.isdigit():
                self._state_slot = int(state_path)
            else:
                self._state_path = state_path

        self._steps = 0
        self._total_reward = 0.0
        self._last_action = 0
        self._last_action_name = 'NOOP'
        self._prev_player_hp = 0
        self._prev_enemy_hp = 0
        self._damage_dealt = 0.0
        self._damage_taken = 0.0
        self._won = False
        self._died = False

    def _read_hp(self) -> tuple[int, int]:
        player_hp = self._core.read16(ADDR_PLAYER_HP)
        enemy_hp = self._core.read16(ADDR_ENEMY_HP)
        return player_hp, enemy_hp

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, self.frame_size, interpolation=cv2.INTER_AREA)
        return resized[:, :, np.newaxis]

    def step(self, action: int):
        keys = ACTIONS[action]
        self._last_action = keys
        self._last_action_name = ACTION_NAMES[action]

        self._core.set_keys(keys)
        for _ in range(self.frame_skip):
            self._core.run_frame()
        self._core.set_keys(0)

        obs = self._preprocess(self._core.get_screen())
        self._steps += 1

        player_hp, enemy_hp = self._read_hp()
        reward = -0.01

        if self._prev_enemy_hp > 0 and enemy_hp < self._prev_enemy_hp:
            dmg = self._prev_enemy_hp - enemy_hp
            reward += dmg * 1.0
            self._damage_dealt += dmg

        if self._prev_player_hp > 0 and player_hp < self._prev_player_hp:
            dmg = self._prev_player_hp - player_hp
            reward -= dmg * 0.5
            self._damage_taken += dmg

        terminated = False
        in_battle = self._prev_player_hp > 0 and self._prev_enemy_hp > 0 and self._steps > 10

        if in_battle and enemy_hp == 0:
            reward += 100.0
            self._won = True
            terminated = True

        if in_battle and player_hp == 0:
            reward -= 50.0
            self._died = True
            terminated = True

        self._prev_player_hp = player_hp
        self._prev_enemy_hp = enemy_hp
        self._total_reward += reward
        truncated = self._steps >= self.max_episode_steps

        info = {
            'frame': self._core.frame_counter,
            'steps': self._steps,
            'action_name': self._last_action_name,
            'total_reward': self._total_reward,
            'player_hp': player_hp,
            'enemy_hp': enemy_hp,
            'damage_dealt': self._damage_dealt,
            'damage_taken': self._damage_taken,
            'won': self._won,
            'died': self._died,
        }

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._core.reset()

        if self._state_slot is not None:
            self._core.load_state_slot(self._state_slot)
        elif self._state_path:
            self._core.load_state(self._state_path)

        self._steps = 0
        self._total_reward = 0.0
        self._last_action = 0
        self._last_action_name = 'NOOP'
        self._damage_dealt = 0.0
        self._damage_taken = 0.0
        self._won = False
        self._died = False

        for _ in range(10):
            self._core.run_frame()

        self._prev_player_hp, self._prev_enemy_hp = self._read_hp()

        obs = self._preprocess(self._core.get_screen())
        info = {
            'frame': self._core.frame_counter,
            'steps': 0,
            'action_name': 'NOOP',
            'total_reward': 0.0,
            'player_hp': self._prev_player_hp,
            'enemy_hp': self._prev_enemy_hp,
            'damage_dealt': 0.0,
            'damage_taken': 0.0,
            'won': False,
            'died': False,
        }
        return obs, info

    def render(self):
        if self.render_mode == 'rgb_array':
            return self._core.get_screen()
        return None

    def render_rgba(self):
        return self._core.get_screen_rgba()

    def close(self):
        if self._core:
            self._core.close()
            self._core = None

    @property
    def last_action_name(self) -> str:
        return self._last_action_name

    @property
    def last_action_keys(self) -> int:
        return self._last_action
