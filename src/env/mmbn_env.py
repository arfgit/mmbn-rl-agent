import gymnasium as gym
import numpy as np
import cv2
from collections import deque
from gymnasium import spaces

from src.env.mgba_core import MgbaCore, GBA_W, GBA_H, KEY_NAMES
from src.env.rewards import BattleRewardFunction

ADDR_PLAYER_HP = 0x0203A9D4
ADDR_ENEMY_HP = 0x0203AAAC
ADDR_BEAST_OUT = 0x0203A9F0
ADDR_PLAYER_COL = 0x020364C7
ADDR_PLAYER_ROW = 0x020364DB
ADDR_CUSTOM_GAUGE = 0x020364C1
ADDR_CHIP_SCREEN = 0x020364C2
ADDR_ATTACK_STATE = 0x020364C8
ADDR_BUSTER_COUNT = 0x020365C7

BATTLE_CROP = (8, 24, 232, 152)

ACTION_IDX_NOOP = 0
ACTION_IDX_A = 1
ACTION_IDX_START = 3
ACTION_IDX_L = 16
ACTION_IDX_R = 17

ATTACK_ACTIONS = {1, 8, 9, 10, 11}
CHIP_ACTIONS = {16, 17}
COOLDOWN_ACTIONS = ATTACK_ACTIONS | CHIP_ACTIONS

ACTIONS = [
    0,
    (1 << 0),
    (1 << 1),
    (1 << 3),
    (1 << 6),
    (1 << 7),
    (1 << 5),
    (1 << 4),
    (1 << 0) | (1 << 6),
    (1 << 0) | (1 << 7),
    (1 << 0) | (1 << 5),
    (1 << 0) | (1 << 4),
    (1 << 1) | (1 << 6),
    (1 << 1) | (1 << 7),
    (1 << 1) | (1 << 5),
    (1 << 1) | (1 << 4),
    (1 << 8),
    (1 << 9),
]

ACTION_NAMES = [
    'NOOP', 'A', 'B', 'START',
    'UP', 'DOWN', 'LEFT', 'RIGHT',
    'A+UP', 'A+DOWN', 'A+LEFT', 'A+RIGHT',
    'B+UP', 'B+DOWN', 'B+LEFT', 'B+RIGHT',
    'L', 'R',
]

STICKY_PROB = 0.25
COOLDOWN_FRAMES = 3


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
        frame_skip: int = 12,
        frame_size: tuple[int, int] = (84, 84),
        max_episode_steps: int = 4500,
        frame_stack: int = 4,
        crop_battle: bool = True,
        sticky_action_prob: float = STICKY_PROB,
    ):
        super().__init__()
        self.rom_path = rom_path
        self.save_path = save_path
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.frame_size = frame_size
        self.max_episode_steps = max_episode_steps
        self._frame_stack_size = frame_stack
        self._crop_battle = crop_battle
        self._sticky_prob = sticky_action_prob

        self.action_space = spaces.Discrete(len(ACTIONS))
        n_channels = frame_stack + 2
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(frame_size[1], frame_size[0], n_channels),
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

        self._reward_fn = BattleRewardFunction(max_episode_steps, len(ACTIONS))
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=frame_stack)
        self._steps = 0
        self._total_reward = 0.0
        self._last_action_idx = 0
        self._last_action_name = 'NOOP'
        self._prev_action_idx = 0
        self._cooldown_remaining = 0
        self._prev_player_hp = 0
        self._prev_enemy_hp = 0
        self._initial_player_hp = 0
        self._initial_enemy_hp = 0
        self._won = False
        self._died = False
        self._action_counts: dict[int, int] = {}
        self._consecutive_noop = 0
        self._prev_beast_out = False
        self._beast_out_rewarded = False
        self._unique_actions_used: set[int] = set()
        self._prev_col = 0
        self._prev_row = 0
        self._move_count = 0
        self._prev_buster_count = 0
        self._prev_custom_gauge = 0
        self._positions_visited: set[tuple[int, int]] = set()

    def _read_hp(self) -> tuple[int, int]:
        return self._core.read16(ADDR_PLAYER_HP), self._core.read16(ADDR_ENEMY_HP)

    def _read_beast_out(self) -> bool:
        return self._core.read8(ADDR_BEAST_OUT) != 0

    def _read_position(self) -> tuple[int, int]:
        col = self._core.read8(ADDR_PLAYER_COL)
        row = self._core.read8(ADDR_PLAYER_ROW)
        if col > 5:
            col = 0
        if row > 5:
            row = 0
        return col, row

    def _read_custom_gauge(self) -> int:
        return self._core.read8(ADDR_CUSTOM_GAUGE)

    def _read_chip_screen_open(self) -> bool:
        return self._core.read8(ADDR_CHIP_SCREEN) > 0

    def _read_buster_count(self) -> int:
        return self._core.read8(ADDR_BUSTER_COUNT)

    def _crop_frame(self, frame: np.ndarray) -> np.ndarray:
        if not self._crop_battle:
            return frame
        x1, y1, x2, y2 = BATTLE_CROP
        return frame[y1:y2, x1:x2]

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        cropped = self._crop_frame(frame)
        gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, self.frame_size, interpolation=cv2.INTER_AREA)

    def _build_obs(self, player_hp: int, enemy_hp: int) -> np.ndarray:
        stacked = np.stack(list(self._frame_buffer), axis=-1)

        hp_channel = np.full(self.frame_size[::-1], 0, dtype=np.uint8)
        if self._initial_player_hp > 0:
            hp_channel[:] = min(255, int((player_hp / self._initial_player_hp) * 255))

        enemy_channel = np.full(self.frame_size[::-1], 0, dtype=np.uint8)
        if self._initial_enemy_hp > 0:
            enemy_channel[:] = min(255, int((enemy_hp / self._initial_enemy_hp) * 255))

        return np.concatenate([stacked, hp_channel[:, :, np.newaxis], enemy_channel[:, :, np.newaxis]], axis=-1)

    def _resolve_action(self, action: int) -> int:
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return ACTION_IDX_NOOP

        if self._sticky_prob > 0 and self._steps > 0 and np.random.random() < self._sticky_prob:
            return self._prev_action_idx

        return action

    def step(self, action: int):
        effective_action = self._resolve_action(action)

        keys = ACTIONS[effective_action]
        self._last_action_idx = effective_action
        self._last_action_name = ACTION_NAMES[effective_action]

        self._core.set_keys(keys)
        for _ in range(self.frame_skip):
            self._core.run_frame()
        self._core.set_keys(0)

        if effective_action in COOLDOWN_ACTIONS:
            self._cooldown_remaining = COOLDOWN_FRAMES

        screen = self._core.get_screen()
        gray = self._preprocess(screen)
        self._steps += 1

        player_hp, enemy_hp = self._read_hp()
        beast_out = self._read_beast_out()
        col, row = self._read_position()
        custom_gauge = self._read_custom_gauge()
        buster_count = self._read_buster_count()

        self._action_counts[effective_action] = self._action_counts.get(effective_action, 0) + 1
        self._unique_actions_used.add(effective_action)

        if effective_action == ACTION_IDX_NOOP:
            self._consecutive_noop += 1
        else:
            self._consecutive_noop = 0

        moved = col != self._prev_col or row != self._prev_row
        if moved:
            self._move_count += 1
            self._positions_visited.add((col, row))

        reward, newly_beast = self._reward_fn.step_reward(
            prev_player_hp=self._prev_player_hp,
            prev_enemy_hp=self._prev_enemy_hp,
            player_hp=player_hp,
            enemy_hp=enemy_hp,
            effective_action=effective_action,
            prev_action=self._prev_action_idx,
            moved=moved,
            custom_gauge=custom_gauge,
            prev_custom_gauge=self._prev_custom_gauge,
            buster_count=buster_count,
            prev_buster_count=self._prev_buster_count,
            beast_out=beast_out,
            prev_beast_out=self._prev_beast_out,
            beast_out_rewarded=self._beast_out_rewarded,
            consecutive_noop=self._consecutive_noop,
            unique_actions=len(self._unique_actions_used),
            positions_visited=len(self._positions_visited),
            steps=self._steps,
            is_chip_action=effective_action in CHIP_ACTIONS,
            is_start_action=effective_action == ACTION_IDX_START,
            noop_idx=ACTION_IDX_NOOP,
        )
        if newly_beast:
            self._beast_out_rewarded = True

        self._prev_col = col
        self._prev_row = row
        self._prev_buster_count = buster_count
        self._prev_custom_gauge = custom_gauge
        self._prev_action_idx = effective_action

        self._frame_buffer.append(gray)
        obs = self._build_obs(player_hp, enemy_hp)

        terminated = False
        in_battle = self._prev_player_hp > 0 and self._prev_enemy_hp > 0 and self._steps > 5

        if in_battle and enemy_hp == 0:
            time_bonus = max(0, 50.0 * (1.0 - self._steps / self.max_episode_steps))
            hp_bonus = 30.0 * (player_hp / max(1, self._initial_player_hp))
            diversity_bonus = min(10.0, len(self._unique_actions_used) * 0.5)
            reward += 100.0 + time_bonus + hp_bonus + diversity_bonus
            self._won = True
            terminated = True

        elif in_battle and player_hp == 0:
            progress_credit = 20.0 * (self._reward_fn.damage_dealt / max(1, self._initial_enemy_hp))
            reward += -50.0 + progress_credit
            self._died = True
            terminated = True

        self._prev_player_hp = player_hp
        self._prev_enemy_hp = enemy_hp
        self._prev_beast_out = beast_out
        self._total_reward += reward
        truncated = self._steps >= self.max_episode_steps

        info = {
            'frame': self._core.frame_counter,
            'steps': self._steps,
            'action_name': self._last_action_name,
            'effective_action': effective_action,
            'requested_action': action,
            'total_reward': self._total_reward,
            'player_hp': player_hp,
            'enemy_hp': enemy_hp,
            'damage_dealt': self._reward_fn.damage_dealt,
            'damage_taken': self._reward_fn.damage_taken,
            'won': self._won,
            'died': self._died,
            'beast_out': beast_out,
            'chip_opens': self._reward_fn.chip_opens,
            'unique_actions': len(self._unique_actions_used),
            'col': col,
            'row': row,
            'moves': self._move_count,
            'positions_visited': len(self._positions_visited),
            'custom_gauge': custom_gauge,
            'buster_shots': buster_count,
            'cooldown': self._cooldown_remaining,
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
        self._last_action_idx = 0
        self._last_action_name = 'NOOP'
        self._prev_action_idx = 0
        self._cooldown_remaining = 0
        self._won = False
        self._died = False
        self._action_counts = {}
        self._consecutive_noop = 0
        self._prev_beast_out = False
        self._beast_out_rewarded = False
        self._unique_actions_used = set()
        self._prev_col = 0
        self._prev_row = 0
        self._move_count = 0
        self._prev_buster_count = 0
        self._prev_custom_gauge = 0
        self._positions_visited = set()
        self._reward_fn.reset()

        for _ in range(10):
            self._core.run_frame()

        self._prev_player_hp, self._prev_enemy_hp = self._read_hp()
        self._initial_player_hp = self._prev_player_hp
        self._initial_enemy_hp = self._prev_enemy_hp
        self._prev_col, self._prev_row = self._read_position()
        self._prev_buster_count = self._read_buster_count()
        self._prev_custom_gauge = self._read_custom_gauge()
        self._positions_visited = {(self._prev_col, self._prev_row)}

        screen = self._core.get_screen()
        gray = self._preprocess(screen)

        self._frame_buffer.clear()
        for _ in range(self._frame_stack_size):
            self._frame_buffer.append(gray)

        obs = self._build_obs(self._prev_player_hp, self._prev_enemy_hp)
        info = {
            'frame': self._core.frame_counter,
            'steps': 0,
            'action_name': 'NOOP',
            'effective_action': 0,
            'requested_action': 0,
            'total_reward': 0.0,
            'player_hp': self._prev_player_hp,
            'enemy_hp': self._prev_enemy_hp,
            'damage_dealt': 0.0,
            'damage_taken': 0.0,
            'won': False,
            'died': False,
            'beast_out': False,
            'chip_opens': 0,
            'unique_actions': 0,
            'col': self._prev_col,
            'row': self._prev_row,
            'moves': 0,
            'positions_visited': 1,
            'custom_gauge': self._prev_custom_gauge,
            'buster_shots': self._prev_buster_count,
            'cooldown': 0,
        }
        return obs, info

    def render(self):
        if self.render_mode == 'rgb_array':
            return self._core.get_screen()
        return None

    def render_rgba(self):
        return self._core.get_screen_rgba()

    def close(self) -> None:
        if self._core:
            self._core.close()
            self._core = None

    @property
    def last_action_name(self) -> str:
        return self._last_action_name

    @property
    def last_action_keys(self) -> int:
        return ACTIONS[self._last_action_idx]
