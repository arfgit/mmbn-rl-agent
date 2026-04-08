import retro
import gymnasium as gym
import numpy as np
import cv2
from src.env.rewards import BattleRewardFunction


class MmbnWrapper(gym.Wrapper):
    """Wraps the gym-retro MMBN environment with preprocessing and custom rewards."""

    def __init__(self, env: gym.Env, frame_size: tuple[int, int] = (84, 84)):
        super().__init__(env)
        self.frame_size = frame_size
        self.reward_fn = BattleRewardFunction()
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(frame_size[1], frame_size[0], 1), dtype=np.uint8
        )

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, self.frame_size, interpolation=cv2.INTER_AREA)
        return resized[:, :, np.newaxis]

    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)
        reward = self.reward_fn.calculate(info)
        return self._preprocess_frame(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.reward_fn.reset()
        return self._preprocess_frame(obs), info


def make_mmbn_env(game: str = "MegaManBattleNetwork-GBA", render_mode: str | None = None) -> gym.Env:
    """Create a wrapped MMBN environment ready for training."""
    env = retro.make(game=game, render_mode=render_mode)
    return MmbnWrapper(env)
