import retro
import gymnasium as gym
import numpy as np
import cv2
from src.env.rewards import BattleRewardFunction, TrainingProgress


class MmbnWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, frame_size: tuple[int, int] = (84, 84)):
        super().__init__(env)
        self.frame_size = frame_size
        self.reward_fn = BattleRewardFunction()
        self.progress = TrainingProgress.load()
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
        episode_stats = self.reward_fn.reset()
        if episode_stats.steps > 0:
            self.progress.record(episode_stats)
            if self.progress.total_episodes % 50 == 0:
                self.progress.save()

        obs, info = self.env.reset(**kwargs)
        return self._preprocess_frame(obs), info

    def close(self):
        episode_stats = self.reward_fn.reset()
        if episode_stats.steps > 0:
            self.progress.record(episode_stats)
        self.progress.save()
        super().close()


def make_mmbn_env(
    game: str = "MegaManBattleNetwork-GBA",
    state: str | None = None,
    render_mode: str | None = None,
) -> gym.Env:
    kwargs = {"game": game, "render_mode": render_mode}
    if state:
        kwargs["state"] = state
    env = retro.make(**kwargs)
    return MmbnWrapper(env)
