import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class EpisodeStats:
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    reward_total: float = 0.0
    steps: int = 0
    won: bool = False
    died: bool = False


@dataclass
class TrainingProgress:
    total_episodes: int = 0
    wins: int = 0
    deaths: int = 0
    best_reward: float = float("-inf")
    win_streak: int = 0
    best_win_streak: int = 0
    episode_history: list[dict] = field(default_factory=list)

    def record(self, stats: EpisodeStats) -> None:
        self.total_episodes += 1
        if stats.won:
            self.wins += 1
            self.win_streak += 1
            self.best_win_streak = max(self.best_win_streak, self.win_streak)
        else:
            self.win_streak = 0
        if stats.died:
            self.deaths += 1
        if stats.reward_total > self.best_reward:
            self.best_reward = stats.reward_total

        self.episode_history.append(asdict(stats))
        if len(self.episode_history) > 1000:
            self.episode_history = self.episode_history[-500:]

    @property
    def win_rate(self) -> float:
        if self.total_episodes == 0:
            return 0.0
        return self.wins / self.total_episodes

    def save(self, path: str = "logs/progress.json") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str = "logs/progress.json") -> "TrainingProgress":
        p = Path(path)
        if not p.exists():
            return cls()
        with open(p) as f:
            data = json.load(f)
        history = data.pop("episode_history", [])
        progress = cls(**data)
        progress.episode_history = history
        return progress


class BattleRewardFunction:
    def __init__(self):
        self.prev_hp = 0
        self.prev_enemy_hp = 0
        self.stats = EpisodeStats()

    def reset(self) -> EpisodeStats:
        result = self.stats
        self.prev_hp = 0
        self.prev_enemy_hp = 0
        self.stats = EpisodeStats()
        return result

    def calculate(self, info: dict) -> float:
        reward = 0.0
        hp = info.get("hp", 0)
        enemy_hp = info.get("enemy_hp", 0)

        if self.prev_enemy_hp > 0:
            damage_dealt = self.prev_enemy_hp - enemy_hp
            if damage_dealt > 0:
                reward += damage_dealt * 1.0
                self.stats.damage_dealt += damage_dealt

        if self.prev_hp > 0:
            damage_taken = self.prev_hp - hp
            if damage_taken > 0:
                reward -= damage_taken * 0.5
                self.stats.damage_taken += damage_taken

        if enemy_hp == 0 and self.prev_enemy_hp > 0:
            reward += 100.0
            self.stats.won = True

        if hp == 0 and self.prev_hp > 0:
            reward -= 50.0
            self.stats.died = True

        reward -= 0.01
        self.stats.steps += 1
        self.stats.reward_total += reward

        self.prev_hp = hp
        self.prev_enemy_hp = enemy_hp

        return reward
