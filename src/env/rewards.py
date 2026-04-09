import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

SWITCH_PENALTY = 0.02


@dataclass
class EpisodeStats:
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    reward_total: float = 0.0
    steps: int = 0
    won: bool = False
    died: bool = False
    time_bonus: float = 0.0
    hp_remaining: float = 0.0


@dataclass
class TrainingProgress:
    total_episodes: int = 0
    wins: int = 0
    deaths: int = 0
    best_reward: float = float("-inf")
    win_streak: int = 0
    best_win_streak: int = 0
    avg_damage_dealt: float = 0.0
    avg_episode_length: float = 0.0
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

        recent = self.episode_history[-50:]
        if recent:
            self.avg_damage_dealt = sum(e["damage_dealt"] for e in recent) / len(recent)
            self.avg_episode_length = sum(e["steps"] for e in recent) / len(recent)

    @property
    def win_rate(self) -> float:
        if self.total_episodes == 0:
            return 0.0
        return self.wins / self.total_episodes

    @property
    def recent_win_rate(self) -> float:
        recent = self.episode_history[-50:]
        if not recent:
            return 0.0
        return sum(1 for e in recent if e["won"]) / len(recent)

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
        data.pop("avg_damage_dealt", None)
        data.pop("avg_episode_length", None)
        progress = cls(**data)
        progress.episode_history = history
        return progress


class BattleRewardFunction:
    def __init__(self, max_episode_steps: int, num_actions: int):
        self._max_steps = max_episode_steps
        self._num_actions = num_actions
        self.damage_dealt = 0.0
        self.damage_taken = 0.0
        self.chip_opens = 0
        self.start_press_count = 0

    def reset(self) -> None:
        self.damage_dealt = 0.0
        self.damage_taken = 0.0
        self.chip_opens = 0
        self.start_press_count = 0

    def step_reward(
        self,
        prev_player_hp: int,
        prev_enemy_hp: int,
        player_hp: int,
        enemy_hp: int,
        effective_action: int,
        prev_action: int,
        moved: bool,
        custom_gauge: int,
        prev_custom_gauge: int,
        buster_count: int,
        prev_buster_count: int,
        beast_out: bool,
        prev_beast_out: bool,
        beast_out_rewarded: bool,
        consecutive_noop: int,
        unique_actions: int,
        positions_visited: int,
        steps: int,
        is_chip_action: bool,
        is_start_action: bool,
        noop_idx: int,
    ) -> tuple[float, bool]:
        time_factor = 1.0 + (steps / self._max_steps) * 0.5
        reward = -0.01 * time_factor

        if prev_enemy_hp > 0 and enemy_hp < prev_enemy_hp:
            dmg = prev_enemy_hp - enemy_hp
            reward += dmg * 1.5
            self.damage_dealt += dmg

        if prev_player_hp > 0 and player_hp < prev_player_hp:
            dmg = prev_player_hp - player_hp
            reward -= dmg * 0.8
            self.damage_taken += dmg

        if effective_action != prev_action and effective_action != noop_idx and prev_action != noop_idx:
            reward -= SWITCH_PENALTY

        if moved:
            took_damage = prev_player_hp > 0 and player_hp < prev_player_hp
            if not took_damage:
                reward += 0.1

        if is_chip_action:
            self.chip_opens += 1
            if custom_gauge > prev_custom_gauge:
                reward += 1.0
            else:
                reward += 0.3

        if buster_count > prev_buster_count:
            reward += 0.3

        if is_start_action:
            self.start_press_count += 1
            if self.start_press_count > 2:
                reward -= 0.5 * (self.start_press_count - 2)

        if consecutive_noop > 8:
            reward -= 0.1 * (consecutive_noop - 8)

        newly_beast = False
        if beast_out and not prev_beast_out and not beast_out_rewarded:
            reward += 20.0
            newly_beast = True

        if steps > 0 and steps % 100 == 0:
            diversity = unique_actions / self._num_actions
            if diversity > 0.25:
                reward += 1.5 * diversity
            if positions_visited >= 3:
                reward += 0.5

        return reward, newly_beast
