class BattleRewardFunction:
    """Custom reward function for MMBN battle system.

    Reward signals:
        - Positive: dealing damage to enemies, defeating enemies, winning battles
        - Negative: losing HP, dying
        - Neutral: time passing (small negative to encourage aggression)

    Memory addresses for HP, enemy HP, etc. need to be mapped in the
    game's data.json integration file for gym-retro.
    """

    def __init__(self):
        self.prev_hp = 0
        self.prev_enemy_hp = 0
        self.prev_score = 0

    def reset(self) -> None:
        self.prev_hp = 0
        self.prev_enemy_hp = 0
        self.prev_score = 0

    def calculate(self, info: dict) -> float:
        reward = 0.0

        # HP-based rewards (requires memory mapping in data.json)
        hp = info.get("hp", 0)
        enemy_hp = info.get("enemy_hp", 0)
        score = info.get("score", 0)

        # Reward for dealing damage
        if self.prev_enemy_hp > 0:
            damage_dealt = self.prev_enemy_hp - enemy_hp
            if damage_dealt > 0:
                reward += damage_dealt * 1.0

        # Penalty for taking damage
        if self.prev_hp > 0:
            damage_taken = self.prev_hp - hp
            if damage_taken > 0:
                reward -= damage_taken * 0.5

        # Reward for score increase (battle wins, etc.)
        score_delta = score - self.prev_score
        if score_delta > 0:
            reward += score_delta * 2.0

        # Small time penalty to encourage action
        reward -= 0.01

        self.prev_hp = hp
        self.prev_enemy_hp = enemy_hp
        self.prev_score = score

        return reward
