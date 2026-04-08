#!/usr/bin/env python3
import argparse
from src.env.rewards import TrainingProgress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="logs/progress.json")
    parser.add_argument("--last", type=int, default=20)
    args = parser.parse_args()

    progress = TrainingProgress.load(args.path)

    print(f"Total episodes:    {progress.total_episodes}")
    print(f"Wins:              {progress.wins}")
    print(f"Deaths:            {progress.deaths}")
    print(f"Win rate:          {progress.win_rate:.1%}")
    print(f"Current streak:    {progress.win_streak}")
    print(f"Best streak:       {progress.best_win_streak}")
    print(f"Best reward:       {progress.best_reward:.2f}")

    recent = progress.episode_history[-args.last:]
    if recent:
        print(f"\nLast {len(recent)} episodes:")
        print(f"{'EP':>5}  {'REWARD':>8}  {'STEPS':>6}  {'DMG DEALT':>10}  {'DMG TAKEN':>10}  {'RESULT':>6}")
        offset = max(0, progress.total_episodes - len(recent))
        for i, ep in enumerate(recent):
            result = "WIN" if ep["won"] else "DEAD" if ep["died"] else "-"
            print(
                f"{offset + i + 1:>5}  "
                f"{ep['reward_total']:>8.2f}  "
                f"{ep['steps']:>6}  "
                f"{ep['damage_dealt']:>10.1f}  "
                f"{ep['damage_taken']:>10.1f}  "
                f"{result:>6}"
            )


if __name__ == "__main__":
    main()
