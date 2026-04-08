#!/usr/bin/env python3
"""Train the MMBN RL agent."""

import argparse
from src.env import make_mmbn_env
from src.agent.trainer import train


def main():
    parser = argparse.ArgumentParser(description="Train MMBN RL agent")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total training timesteps")
    parser.add_argument("--checkpoint-freq", type=int, default=10_000, help="Checkpoint save frequency")
    parser.add_argument("--model-path", type=str, default="models/mmbn_dqn", help="Final model save path")
    args = parser.parse_args()

    env = make_mmbn_env()
    model = train(
        env,
        total_timesteps=args.timesteps,
        checkpoint_freq=args.checkpoint_freq,
        model_path=args.model_path,
    )
    print(f"Training complete. Model saved to {args.model_path}")
    env.close()


if __name__ == "__main__":
    main()
