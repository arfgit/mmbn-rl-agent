#!/usr/bin/env python3
import argparse
from src.env import make_mmbn_env
from src.agent.trainer import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--checkpoint-freq", type=int, default=10_000)
    parser.add_argument("--model-path", type=str, default="models/mmbn_dqn")
    parser.add_argument("--state", type=str, default=None)
    args = parser.parse_args()

    env = make_mmbn_env(state=args.state)
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
