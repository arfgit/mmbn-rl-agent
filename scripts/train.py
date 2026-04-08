#!/usr/bin/env python3
import argparse
from pathlib import Path

from src.env import MmbnEnv
from src.agent.trainer import train


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROM_PATH = str(PROJECT_ROOT / 'roms' / 'Mega Man Battle Network 6 - Cybeast Gregar (USA).gba')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--checkpoint-freq", type=int, default=10_000)
    parser.add_argument("--model-path", type=str, default="models/mmbn_dqn")
    args = parser.parse_args()

    env = MmbnEnv(rom_path=ROM_PATH)
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
