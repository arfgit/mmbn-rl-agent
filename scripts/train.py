#!/usr/bin/env python3
import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--checkpoint-freq", type=int, default=10_000)
    parser.add_argument("--model-path", type=str, default="models/mmbn_dqn")
    parser.add_argument("--state", type=str, default=None)
    args = parser.parse_args()

    print("Training requires the mGBA gym environment (coming soon)")
    print(f"Would train for {args.timesteps} timesteps")
    sys.exit(1)


if __name__ == "__main__":
    main()
