#!/usr/bin/env python3
import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/mmbn_dqn.zip")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--state", type=str, default=None)
    args = parser.parse_args()

    print("Evaluation requires the mGBA gym environment (coming soon)")
    sys.exit(1)


if __name__ == "__main__":
    main()
