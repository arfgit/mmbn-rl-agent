#!/usr/bin/env python3
"""Evaluate a trained MMBN RL agent with rendering."""

import argparse
from stable_baselines3 import DQN
from src.env import make_mmbn_env


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained MMBN agent")
    parser.add_argument("--model-path", type=str, default="models/mmbn_dqn.zip", help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    args = parser.parse_args()

    env = make_mmbn_env(render_mode="human")
    model = DQN.load(args.model_path, env=env)

    for ep in range(args.episodes):
        obs, _info = env.reset()
        total_reward = 0.0
        done = False
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        print(f"Episode {ep + 1}: reward={total_reward:.2f}, steps={steps}")

    env.close()


if __name__ == "__main__":
    main()
