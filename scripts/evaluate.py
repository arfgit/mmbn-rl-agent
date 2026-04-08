#!/usr/bin/env python3
import argparse
from stable_baselines3 import DQN
from src.env import make_mmbn_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/mmbn_dqn.zip")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--state", type=str, default=None)
    args = parser.parse_args()

    env = make_mmbn_env(state=args.state, render_mode="human")
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
