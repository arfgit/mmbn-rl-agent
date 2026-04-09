#!/usr/bin/env python3
import argparse
from pathlib import Path

from src.env import MmbnEnv
from src.agent.trainer import train


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROM_PATH = str(PROJECT_ROOT / 'roms' / 'Mega Man Battle Network 6 - Cybeast Gregar (USA).gba')
SAVE_PATH = str(PROJECT_ROOT / 'roms' / 'Mega Man Battle Network 6 - Cybeast Gregar (USA).sav')


def make_env(args) -> MmbnEnv:
    save_path = SAVE_PATH if Path(SAVE_PATH).exists() else None
    return MmbnEnv(
        rom_path=ROM_PATH,
        save_path=save_path,
        state_path=args.state,
        frame_skip=args.frame_skip,
        frame_stack=args.frame_stack,
        crop_battle=not args.no_crop,
        sticky_action_prob=args.sticky,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--checkpoint-freq", type=int, default=25_000)
    parser.add_argument("--model-path", type=str, default="models/mmbn")
    parser.add_argument("--state", type=str, default="1")
    parser.add_argument("--algo", type=str, default="ppo", choices=["dqn", "qrdqn", "ppo", "rppo"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--frame-skip", type=int, default=12)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--sticky", type=float, default=0.25)
    args = parser.parse_args()

    train_env = make_env(args)
    eval_env = make_env(args)

    model = train(
        train_env,
        total_timesteps=args.timesteps,
        checkpoint_freq=args.checkpoint_freq,
        model_path=args.model_path,
        algo=args.algo,
        resume_path=args.resume,
        eval_env=eval_env,
    )
    print(f"Training complete. Model saved to {args.model_path}")
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
