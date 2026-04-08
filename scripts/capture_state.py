#!/usr/bin/env python3
import argparse
import retro
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="MegaManBattleNetwork-GBA")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--frames", type=int, default=0)
    args = parser.parse_args()

    env = retro.make(game=args.game, render_mode="human")
    obs = env.reset()

    if args.frames > 0:
        print(f"Advancing {args.frames} frames...")
        for i in range(args.frames):
            obs, _, done, _, _ = env.step(env.action_space.sample())
            if done:
                obs = env.reset()

    print("Press Ctrl+C when you're at the desired state to capture it.")
    try:
        while True:
            obs, _, done, _, _ = env.step(0)
            env.render()
            if done:
                break
    except KeyboardInterrupt:
        pass

    state = env.em.get_state()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        f.write(state)
    print(f"State saved to {output}")

    game_dir = Path(retro.data.path()) / "stable" / args.game
    if game_dir.exists():
        retro_copy = game_dir / output.name
        with open(retro_copy, "wb") as f:
            f.write(state)
        print(f"State also copied to {retro_copy}")

    env.close()


if __name__ == "__main__":
    main()
