# MMBN RL Agent

Reinforcement learning agent that learns to play MegaMan Battle Network (GBA) using deep Q-learning with save state training.

## Stack

| Component | Tool |
|-----------|------|
| Emulator | stable-retro (VBA-M core) |
| RL Framework | Stable-Baselines3 (DQN + CnnPolicy) |
| ML Backend | PyTorch |
| Logging | TensorBoard |

## How It Works

```
Save State (boss fight)
    ↓
stable-retro loads state
    ↓
Gym Environment  ←→  DQN Agent (CNN policy)
    ↓                        ↓
 pixels               button press
    ↓                        ↓
    reward ←←←←←←←←←←←←←←←←←
    ↓
win or die → reload save state → next episode
```

The agent trains on a specific battle by loading a save state at the start of every episode. When it wins or dies, the state reloads and it tries again. A reward function tracks damage dealt, damage taken, kills, and deaths.

## Setup

```bash
conda create -y -p .venv python=3.12
conda activate .venv/
pip install -r requirements.txt
python scripts/import_rom.py path/to/MegaManBattleNetwork.gba
```

## Usage

```bash
# Capture a save state at a specific fight
python scripts/capture_state.py --output fight.state

# Train on that fight
python scripts/train.py --state fight.state --timesteps 500000

# Evaluate
python scripts/evaluate.py --state fight.state

# View progression stats
python scripts/progress.py

# TensorBoard
tensorboard --logdir logs/
```

## Progression Tracking

Training progress is saved to `logs/progress.json` every 50 episodes:

- Win/loss record and win rate
- Current and best win streaks
- Damage dealt/taken per episode
- Reward totals and best reward

View it anytime with `python scripts/progress.py`:
```
Total episodes:    1250
Wins:              312
Deaths:            938
Win rate:          25.0%
Current streak:    3
Best streak:       8
Best reward:       142.50

Last 20 episodes:
   EP    REWARD   STEPS   DMG DEALT   DMG TAKEN  RESULT
 1231     42.30     180       120.0        60.0     WIN
 1232    -38.50      95        30.0       150.0    DEAD
 ...
```

## Project Structure

```
src/
  env/          Gym wrappers, reward function, progression tracking
  agent/        DQN config and training logic
  utils/        Screen processing and helpers
scripts/        CLI entry points (train, evaluate, capture, progress)
models/         Saved checkpoints (gitignored)
logs/           TensorBoard + progress.json (gitignored)
roms/           ROM files (gitignored)
```

## License

MIT
