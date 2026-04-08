# MMBN RL Agent

Reinforcement learning agent that learns to play MegaMan Battle Network (GBA) using deep Q-learning.

## Stack

| Component | Tool |
|-----------|------|
| Emulator | stable-retro (VBA-M core) |
| RL Framework | Stable-Baselines3 (DQN + CnnPolicy) |
| ML Backend | PyTorch |
| Logging | TensorBoard |

## How It Works

```
ROM (GBA file)
    ↓
stable-retro (emulator wrapper)
    ↓
OpenAI Gym Environment  ←→  DQN Agent (CNN policy)
    ↓                              ↓
 state (pixels)              action (button press)
    ↓                              ↓
        reward signal ←←←←←←←←←←←
```

The agent observes raw game pixels, passes them through a CNN to extract spatial features, and outputs button presses. A custom reward function scores battle performance (damage dealt, HP preserved, enemies defeated).

## Setup

```bash
# Create conda environment (stable-retro requires Python <3.13)
conda create -y -p .venv python=3.12
conda activate .venv/

# Install dependencies
pip install -r requirements.txt

# Import your ROM (you need to provide your own .gba file)
python scripts/import_rom.py path/to/MegaManBattleNetwork.gba
```

## Usage

```bash
# Train the agent
python scripts/train.py

# Train with custom timesteps
python scripts/train.py --timesteps 500000

# Evaluate a trained model
python scripts/evaluate.py

# Monitor training
tensorboard --logdir logs/
```

## Project Structure

```
src/
  env/          Custom Gym wrappers and reward functions
  agent/        RL agent config and training logic
  utils/        Screen processing and helpers
scripts/        CLI entry points
tests/          Unit tests
models/         Saved checkpoints (gitignored)
logs/           TensorBoard logs (gitignored)
roms/           ROM files (gitignored)
```

## Complexity

- **Time:** Environment step = O(1) per frame | Training scales with `total_timesteps`
- **Space:** Replay buffer dominates — O(buffer_size x frame_size)

## License

MIT
