# MMBN RL Agent

Reinforcement learning agent that learns to play MegaMan Battle Network 6: Cybeast Gregar (GBA) using deep Q-learning with save state training.

## Stack

| Component | Tool |
|-----------|------|
| Emulator | mGBA |
| RL Framework | Stable-Baselines3 (DQN + CnnPolicy) |
| ML Backend | PyTorch |
| Agent UI | pyglet |
| Logging | TensorBoard |

## How It Works

```
Save State (boss fight)
    ↓
mGBA loads state
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

# Install mGBA (macOS)
brew install mgba

# Copy your ROM and save file into roms/
cp path/to/rom.gba roms/
cp path/to/save.sav "roms/Mega Man Battle Network 6 - Cybeast Gregar (USA).sav"
```

## Playing the Game

```bash
python scripts/play.py
python scripts/play.py --scale 5
```

### Controls

| Key | Action |
|-----|--------|
| Arrow keys | D-pad |
| Z | B button |
| X | A button |
| A / S | L / R triggers |
| Enter | Start |
| Backspace | Select |
| Shift+F1-F9 | Save to slot |
| F1-F9 | Load from slot |

### Creating a Save State for Training

1. Run `python scripts/play.py`
2. Continue from your save, navigate to the fight
3. Press **Shift+F1** to save the state
4. mGBA stores the state automatically

## Training

```bash
python scripts/train.py --timesteps 500000
python scripts/evaluate.py
python scripts/progress.py
tensorboard --logdir logs/
```

## Progression Tracking

Training progress is saved to `logs/progress.json` every 50 episodes:

```
Total episodes:    1250
Wins:              312
Deaths:            938
Win rate:          25.0%
Current streak:    3
Best streak:       8
Best reward:       142.50
```

## Project Structure

```
src/
  env/          Reward function and progression tracking
  agent/        DQN config and training logic
  utils/        Screen processing and helpers
scripts/
  play.py       Launch mGBA to play manually
  train.py      Train the RL agent
  evaluate.py   Run a trained agent with rendering
  progress.py   View training stats
models/         Saved checkpoints (gitignored)
logs/           TensorBoard + progress.json (gitignored)
roms/           ROM and save files (gitignored)
```

## License

MIT
