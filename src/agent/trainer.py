from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import gymnasium as gym


def create_agent(env: gym.Env, **kwargs) -> DQN:
    """Create a DQN agent with CNN policy for pixel-based observations."""
    defaults = {
        "policy": "CnnPolicy",
        "env": env,
        "verbose": 1,
        "buffer_size": 50_000,
        "learning_starts": 10_000,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.01,
        "target_update_interval": 1000,
        "tensorboard_log": "logs/",
    }
    defaults.update(kwargs)

    policy = defaults.pop("policy")
    env = defaults.pop("env")
    return DQN(policy, env, **defaults)


def train(
    env: gym.Env,
    total_timesteps: int = 100_000,
    checkpoint_freq: int = 10_000,
    model_path: str = "models/mmbn_dqn",
) -> DQN:
    """Train a DQN agent and save checkpoints."""
    model = create_agent(env)

    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path="models/checkpoints/",
        name_prefix="mmbn_dqn",
    )

    eval_cb = EvalCallback(
        env,
        best_model_save_path="models/best/",
        log_path="logs/eval/",
        eval_freq=checkpoint_freq,
        n_eval_episodes=5,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, eval_cb],
    )
    model.save(model_path)
    return model
