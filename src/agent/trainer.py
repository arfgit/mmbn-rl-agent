from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from sb3_contrib import RecurrentPPO, QRDQN
import gymnasium as gym
import torch

ALGO_MAP = {
    "dqn": DQN,
    "qrdqn": QRDQN,
    "ppo": PPO,
    "rppo": RecurrentPPO,
}

OFF_POLICY_ALGOS = {"dqn", "qrdqn"}


def create_agent(env: gym.Env, algo: str = "ppo", **kwargs) -> DQN | PPO | RecurrentPPO | QRDQN:
    if algo in ("ppo", "rppo"):
        defaults = {
            "policy": "CnnLstmPolicy" if algo == "rppo" else "CnnPolicy",
            "env": env,
            "verbose": 1,
            "n_steps": 512,
            "batch_size": 64,
            "n_epochs": 4,
            "learning_rate": 2.5e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "tensorboard_log": "logs/",
            "policy_kwargs": {
                "normalize_images": True,
            },
        }
    else:
        defaults = {
            "policy": "CnnPolicy",
            "env": env,
            "verbose": 1,
            "buffer_size": 200_000,
            "learning_starts": 25_000,
            "batch_size": 64,
            "learning_rate": 5e-5,
            "gamma": 0.99,
            "exploration_fraction": 0.4,
            "exploration_final_eps": 0.02,
            "target_update_interval": 2000,
            "train_freq": 4,
            "gradient_steps": 1,
            "tensorboard_log": "logs/",
            "policy_kwargs": {
                "net_arch": [256, 256],
                "normalize_images": True,
            },
        }

    if torch.backends.mps.is_available():
        defaults["device"] = "mps"
    elif torch.cuda.is_available():
        defaults["device"] = "cuda"

    defaults.update(kwargs)
    policy = defaults.pop("policy")
    env_arg = defaults.pop("env")

    cls = ALGO_MAP[algo]
    return cls(policy, env_arg, **defaults)


def train(
    env: gym.Env,
    total_timesteps: int = 1_000_000,
    checkpoint_freq: int = 25_000,
    model_path: str = "models/mmbn",
    algo: str = "ppo",
    resume_path: str | None = None,
    eval_env: gym.Env | None = None,
) -> DQN | PPO | RecurrentPPO | QRDQN:
    if resume_path:
        cls = ALGO_MAP[algo]
        load_kwargs: dict = {"env": env}
        if algo in OFF_POLICY_ALGOS:
            load_kwargs["custom_objects"] = {"learning_starts": 0}
        model = cls.load(resume_path, **load_kwargs)
    else:
        model = create_agent(env, algo=algo)

    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path="models/checkpoints/",
        name_prefix=f"mmbn_{algo}",
    )

    eval_cb = EvalCallback(
        eval_env if eval_env is not None else env,
        best_model_save_path="models/best/",
        log_path="logs/eval/",
        eval_freq=checkpoint_freq,
        n_eval_episodes=10,
        deterministic=True,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, eval_cb],
        reset_num_timesteps=resume_path is None,
    )
    model.save(model_path)

    if algo in OFF_POLICY_ALGOS and hasattr(model, 'save_replay_buffer'):
        model.save_replay_buffer(f"{model_path}_replay_buffer")

    return model
