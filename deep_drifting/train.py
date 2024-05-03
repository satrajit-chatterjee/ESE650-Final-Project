from typing import Optional
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

import wandb
from deep_drifting.config import load_env_config, load_model_config, EnvConfig
from deep_drifting.environments.drifting import DeepDriftingEnv, wrap_env
from deep_drifting.schedulers import linear_schedule

from f1tenth_gym.envs import F110Env

def eval_wrap_env(env_config: EnvConfig, render_mode: Optional[str] = None, max_episode_steps: Optional[int] = 100000):
    env : F110Env = gym.make(
        "f1tenth_gym:f1tenth-v0",
         config = {
            "num_agents": 1,
            "observation_config": {
                "type": "dynamic_state"
            },
            "params": {
                "mu": env_config.mu
            },
            "reset_config": {
                "type": "cl_grid_static"
            },
            "map": env_config.map
         },
         render_mode=render_mode,
    )
    env = DeepDriftingEnv(env, **asdict(env_config))
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--model-config", type=str)
    parser.add_argument("-e", "--env-config", type=str)
    parser.add_argument("-t", "--timesteps", type=int)
    parser.add_argument("-n", "--num-envs", type=int, default=6)
    parser.add_argument("-i", "--id", type=str)
    args = parser.parse_args()

    model_config = load_model_config(args.model_config)
    env_config = load_env_config(args.env_config)

    config = {
        "env_name": "DeepDrifting",
        **asdict(model_config),
        **asdict(env_config),
    }

    resume = None
    if args.id is not None and Path(f"runs/{args.id}").is_dir():
        resume = "allow"

    run = wandb.init(
        project="sb3_new",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        id=args.id,
        resume=resume
    )

    if args.timesteps is not None:
        model_config.timesteps = args.timesteps

    if args.env_config is not None:
        wandb.save(args.env_config, policy="now")

    train_env = make_vec_env(wrap_env, n_envs=args.num_envs, seed=42, env_kwargs={"env_config": env_config})
    if Path(f"models/{run.id}").is_dir():
        model = PPO.load(f"models/{run.id}/best_model.zip", tensorboard_log=f"runs/{run.id}", device=model_config.device)
        model.set_env(train_env)
    else:
        model = PPO(
            model_config.policy,
            train_env,
            n_steps=model_config.n_steps,
            learning_rate=linear_schedule(model_config.lr),
            gamma=model_config.gamma,
            gae_lambda=model_config.gae_lambda,
            verbose=1,
            policy_kwargs={
                "net_arch": model_config.net_arch
            },
            device=model_config.device,
            tensorboard_log=f"runs/{run.id}"
        )

    eval_env = make_vec_env(eval_wrap_env, n_envs=args.num_envs, seed=42, env_kwargs={"env_config": env_config})

    eval_callback = EvalCallback(
        eval_env, best_model_save_path=f"models/{run.id}",
        log_path="./logs/", eval_freq=10000,
        deterministic=True, render=False
    )

    model.learn(
        total_timesteps=model_config.timesteps,
        progress_bar=True,
        callback=CallbackList([
            eval_callback,
            WandbCallback(verbose=2)
        ])
    )