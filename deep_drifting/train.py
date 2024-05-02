from argparse import ArgumentParser
from dataclasses import asdict

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback

import wandb
from deep_drifting.config import load_env_config, load_model_config
from deep_drifting.environments.drifting import wrap_env
from deep_drifting.schedulers import linear_schedule

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--model-config", type=str)
    parser.add_argument("-e", "--env-config", type=str)
    parser.add_argument("-t", "--timesteps", type=int)
    parser.add_argument("-i", "--id", type=str)
    args = parser.parse_args()

    model_config = load_model_config(args.model_config)
    env_config = load_env_config(args.env_config)

    config = {
        "env_name": "DeepDrifting",
        **asdict(model_config),
        **asdict(env_config),
    }

    run = wandb.init(
        project="sb3_new",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        id=args.id,
    )

    if args.env_config is not None:
        wandb.save(args.env_config)

    vec_env = make_vec_env(wrap_env, n_envs=6, seed=42, env_kwargs={"env_config": env_config})
    model = PPO(
        model_config.policy,
        vec_env,
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

    eval_callback = EvalCallback(
        vec_env, best_model_save_path=f"models/{run.id}",
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