import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from deep_drifting.environments.drifting import DeepDriftingEnv
from deep_drifting.schedulers import linear_schedule
from deep_drifting.callbacks import SaveOnBestTrainingRewardCallback

import wandb
from wandb.integration.sb3 import WandbCallback

from f1tenth_gym.envs import F110Env

def wrap_env():
    env : F110Env = gym.make(
        "f1tenth_gym:f1tenth-v0",
         config = {
            "num_agents": 1,
            "observation_config": {
                "type": "dynamic_state"
            },
            "params": {
                "mu": 0.4
            },
            "map": "Hockenheim",
         },
    )
    env = DeepDriftingEnv(env)
    env = Monitor(env)
    return env

if __name__ == "__main__":

    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 1000000,
        "env_name": "DeepDrifting"
    }

    run = wandb.init(
        project="sb3",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    )
    # TODO(rahul): make params configurable
    vec_env = make_vec_env(wrap_env, n_envs=3, seed=42)
    model = PPO(
        config["policy_type"],
        vec_env,
        learning_rate=linear_schedule(0.0003),
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1,
        device="cuda",
        tensorboard_log=f"runs/{run.id}"
    )

    # eval_callback = EvalCallback(
    #     vec_env, best_model_save_path="./logs/",
    #     log_path="./logs/", eval_freq=5000,
    #     deterministic=True, render=True
    # )

    model.learn(
        total_timesteps=config["total_timesteps"],
        progress_bar=True,
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        )
    )