from argparse import ArgumentParser
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

from deep_drifting.environments.drifting import wrap_env
from f1tenth_gym.envs import F110Env
from deep_drifting.config import EnvConfig, load_env_config

import yaml


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-e", "--env-config", type=str)
    parser.add_argument("--map", type=str)
    args = parser.parse_args()

    env_config = load_env_config(args.env_config)

    if args.map is not None:
        env_config.map = args.map
    eval_env = wrap_env(env_config, render_mode="human")

    model_path = Path(args.model).absolute()
    model = PPO.load(model_path)

    obs, info = eval_env.reset()
    eval_env.render()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = eval_env.step(action)
        print(reward)
        frame = eval_env.render()
