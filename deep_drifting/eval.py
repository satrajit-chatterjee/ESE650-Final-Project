from argparse import ArgumentParser
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

from deep_drifting.environments.drifting import wrap_env
from f1tenth_gym.envs import F110Env
from deep_drifting.config import EnvConfig, load_env_config

import numpy as np

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
    actions = []
    observations = []
    states = []
    while not done:
        action, _ = model.predict(obs)
        actions.append(action)
        obs, reward, done, truncated, info = eval_env.step(action)
        observations.append(obs)
        print(reward)
        frame = eval_env.render()

    np.save("actions.npy", np.asarray(actions))
    np.save("observations.npy", np.asarray(observations))
    np.save("positions.npy", np.asarray(eval_env.positions))
