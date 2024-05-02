import gymnasium as gym
from stable_baselines3 import PPO
from pathlib import Path

from deep_drifting.environments.drifting import DeepDriftingEnv
from f1tenth_gym.envs import F110Env

from argparse import ArgumentParser


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
            "reset_config": {
                "type": "cl_grid_static"
            },
            "map": "Hockenheim",
         },
         render_mode="human"
    )
    env = DeepDriftingEnv(env)
    return env

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)

    args = parser.parse_args()

    eval_env = wrap_env()
    model_path = Path(args.model).absolute()
    model = PPO.load(model_path)

    obs, info = eval_env.reset()
    eval_env.render()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = eval_env.step(action)
        frame = eval_env.render()
