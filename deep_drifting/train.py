import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from f1tenth_gym.envs import F110Env

def wrap_env():
    env : F110Env = gym.make(
        "f1tenth_gym:f1tenth-v0",
         config = {
            "num_agents": 1,
            # "observation_config": {
            #     "type": "dynamic_state"
            # },
         },
    )

    return env

if __name__ == "__main__":
    # env : F110Env = gym.make(
    #     "f1tenth_gym:f1tenth-v0",
    #      config = {
    #         "num_agents": 1,
    #         "observation_config": {
    #             "type": "dynamic_state"
    #         }
    #      }
    # )

    # TODO(rahul): make params configurable
    vec_env = make_vec_env(wrap_env, n_envs=1, seed=42)
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        learning_rate=0.0003, # TODO(rahul): make a learning schedule
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1,
        device="cuda"
    )

    # eval_callback = EvalCallback(
    #     vec_env, best_model_save_path="./logs/",
    #     log_path="./logs/", eval_freq=5000,
    #     deterministic=True, render=True
    # )

    model.learn(total_timesteps=10000, progress_bar=True)