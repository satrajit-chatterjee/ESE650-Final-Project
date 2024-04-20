import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from f1tenth_gym.envs import F110Env

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
    vec_env = make_vec_env(F110Env, n_envs=1, seed=42)
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        learning_rate=0.0003, # TODO(rahul): make a learning schedule
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1,
        device="cuda"
    )

    model.learn(total_timesteps=10000)