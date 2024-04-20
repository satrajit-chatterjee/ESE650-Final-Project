import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env

from f1tenth_gym.envs import F110Env

if __name__ == "__main__":
    env : F110Env = gym.make(
        "f1tenth_gym:f1tenth-v0",
         config = {
            "num_agents": 1,
            "observation_config": {
                "type": "dynamic_state"
            }
         }
    )

