import gymnasium as gym

from f1tenth_gym.envs import F110Env

class DeepDriftingEnv(gym.Wrapper):
    env : F110Env
    
    def __init__(self, env: F110Env):
        super().__init__(env)

        self.env

    def step(self, action):
        obs, _, done, truncated, info = super().step(action)

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
    print(env.action_space)
    print(env.observation_space)
    # env.reset()