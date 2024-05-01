import gymnasium as gym
import numpy as np

from f1tenth_gym.envs import F110Env

from scipy.spatial.transform import Rotation
from deep_drifting import se3

def orthogonal_distance_to_line(
    l1: np.ndarray,
    l2: np.ndarray,
    p: np.ndarray
) -> float:
    n = np.flip(l2 - l1) * [-1, 1]
    return np.abs(np.dot(p - l1, n) / np.linalg.norm(n))

def interpolate(x: np.ndarray, input_range: np.ndarray, output_range: np.ndarray) -> np.ndarray:
    """Interpolates some shit

    Parameters
    ----------
    x : np.ndarray
        [N] array of values to convert ranges
    input_range : np.ndarray
        [N, 2] array where the first column represents the lower bound
        of the input range and the second column represents the upper bound
    output_range : np.ndarray
        [N, 2] array where the first column represents the lower bound
        of the output range and the second column represents the upper bound

    Returns
    -------
    np.ndarray
        [N] array of converted values
    """
    in_low, in_high = input_range.T
    out_low, out_high = output_range.T

    return (x - in_low) / (in_high - in_low) * (out_high - out_low) + out_low

class DeepDriftingEnv(gym.Wrapper):
    env : F110Env
    
    def __init__(self, env: F110Env):
        super().__init__(env)

        print(self.env.action_space.low)
        self.action_space = gym.spaces.Box(
            low = -1.0, high = 1.0,
            shape = env.action_space.shape[1:]
        )
        print(self.env.observation_space.sample())

        large_num = 1e30

        # in deep drifting they have x = left, y = up, z = forward
        # vy [0:1], vx [0:1]
        # omega [?], slip_angle[?]
        # 5 waypoints [0:1] (wy1, wx1, ...)
        # engine_rpm [0:1]
        # distance to path [0:1]
        self.observation_space = gym.spaces.Box(
            low = np.array([0.0, 0.0, -large_num, large_num]),
            high = np.array([1.0, 1.0, large_num, large_num])
        )

        xs = self.env.unwrapped.track.raceline.xs
        ys = self.env.unwrapped.track.raceline.ys
        self.waypoints = np.column_stack([xs, ys])

    def step(self, normalized_action: np.ndarray):

        action = interpolate(
            normalized_action,
            np.column_stack([self.action_space.low, self.action_space.high]),
            np.column_stack([self.env.action_space.low.squeeze(), self.env.action_space.high.squeeze()]),
        )

        obs, _, done, truncated, info = self.env.step(action)

        # 1 agent
        obs = obs["agent_0"]

        x = obs["pose_x"]
        y = obs["pose_y"]
        yaw = obs["pose_theta"]
        vx = obs["linear_vel_x"]
        vy = obs["linear_vel_y"]
        omega = obs["ang_vel_z"]
        beta = obs["beta"]

        x, y, 
        print(obs)

        R = Rotation.from_euler("Z", yaw).as_matrix()
        t = np.array([x, y, 0])
        T = se3.from_rotation_translation(R, t)

        relative_waypoints = se3.transform_points(se3.inverse(T), self.waypoints)
        waypoint_distances = np.linalg.norm(relative_waypoints, axis=1)
        self.closest_waypoint_index = np.argmin(waypoint_distances)
        next_waypoints = self.waypoints[self.closest_waypoint_index:self.closest_waypoint_index+5]


        normalized_obs = np.array([
            vx, vy, omega, beta
        ])

        # TODO(rahul): normalize observation

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
    env = DeepDriftingEnv(env)

    print(env.action_space)
    print(env.observation_space)

    action = env.action_space.sample()
    env.reset()
    # print(orthogonal_distance_to_line(
    #     np.array([0.26, 1.31]),
    #     np.array([2, 3]),
    #     np.array([2.94, 1.08])
    # ))