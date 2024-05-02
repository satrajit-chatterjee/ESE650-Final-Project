from dataclasses import asdict
from typing import Optional

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from deep_drifting import se3
from deep_drifting.config import EnvConfig
from f1tenth_gym.envs import F110Env
from f1tenth_gym.envs.rendering import EnvRenderer

def signed_orthogonal_distance_to_line(
    l1: np.ndarray,
    l2: np.ndarray,
    p: np.ndarray
) -> float:
    n = np.flip(l2 - l1) * [-1, 1]
    return np.dot(p - l1, n) / np.linalg.norm(n)

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
    
    def __init__(
        self,
        env: F110Env, 
        l_path: float = 0.3, l_drift: float = 0.7,
        l_vx: float = 0.0, l_vy: float = 0.0,
        skip: int = 3,
        max_dist_from_path: float = 5.0,
        **kwargs
    ):
        super().__init__(env)

        self.l_path = l_path
        self.l_drift = l_drift
        self.l_vx = l_vx
        self.l_vy = l_vy
        self.skip = skip
        self.max_dist_from_path = max_dist_from_path

        if env.render_mode is not None:
            env.unwrapped.add_render_callback(env.unwrapped.track.centerline.render_waypoints)
            env.unwrapped.add_render_callback(self.render_distance_to_path)
            env.unwrapped.add_render_callback(self.render_next_waypoints)
            env.unwrapped.add_render_callback(self.render_velocities)
            env.unwrapped.add_render_callback(self.render_positions)

        self.action_space = gym.spaces.Box(
            low = -1.0, high = 1.0,
            shape = env.action_space.shape[1:]
        )

        large_num = 1e30

        # vx, vy, yaw_rate, beta, path_distance, wx1, wy1, ... wx5, wy5
        obs_low = np.full(15, -large_num)
        obs_high = np.full(15, large_num)
        obs_low[[0, 1, 3, 4]] = 0.0
        obs_high[[0, 1]] = 1.0
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high)

        xs = self.env.unwrapped.track.centerline.xs
        ys = self.env.unwrapped.track.centerline.ys
        self.waypoints = np.column_stack([xs, ys])
        
        self.T = np.eye(4)
        self.signed_distance_to_path: float = 0
        self.current_waypoint: int = 0
        self.next_waypoints: np.ndarray = np.arange(0, 5 * self.skip, self.skip)

        self.positions = [np.zeros(2)]

    def step(self, normalized_action: np.ndarray):

        action = interpolate(
            normalized_action,
            np.column_stack([self.action_space.low, self.action_space.high]),
            np.column_stack([self.env.action_space.low.squeeze(), self.env.action_space.high.squeeze()]),
        )[None]

        obs, _, done, truncated, info = self.env.step(action)

        # 1 agent
        obs = obs["agent_0"]

        current_position = np.asarray([obs["pose_x"], obs["pose_y"]])
        yaw = obs["pose_theta"]
        v = obs["linear_vel_x"]
        omega = obs["ang_vel_z"]
        beta = obs["beta"]

        vx = v * np.cos(beta)
        vy = v * np.sin(beta)

        self.vx = vx
        self.vy = vy

        self.next_waypoints = np.mod(np.arange(self.current_waypoint, self.current_waypoint + 5 * self.skip, self.skip, dtype=int), self.waypoints.shape[0])
        relative_waypoints = self.compute_relative_waypoints(current_position, yaw)
        waypoint_distances = np.linalg.norm(relative_waypoints[[self.current_waypoint - 1, self.next_waypoints[0]]], axis=1)
        # print(waypoint_distances)
        self.signed_distance_to_path = signed_orthogonal_distance_to_line(
            relative_waypoints[self.current_waypoint - 1],
            relative_waypoints[self.current_waypoint],
            np.zeros(2)
        )
        normalized_obs = np.array([
            vx / self.env.unwrapped.params["v_max"],
            vy / self.env.unwrapped.params["v_max"],
            omega,
            beta,
            np.abs(self.signed_distance_to_path) / self.max_dist_from_path
        ])
        normalized_obs = np.concatenate([normalized_obs, relative_waypoints[self.next_waypoints].flatten()])

        if (np.abs(self.signed_distance_to_path) > self.max_dist_from_path):
            done = True

        reward = 0
        if (waypoint_distances[1] < waypoint_distances[0]):
            r_path = np.exp(-3 * np.square(normalized_obs[4]))
            r_drift = 1 / (1 + np.power(np.abs((np.abs(beta) - 0.87) / 0.26), 3))
            r_vx = np.maximum(normalized_obs[0], 0)
            r_vy = np.maximum(normalized_obs[1], 0)

            reward += self.l_path * r_path
            reward += self.l_drift * r_drift
            reward += self.l_vx * r_vx
            reward += self.l_vy * r_vy

            self.current_waypoint = self.next_waypoints[1]

        return normalized_obs, reward, done, truncated, info


    def compute_relative_waypoints(self, current_position: Optional[np.ndarray] = None, yaw: Optional[float] = None):
        if current_position is not None and yaw is not None:
            R = Rotation.from_euler("Z", yaw).as_matrix()
            t = np.pad(current_position, (0, 1))
            self.positions.append(current_position)
            self.T = se3.from_rotation_translation(R, t)
        return se3.transform_points(se3.inverse(self.T), self.waypoints)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)


        self.T = np.eye(4)
        self.signed_distance_to_path: float = 0
        self.current_waypoint: int = 0
        self.next_waypoints: np.ndarray = np.arange(0, 5 * self.skip, self.skip)
        self.positions = []

        action = np.array([0, -0.6])
        obs, _, _, _, info = self.step(action)

        relative_waypoints = self.compute_relative_waypoints()
        waypoint_distances = np.linalg.norm(relative_waypoints, axis=1)
        self.current_waypoint = np.argmin(waypoint_distances)
        if relative_waypoints[self.current_waypoint][0] <= 0.0:
            self.current_waypoint = (self.current_waypoint + self.skip) % self.waypoints.shape[0]
        # print(f"RESET: {self.current_waypoint}")
        self.next_waypoints = np.mod(np.arange(self.current_waypoint, self.current_waypoint + 5 * self.skip, self.skip, dtype=int), self.waypoints.shape[0])
        return obs, info

    def render_velocities(self, renderer: EnvRenderer):
        current_position = self.T[:2, 3]
        vx = se3.transform_points(self.T, np.array([self.vx, 0, 0, 0]))[:2]
        vy = se3.transform_points(self.T, np.array([0, self.vy, 0, 0]))[:2]
        renderer.render_lines(np.vstack([current_position, current_position + vx]), color=(255, 0, 0))
        renderer.render_lines(np.vstack([current_position, current_position + vy]), color=(0, 100, 200))

    def render_distance_to_path(self, renderer: EnvRenderer):
        previous_waypoint = self.waypoints[self.current_waypoint - 1]
        current_waypoint = self.waypoints[self.current_waypoint]
        n = np.flip(current_waypoint - previous_waypoint) * [-1, 1]
        n /= np.linalg.norm(n)
        current_position = self.T[:2, 3]
        renderer.render_lines(np.vstack([current_position, current_position - n * self.signed_distance_to_path]), color=(0, 255, 0))

    def render_next_waypoints(self, renderer: EnvRenderer):
        renderer.render_lines(self.waypoints[self.next_waypoints], color=(255, 0, 0), size=2)

    def render_positions(self, renderer: EnvRenderer):
        if len(self.positions) == 1: return
        renderer.render_lines(np.asarray(self.positions), color=(200, 100, 0))

def wrap_env(env_config: EnvConfig, render_mode: Optional[str] = None):
    env : F110Env = gym.make(
        "f1tenth_gym:f1tenth-v0",
         config = {
            "num_agents": 1,
            "observation_config": {
                "type": "dynamic_state"
            },
            "params": {
                "mu": env_config.mu
            },
            "reset_config": {
                "type": "cl_grid_static"
            },
            "map": env_config.map
         },
         render_mode=render_mode
    )
    env = DeepDriftingEnv(env, **asdict(env_config))
    return env

if __name__ == "__main__":
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
    print(env)
    env = DeepDriftingEnv(env)
    print(env.unwrapped.config["params"])

    print(env.action_space)
    print(env.observation_space)

    obs, info = env.reset()
    env.render()
    done = False
    while not done:
        action = env.action_space.sample()
        # action = np.array([0.0, 0.6])
        obs, step_reward, done, truncated, info = env.step(action)
        # print(obs, step_reward)
        frame = env.render()