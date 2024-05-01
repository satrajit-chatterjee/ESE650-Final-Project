import gymnasium as gym
import numpy as np

from f1tenth_gym.envs import F110Env
from f1tenth_gym.envs.rendering import EnvRenderer

from scipy.spatial.transform import Rotation
from deep_drifting import se3

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
    
    def __init__(self, env: F110Env):
        super().__init__(env)

        if env.render_mode is not None:
            env.unwrapped.add_render_callback(env.unwrapped.track.centerline.render_waypoints)
            env.unwrapped.add_render_callback(self.render_distance_to_path)
            env.unwrapped.add_render_callback(self.render_local_waypoints)
            env.unwrapped.add_render_callback(self.render_velocities)

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

        self.max_dist_from_path = 5.0
        
        self.T = np.eye(4)
        self.distance_to_path = 0
        self.current_waypoint = None
        self.next_waypoints = None

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

        self.compute_local_waypoints(current_position, yaw)
        normalized_obs = np.array([
            vx / self.env.unwrapped.params["v_max"],
            vy / self.env.unwrapped.params["v_max"],
            omega,
            beta,
            np.abs(self.signed_distance_to_path) / self.max_dist_from_path
        ])
        normalized_obs = np.concatenate([normalized_obs, self.next_waypoints.flatten()])

        if (np.abs(self.signed_distance_to_path) > self.max_dist_from_path):
            done = True

        r_path = np.exp(-3 * np.square(normalized_obs[4]))
        r_drift = 1 / (1 + np.power(np.abs((np.abs(beta) - 0.87) / 0.26), 3))

        reward = 0.05 * r_path + 0.95 * r_drift

        return normalized_obs, reward, done, truncated, info

    def compute_local_waypoints(self, current_position: np.ndarray, yaw: float):
        R = Rotation.from_euler("Z", yaw).as_matrix()
        t = np.pad(current_position, (0, 1))
        T = se3.from_rotation_translation(R, t)

        self.T = T

        relative_waypoints = se3.transform_points(se3.inverse(T), self.waypoints)
        waypoint_distances = np.linalg.norm(relative_waypoints, axis=1)
        closest_waypoint_index = np.argmin(waypoint_distances)
        if relative_waypoints[closest_waypoint_index][0] < 0:
            closest_waypoint_index = (closest_waypoint_index + 1) % self.waypoints.shape[0]
        waypoints_shifted = np.roll(relative_waypoints, -closest_waypoint_index + 1, axis=0)

        self.signed_distance_to_path = signed_orthogonal_distance_to_line(waypoints_shifted[0], waypoints_shifted[1], np.zeros(2))
        self.current_waypoint = waypoints_shifted[0]
        self.next_waypoints = waypoints_shifted[1:6]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        action = np.array([0, -0.6])
        obs, _, _, _, info = self.step(action)
        return obs, info

    def render_velocities(self, renderer: EnvRenderer):
        current_position = self.T[:2, 3]
        vx = se3.transform_points(self.T, np.array([self.vx, 0, 0, 0]))[:2]
        vy = se3.transform_points(self.T, np.array([0, self.vy, 0, 0]))[:2]
        renderer.render_lines(np.vstack([current_position, current_position + vx]), color=(255, 0, 0))
        renderer.render_lines(np.vstack([current_position, current_position + vy]), color=(0, 255, 0))

    def render_distance_to_path(self, renderer: EnvRenderer):
        n = np.flip(se3.transform_points(self.T, np.pad(self.next_waypoints[0] - self.current_waypoint, (0, 2)))[:2]) * [-1, 1]
        n /= np.linalg.norm(n)
        current_position = self.T[:2, 3]
        renderer.render_lines(np.vstack([current_position, current_position - n * self.signed_distance_to_path]), color=(0, 255, 0))

    def render_local_waypoints(self, renderer: EnvRenderer):
        abs_waypoints = se3.transform_points(self.T, self.next_waypoints)
        renderer.render_closed_lines(abs_waypoints, color=(255, 0, 0), size=2)


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
            "map": "Spielberg",
         },
         render_mode="human"
    )
    print(env)
    env = DeepDriftingEnv(env)
    print(env.unwrapped.config["params"])

    print(env.action_space)
    print(env.observation_space)

    obs, info = env.reset()
    print(obs)
    env.render()
    done = False
    while not done:
        action = env.action_space.sample()
        # action = np.array([0.0, 0.0])
        obs, step_reward, done, truncated, info = env.step(action)
        print(obs, step_reward)
        frame = env.render()