#!/usr/bin/env python3

from typing import Tuple, List
import tensorrt as trt
from dataclasses import dataclass, field
import os
import rclpy
from pathlib import Path
from rclpy.node import Node
from scipy.spatial.transform import Rotation

import numpy as np
from drifting_interfaces.msg import StateEstimatesStamped
from ackermann_msgs.msg import AckermannDriveStamped
from utilities.params import register_config
from utilities import se3
from utilities.vis import vis_points
from state_observation import common
from state_observation.convert_trt import get_engine

ROOT = str(Path(os.getcwd()))

@dataclass
class DriftingConfig():
    """
    Dataclass containing the configuration parameters for the controller node
    """
    drive_topic: str = '/drive'
    state_obs_topic: str = '/ego_racecar/states'

    local_frame: str = 'laser'
    global_frame: str = 'map'

    v_max: float = 10.0
    max_dist_from_path: float = 5.0
    skip: int = 3

    onnx_model_path: str = os.path.join(ROOT, 'models', 'drifting.onnx')
    engine_path: str = os.path.join(ROOT, 'models', 'drifting.engine')
    waypoints_path: str = os.path.join(ROOT, 'tracks', 'waypoints.csv')

    action_steering_range: List[float] = field(default_factory=lambda: [-0.4189, 0.4189])
    action_velocity_range: List[float] = field(default_factory=lambda: [-1.0, 4.0]) # span 5

    visualize: bool = True


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

class ControllerNode(Node):
    def __init__(self) -> None:
        """
        Initialize the controller node

        1. Load the configuration
        2. Create the publisher for the drive commands
        3. Create the subscriber for the state observations
        4. Load the TensorRT model
        """
        super().__init__('controller_node')

        self.config = register_config(self, DriftingConfig())
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.config.drive_topic, 10)
        self.state_sub = self.create_subscription(StateEstimatesStamped, self.config.state_obs_topic, self.state_callback, 10)
        self.waypoints = self.load_waypoints()


        # TensorRT things
        self.model_engine, self.model_context = self.load_model()
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.model_engine)

        self.model_action_bounds = np.array([
            [-1.0, 1.0],
            [-1.0, 1.0]
        ])

        self.control_action_bounds = np.array([
            self.config.action_steering_range,
            self.config.action_velocity_range
        ])
    
    def load_waypoints(self) -> np.ndarray:
        """
        Load the waypoints from a file

        :return: np.ndarray of shape (n, 2) containing the waypoints
        """
        waypoints = np.loadtxt(self.config.waypoints_path, delimiter=',')[:, :2]
        if self.config.visualize:
            vis_points(self, "all_waypoints", waypoints, self.config.global_frame)
        return waypoints

    
    def load_model(self) -> Tuple[trt.ICudaEngine, trt.IExecutionContext]:
        """
        Load the TensorRT model

        :return: TensorRT model engine and execution context
        """
        engine = get_engine(self.config.onnx_model_path, self.config.engine_path, fp16=True)
        context = engine.create_execution_context()
        return engine, context
    
    def signed_orthogonal_distance_to_line(
        self,
        l1: np.ndarray,
        l2: np.ndarray,
        p: np.ndarray
    ) -> float:
        """
        Compute the signed orthogonal distance from a point to a line
        :param l1: np.ndarray of shape (2,) containing the first point of the line
        :param l2: np.ndarray of shape (2,) containing the second point of the line
        :param p: np.ndarray of shape (2,) containing the point
        :return: float containing the signed orthogonal distance
        """
        n = np.flip(l2 - l1) * [-1, 1]
        return np.dot(p - l1, n) / np.linalg.norm(n)
            
    
    def compute_local_waypoints(self, current_position: np.ndarray, yaw: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute the local waypoints given the current position and orientation

        :param current_position: np.ndarray of shape (2,) containing the current position
        :param yaw: float containing the current orientation
        :return: Tuple containing the current waypoint, the next waypoints and the signed distance to the path
        """
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

        signed_distance_to_path = self.signed_orthogonal_distance_to_line(waypoints_shifted[0], waypoints_shifted[1], np.zeros(2))
        current_waypoint = waypoints_shifted[0]
        next_waypoints = waypoints_shifted[1:1+5*self.config.skip:self.config.skip]
        return current_waypoint, next_waypoints, signed_distance_to_path 
    
    
    def state_callback(self, msg: StateEstimatesStamped) -> None:
        """
        Callback function for the state observations. 
        Takes the state observations and predicts the action to take

        :param msg: StateEstimatesStamped message
        """
        current_position = np.array([msg.x, msg.y])
        yaw = msg.yaw
        vel_x = msg.vel_x / self.config.v_max
        vel_y = msg.vel_y / self.config.v_max
        angular_velocity = msg.angular_vel
        slip_angle = msg.slip_angle

        _, next_waypoints, signed_distance_to_path = self.compute_local_waypoints(current_position, yaw)

        normed_abs_dist2path = np.abs(signed_distance_to_path) / self.config.max_dist_from_path

        flattened_next_waypoints = next_waypoints.flatten()

        state = np.array([vel_x, vel_y, angular_velocity, slip_angle, normed_abs_dist2path, *flattened_next_waypoints], np.float16)
        normalized_action = self.predict(state)
        self.publish_action(normalized_action)

    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        Predict the action to take given the state observations

        :param state: np.ndarray of shape (6,) containing the state observations
        :return: np.ndarray of shape (2,) containing the predicted action
        """
        self.inputs[0].host = state
        values, log_prob, normalized_action = common.do_inference_v2(self.model_context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)

        return normalized_action

    
    def publish_action(self, normalized_action: np.ndarray) -> None:
        """
        Publish the action to take to the drive topic

        :param action: np.ndarray of shape (2,) containing the action to take
        """
        print(normalized_action)
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        action = interpolate(normalized_action, self.model_action_bounds, self.control_action_bounds)
        msg.drive.steering_angle = action[0]
        msg.drive.speed = action[1]
        self.drive_pub.publish(msg)


def main(args=None) -> None:
    """
    Main function to initialize the controller node
    """
    rclpy.init(args=args)
    controller_node = ControllerNode()
    print("Controller Node Initialized")
    rclpy.spin(controller_node)

    controller_node.destroy_node()
    rclpy.shutdown()
    

if __name__ == '__main__':
    main()
