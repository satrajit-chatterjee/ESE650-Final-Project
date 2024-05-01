#!/usr/bin/env python3

from typing import Tuple
import tensorrt as trt
from dataclasses import dataclass
import os
import rclpy
from pathlib import Path
from rclpy.node import Node
import common
from scipy.spatial.transform import Rotation

import numpy as np
from drifting_interfaces.msg import StateEstimatesStamped
from ackermann_msgs.msg import AckermannDriveStamped
from utilities.params import register_config
from utilities import se3
from convert_trt import get_engine

ROOT = str(Path(os.getcwd()))

@dataclass
class DriftingConfig():
    """
    Dataclass containing the configuration parameters for the controller node
    """
    drive_topic: str = '/drive'
    state_obs_topic: str = '/ego_racecar/states'

    local_frame: str = 'ego_racecar/base_link'
    global_frame: str = 'map'

    v_max: float = 10.0
    max_dist_from_path: float = 5.0

    onnx_model_path: str = os.path.join(ROOT, 'models', 'drifting.onnx')
    engine_path: str = os.path.join(ROOT, 'models', 'drifting.engine')
    waypoints_path: str = os.path.join(ROOT, 'tracks', 'waypoints.csv')


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
        self.engine, self.context = self.load_model()
        self.waypoints = self.load_waypoints()

    
    def load_waypoints(self) -> np.ndarray:
        """
        Load the waypoints from a file

        :return: np.ndarray of shape (n, 2) containing the waypoints
        """
        return np.loadtxt(self.config.waypoints_path, delimiter=',')

    
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
        next_waypoints = waypoints_shifted[1:6]
        return current_waypoint, next_waypoints, signed_distance_to_path 
    
    
    def state_callback(self, msg: StateEstimatesStamped) -> None:
        """
        Callback function for the state observations. 
        Takes the state observations and predicts the action to take

        :param msg: StateEstimatesStamped message
        """
        x, y = msg.x, msg.y
        yaw = msg.yaw
        vel_x = msg.vel_x / self.config.v_max
        vel_y = msg.vel_y / self.config.v_max
        angular_velocity = msg.angular_velocity
        slip_angle = msg.slip_angle

        _, next_waypoints, signed_distance_to_path = self.compute_local_waypoints(np.array([x, y]), yaw)

        normed_abs_dist2path = np.abs(signed_distance_to_path) / self.config.max_dist_from_path

        flattened_next_waypoints = next_waypoints.flatten()

        state = np.array([vel_x, vel_y, angular_velocity, slip_angle, normed_abs_dist2path, *flattened_next_waypoints])
        assert state.shape == (15,)
        action = self.predict(state)
        self.publish_action(action)

    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        Predict the action to take given the state observations

        :param state: np.ndarray of shape (6,) containing the state observations
        :return: np.ndarray of shape (2,) containing the predicted action
        """
        inputs, outputs, bindings, stream = common.allocate_buffers(self.engine)
        inputs[0].host = state
        output = common.do_inference_v2(self.context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        return output

    
    def publish_action(self, action: np.ndarray) -> None:
        """
        Publish the action to take to the drive topic

        :param action: np.ndarray of shape (2,) containing the action to take
        """
        msg = AckermannDriveStamped()
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
