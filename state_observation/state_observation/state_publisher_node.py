#!/usr/bin/env python3

from dataclasses import dataclass
import os
import rclpy
from pathlib import Path
from rclpy.node import Node

import numpy as np
import tf_transformations as tf
from drifting_interfaces.msg import StateEstimatesStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from ackermann_msgs.msg import AckermannDriveStamped
from utilities.params import register_config
from utilities import se3

ROOT = str(Path(os.getcwd()))

@dataclass
class StateObservationConfig():
    odom_topic: str = '/ego_racecar/odom'
    lidar_topic: str = '/scan'
    drive_topic: str = '/drive'
    state_obs_topic: str = '/ego_racecar/states'
    erpm_topic: str = '/commands/motor/speed'

    local_frame: str = 'ego_racecar/base_link'
    global_frame: str = 'map'

    # Vehicle parameters
    wheelbase: float = 0.25  # meters

class StateObservation(Node):
    def __init__(self):
        super().__init__('state_publisher_node')

        self.config = register_config(self, StateObservationConfig())

        self.odom_sub = self.create_subscription(Odometry, self.config.odom_topic, self.pose_callback, 10)
        self.drive_sub = self.create_subscription(AckermannDriveStamped, self.config.drive_topic, self.drive_callback, 10)
        self.erpm_sub = self.create_subscription(Float64, self.config.erpm_topic, self.erpm_callback, 10)

        self.state_pub = self.create_publisher(StateEstimatesStamped, self.config.state_obs_topic, 10)

        self.erpm = 0.0
        self.steering_angle = 0.0

    def drive_callback(self, msg):
        self.steering_angle = msg.drive.steering_angle

    def erpm_callback(self, msg):
        self.erpm = msg.data


    def pose_callback(self, msg):
        # Get the pose of the car in the map frame
        new_x = msg.pose.pose.position.x
        new_y = msg.pose.pose.position.y

        # orientation of the car in the map frame
        yaw = tf.euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2]

        # longitudinal velocity in body frame
        v = msg.twist.twist.linear.x

        # Ref: https://link.springer.com/book/10.1007/978-1-4614-1433-9 (Chapter 2)
        slip_angle = np.arctan(0.5*np.tan(self.steering_angle))
        v_y_car = v * np.sin(slip_angle)
        v_x_car = v * np.cos(slip_angle)
        angular_velocity = v*np.cos(slip_angle)*np.tan(self.steering_angle) / self.config.wheelbase

        # Publish the state estimates
        state_msg = StateEstimatesStamped()
        state_msg.header.stamp = self.get_clock().now().to_msg()
        state_msg.header.frame_id = self.config.local_frame
        state_msg.x = new_x
        state_msg.y = new_y
        state_msg.vel_x = v_x_car
        state_msg.vel_y = v_y_car
        state_msg.yaw = yaw
        state_msg.slip_angle = slip_angle
        state_msg.angular_vel = angular_velocity
        state_msg.erpm = self.erpm
        self.state_pub.publish(state_msg)


def main(args=None):
    rclpy.init(args=args)
    state_publisher_node = StateObservation()
    print("State Observer Initialized")
    rclpy.spin(state_publisher_node)

    state_publisher_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
