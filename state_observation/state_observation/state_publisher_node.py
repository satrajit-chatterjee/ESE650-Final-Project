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

class StateObservation(Node):
    def __init__(self):
        super().__init__('state_publisher_node')

        self.config = register_config(self, StateObservationConfig())

        self.odom_sub = self.create_subscription(Odometry, self.config.odom_topic, self.pose_callback, 10)
        self.erpm_sub = self.create_subscription(Float64, self.config.erpm_topic, self.erpm_callback, 10)

        self.state_pub = self.create_publisher(StateEstimatesStamped, self.config.state_obs_topic, 10)

        self.erpm = 0.0

    def erpm_callback(self, msg):
        self.erpm = msg.data


    def pose_callback(self, msg):
        new_x = msg.pose.pose.position.x
        new_y = msg.pose.pose.position.y
        yaw = tf.euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2]

        v_map = msg.twist.twist.linear.x

        v_x_map, v_y_map = v_map * np.cos(yaw), v_map * np.sin(yaw)

        # Find velocity in local frame
        R_map2car = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        v_x_car, v_y_car = np.dot(R_map2car, np.array([v_x_map, v_y_map]))
        
        slip_angle = np.arctan2(v_y_car, v_x_car)
        angular_velocity = msg.twist.twist.angular.z

        # Publish the state estimates
        state_msg = StateEstimatesStamped()
        state_msg.header.stamp = self.get_clock().now().to_msg()
        state_msg.header.frame_id = 'map'
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
