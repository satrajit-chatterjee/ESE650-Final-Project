#!/usr/bin/env python3

from dataclasses import dataclass
import os
import rclpy
from pathlib import Path
from rclpy.node import Node

import numpy as np
import tf_transformations as tf
from geometry_msgs.msg import PointStamped
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from utilities.params import register_config
from utilities import se3

ROOT = str(Path(os.getcwd()))

@dataclass
class StateObservationConfig():
    odom_topic: str = '/ego_racecar/odom'
    lidar_topic: str = '/scan'
    drive_topic: str = '/drive'
    state_obs_topic: str = '/ego_racecar/states'

class StateObservation(Node):
    def __init__(self):
        super().__init__('state_publisher_node')

        self.config = register_config(self, StateObservationConfig())

        self.odom_sub = self.create_subscription(Odometry, self.config.odom_topic, self.pose_callback, 10)
        self.lidar_sub = self.create_subscription(LaserScan, self.config.lidar_topic, self.lidar_callback, 10)
        self.drive_sub = self.create_subscription(AckermannDriveStamped, self.config.drive_topic, self.drive_callback, 10)

        self.state_pub = self.create_publisher(PointStamped, self.config.state_obs_topic, 10)

        self.prev_time = self.get_clock().now()

        self.current_pose = [0, 0]

        self.desired_heading = 0
        self.commanded_speed = 0
        self.steer_angle = 0
        self.past_steer_angle = 0
        self.angular_velocity = 0

    def lidar_callback(self, msg):
        # Get the angle of the furthest point from the lidar scan
        max_index = np.argmax(msg.ranges)
        max_angle = msg.angle_min + max_index * msg.angle_increment

        # Get heading of the robot
        self.desired_heading = max_angle
        
    def drive_callback(self, msg):
        # Get the steering angle from the ackermann drive message
        self.past_steer_angle = self.steer_angle
        self.steer_angle = msg.drive.steering_angle
        current_time = self.get_clock().now()
        dt = (current_time - self.prev_time).nanoseconds / 1e9
        self.angular_velocity = (self.steer_angle - self.past_steer_angle) / dt
        self.prev_time = current_time
        
        self.commanded_speed = msg.drive.speed

    def pose_callback(self, msg):
        new_x = msg.pose.pose.position.x
        new_y = msg.pose.pose.position.y

        velocity = msg.twist.twist.linear.x

        # TODO: (satrajic) Implement filtering for odometry data
        covariances = msg.pose.covariance

        base_T_map = se3.inverse(se3.from_msg(msg.pose.pose))
        R_base_map = base_T_map[:3, :3]
        _, _, base_orientation = tf.euler_from_matrix(R_base_map)

        # velocity along x-axis
        x_vel = velocity * np.cos(self.desired_heading)

        # velocity along y-axis
        y_vel = velocity * np.sin(self.desired_heading)
        
        slip_angle = np.arctan2(y_vel, x_vel)  # same as desired heading, no need to calc

        # update current pose
        self.current_pose = [new_x, new_y]

        # print velocity along x-axis, y-axis and slip angle
        print(f'Desired heading: {np.degrees(self.desired_heading):.2f}, pred_velocity: {velocity:.2f}, x_vel: {x_vel:.2f}, y_vel: {y_vel:.2f}, slip_angle: {np.degrees(slip_angle):.2f}, yaw_rate: {self.angular_velocity:.2f}')
    
        # write velocity, x_vel, y_vel, slip angle, and actual velocity to csv file
        with open(ROOT + '/src/state_observation/state_observation.csv', 'a') as f:
            f.write(f'{velocity},{x_vel},{y_vel},{slip_angle},{self.commanded_speed}\n')


def main(args=None):
    rclpy.init(args=args)
    state_publisher_node = StateObservation()
    print("State Observer Initialized")
    rclpy.spin(state_publisher_node)

    state_publisher_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
