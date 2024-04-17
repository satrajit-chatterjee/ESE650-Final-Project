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
from utilities.params import register_config

ROOT = str(Path(os.getcwd()))

@dataclass
class StateObservationConfig():
    odom_topic: str = '/ego_racecar/odom'
    state_obs_topic: str = '/ego_racecar/state_observations'

class StateObservation(Node):
    def __init__(self):
        super().__init__('state_publisher_node')

        self.config = register_config(self, StateObservationConfig())

        self.odom_sub = self.create_subscription(Odometry, self.config.odom_topic, self.pose_callback, 10)
        self.state_pub = self.create_publisher(PointStamped, self.config.state_obs_topic, 10)

        self.prev_time = self.get_clock().now()

        self.current_pose = [0, 0]

    def pose_callback(self, msg):
        new_x = msg.pose.pose.position.x
        new_y = msg.pose.pose.position.y

        velocity_actual = msg.twist.twist.linear.x

        # TODO: (satrajic) Implement filtering for odometry data
        covariances = msg.pose.covariance

        orientation = msg.pose.pose.orientation
        _, _, yaw = tf.euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])

        current_time = self.get_clock().now()
        dt = (current_time - self.prev_time).nanoseconds / 1e9

        displacement = np.sqrt((new_x - self.current_pose[0])**2 + (new_y - self.current_pose[1])**2)
        velocity = displacement / dt
        print (f'Velocity: {velocity}, displacement: {displacement}, dt: {dt}')        
        # velocity along x-axis
        x_vel = velocity * np.cos(yaw)

        # velocity along y-axis
        y_vel = velocity * np.sin(yaw)
        
        slip_angle = np.arctan2(y_vel, x_vel)

        # update current pose
        self.current_pose = [new_x, new_y]
        self.prev_time = current_time

        # print velocity along x-axis, y-axis and slip angle
        # print(f'x_vel: {x_vel}, y_vel: {y_vel}, slip_angle: {slip_angle}')
    
        # write velocity, x_vel, y_vel, slip angle, and actual velocity to csv file
        with open(ROOT + '/src/state_observation/state_observation.csv', 'a') as f:
            f.write(f'{velocity},{x_vel},{y_vel},{slip_angle},{velocity_actual}\n')


def main(args=None):
    rclpy.init(args=args)
    state_publisher_node = StateObservation()
    print("State Observer Initialized")
    rclpy.spin(state_publisher_node)

    state_publisher_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
