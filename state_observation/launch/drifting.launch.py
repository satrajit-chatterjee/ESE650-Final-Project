# Launch file that launches the state observation node and the controller node

from launch_ros.actions import Node
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='state_observation',
            executable='state_publisher_node',
            name='state_publisher',
            output='screen'
        ),
        Node(
            package='state_observation',
            executable='controller_node',
            name='controller',
            output='screen'
        )
    ])

