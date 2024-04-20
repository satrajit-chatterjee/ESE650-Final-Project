# Visualization of Rosbags in Foxglove

Assuming, Foxglove is installed, because these rosbags have custom message types, we must first convert them to the `.mcap` data format. See: [https://docs.foxglove.dev/docs/connecting-to-data/frameworks/ros2/#local-data](https://docs.foxglove.dev/docs/connecting-to-data/frameworks/ros2/#local-data)

## Prerequisites

### Messages

Because we are using messages that come from the car, we need to have the on-car packages in our ROS 2 workspace. In your `sim_ws`, please execute:
```bash
cd src
git clone https://github.com/f1tenth/f1tenth_system.git
cd f1tenth_system
git submodule update --init --force --remote
cd ../..
rosdep install -r --from-paths src --ignore-src --rosdistro foxy
colcon build
```

The above should run without errors in order for the conversion to work.

Additionally, you will need to have the [`drifting_interfaces`](../../drifting_interfaces/package.xml) and [`state_observation`](../../state_observation/package.xml) packages built in your `sim_ws`. If this repo is already installed within your `sim_ws`, you can skip the following step.

If not, one way to do this is to execute the following from inside `sim_ws`:
```bash
cd src
ln -s <THIS_REPO_PATH>/drifting_interfaces
ln -s <THIS_REPO_PATH>/state_observation
```

Once this is done, we simply need to source our workspace environment by executing (NOTE: users with specific shells like zsh should modify the below commands appropriately.): This will load the needed paths for all message types to be found.
```bash
source /opt/ros/foxy/setup.sh
source <SIM_WS>/install/local_setup.sh
```

### MCAP

Finally, we will need to have access to [`mcap`](https://mcap.dev/). Installation instructions can be found here: [https://mcap.dev/guides/cli](https://mcap.dev/guides/cli).

##### Nix 

Users who have Nix installed on their systems can execute the following command to get `mcap` in a shell.
```bash
nix shell nixpkgs#mcap-cli
```

## Conversion

To convert a rosbag, assuming all of the prerequisites are satisfied, one can simply run:
```bash
mcap convert <ROS_BAG_NAME>.db3 <ROS_BAG_NAME>.mcap
```

## Visualization

We recommend using [Foxglove Studio](https://app.foxglove.dev/) for visualization of these rosbags. To do so, one can simply drag the generated `.mcap` file into the app.
