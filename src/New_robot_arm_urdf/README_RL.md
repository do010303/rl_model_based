# Robot Arm RL Training Setup

This package provides a complete setup for training a 4-DOF robot arm using reinforcement learning in Gazebo simulation with ROS Noetic.

## Prerequisites

- Ubuntu 20.04 LTS
- ROS Noetic (full desktop install)
- Python 3.8+
- Gazebo 11

## Installation

### 1. Install ROS Dependencies

```bash
sudo apt update
sudo apt install ros-noetic-gazebo-ros-pkgs ros-noetic-gazebo-ros-control
sudo apt install ros-noetic-controller-manager ros-noetic-effort-controllers
sudo apt install ros-noetic-joint-state-controller ros-noetic-robot-state-publisher
sudo apt install ros-noetic-xacro ros-noetic-tf2-tools
```

### 2. Install Python Dependencies

```bash
cd /home/ducanh/rl_model_based/New_robot_arm_urdf
pip install -r requirements.txt
```

### 3. Build the Package

```bash
cd /home/ducanh/rl_model_based
catkin_make
source devel/setup.bash
```

## Package Structure

```
New_robot_arm_urdf/
├── urdf/
│   ├── New.xacro          # Main robot URDF with transmissions
│   ├── New.gazebo         # Gazebo-specific properties
│   └── materials.xacro    # Material definitions
├── launch/
│   ├── gazebo.launch      # Basic Gazebo simulation
│   ├── controller.launch  # Controllers only
│   ├── rl_training.launch # RL training mode (headless)
│   └── controller.yaml    # Controller parameters
├── scripts/
│   ├── robot_arm_env.py   # RL environment wrapper
│   ├── train_robot_arm.py # Training script
│   └── target_marker.py   # Target visualization
├── meshes/                # STL mesh files
└── requirements.txt       # Python dependencies
```

## Usage

### 1. Basic Simulation (with GUI)

```bash
# Terminal 1: Launch Gazebo with robot
roslaunch New_description gazebo.launch

# Terminal 2: Start controllers
roslaunch New_description controller.launch
```

### 2. RL Training (headless, faster)

```bash
# All-in-one training command
cd /home/ducanh/rl_model_based/New_robot_arm_urdf/scripts
python train_robot_arm.py --algorithm SAC --timesteps 100000

# Or with GUI for debugging
python train_robot_arm.py --algorithm SAC --timesteps 100000 --gui
```

### 3. Test Environment Only

```bash
cd /home/ducanh/rl_model_based/New_robot_arm_urdf/scripts
python robot_arm_env.py
```

## Training Options

### Available Algorithms
- **SAC** (Soft Actor-Critic) - Good for continuous control
- **PPO** (Proximal Policy Optimization) - Stable and reliable
- **TD3** (Twin Delayed DDPG) - Good performance on robotics tasks
- **Custom** - Integrate with your own RL implementation

### Command Line Options

```bash
python train_robot_arm.py --help

Options:
  --algorithm {SAC,PPO,TD3,custom}  RL algorithm to use
  --timesteps INT                   Total training timesteps  
  --mode {train,test}               Training or testing mode
  --model-path PATH                 Path to trained model for testing
  --gui                             Show Gazebo GUI (slower training)
```

## Environment Details

- **Observation Space**: 14 dimensions
  - 4 joint positions
  - 4 joint velocities  
  - 3 target position (x, y, z)
  - 3 end-effector position (x, y, z)

- **Action Space**: 4 dimensions (joint position commands)
- **Reward Function**: 
  - Distance to target (negative)
  - Success bonus (+100 when target reached)
  - Joint velocity penalty (encourages smooth motion)
  - Joint limit penalty

## Integration with Parent RL Project

To integrate with your parent RL project in `/home/ducanh/rl_model_based/`:

1. **Import the environment**:
```python
sys.path.append('/home/ducanh/rl_model_based/New_robot_arm_urdf/scripts')
from robot_arm_env import RobotArmEnv

env = RobotArmEnv()
```

2. **Use with your RL algorithms**:
```python
# Your custom training loop
for episode in range(num_episodes):
    obs = env.reset()
    while not done:
        action = your_policy.predict(obs)
        obs, reward, done, info = env.step(action)
        your_policy.update(obs, action, reward)
```

## Troubleshooting

### Common Issues

1. **"Could not find package 'New_description'"**
   - Make sure you've run `catkin_make` and sourced the workspace
   - Check that `ROS_PACKAGE_PATH` includes your workspace

2. **Joint controllers not starting**
   - Verify Gazebo ros_control plugin is loaded
   - Check controller.yaml namespace matches launch file

3. **Python import errors** 
   - Install missing dependencies: `pip install -r requirements.txt`
   - Check Python path includes the scripts directory

4. **Gazebo crashes or slow performance**
   - Use headless mode: `--gui` flag off
   - Reduce physics update rate in Gazebo settings
   - Close other applications to free memory

### Useful Commands

```bash
# Check running nodes
rosnode list

# Monitor joint states
rostopic echo /New/joint_states

# Check controller status  
rosservice call /New/controller_manager/list_controllers

# View in RViz
rosrun rviz rviz -d launch/urdf.rviz
```

## Next Steps

1. **Tune hyperparameters** in `controller.yaml` and training script
2. **Modify reward function** in `robot_arm_env.py` for your specific task
3. **Add sensors** (cameras, force sensors) to the URDF
4. **Implement advanced RL algorithms** or integrate with existing frameworks
5. **Scale up training** with multiple parallel environments

## Files Modified from Original Export

- `urdf/New.xacro`: Added transmission elements
- `launch/controller.yaml`: Fixed namespace 
- `package.xml`: Added RL dependencies
- `launch/rl_training.launch`: New RL-specific launch file
- `scripts/`: All new RL training files