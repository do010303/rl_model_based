# Visual MBPO Training System for ROS Noetic

## Overview

This system provides a complete visual reinforcement learning environment for a 4-DOF robot arm using Gazebo simulation, integrated with Model-Based Policy Optimization (MBPO) training.

## Features

- **ROS Noetic Compatibility**: Full integration with ROS Noetic and Gazebo
- **4-DOF Robot Arm**: Adapted from original 6-DOF to 4-DOF configuration  
- **Visual Workspace**: Real-time Gazebo simulation with workspace boundaries
- **Target Sphere**: Dynamic red target sphere spawning within cylindrical workspace
- **MBPO Training**: Model-based reinforcement learning with DDPG policy
- **TF Integration**: Real-time end-effector tracking using TF transforms
- **Mathematical Precision**: Cylindrical workspace with polar coordinate sampling

## Architecture

### Core Components

1. **GazeboRLEnvironment** (`gazebo_rl_environment.py`)
   - ROS Noetic adaptation of robotic_arm_environment
   - TF-based end-effector tracking with forward kinematics fallback
   - Synchronized message handling for joint states and target positions
   - Cylindrical workspace management (0.18m radius)

2. **VisualMBPOTrainer** (`train_gazebo_mbpo_visual.py`)  
   - MBPO training integration with visual environment
   - Connects existing MBPO components with Gazebo simulation
   - Episode management and statistics tracking

3. **Target Management**
   - Dynamic target sphere spawning using Gazebo services
   - SDF model definition for realistic physics
   - Mathematical position sampling within workspace bounds

### Workspace Configuration

- **Shape**: Cylindrical workspace (not spherical)
- **Radius**: 0.18m (reachable robot workspace)
- **Height**: 0.02m to 0.25m above table
- **Visualization**: Blue cylinder boundary in Gazebo
- **Target Sampling**: Polar coordinates for uniform distribution

## Installation & Setup

### Prerequisites
```bash
# ROS Noetic with Gazebo
sudo apt install ros-noetic-desktop-full
sudo apt install ros-noetic-gazebo-ros-pkgs
sudo apt install ros-noetic-gazebo-ros-control
sudo apt install ros-noetic-joint-state-controller
sudo apt install ros-noetic-position-controllers
```

### Python Dependencies
```bash
pip3 install numpy matplotlib torch tensorboard
# Additional RL dependencies from existing project
```

## Usage

### 1. Launch Visual Training Environment
```bash
# Terminal 1: Launch Gazebo simulation with robot and workspace
roslaunch new_robot_arm_urdf visual_mbpo_training.launch

# Wait for Gazebo to fully load, then proceed
```

### 2. Test Environment (Optional)
```bash
# Terminal 2: Test environment functionality
python3 test_gazebo_rl_environment.py
```

### 3. Start MBPO Training
```bash
# Terminal 2: Start visual MBPO training
python3 train_gazebo_mbpo_visual.py
```

## File Structure

```
/home/ducanh/rl_model_based/
├── gazebo_rl_environment.py          # Main RL environment (ROS Noetic)
├── train_gazebo_mbpo_visual.py       # MBPO training integration  
├── test_gazebo_rl_environment.py     # Environment testing script
├── models/
│   └── target_sphere/                # Gazebo target sphere model
│       ├── model.sdf                 # SDF model definition
│       └── model.config              # Model configuration
└── New_robot_arm_urdf/
    ├── launch/
    │   └── visual_mbpo_training.launch # Complete system launch
    ├── worlds/
    │   └── training_world.world       # Gazebo world with workspace
    └── config/
        ├── joint_position_controllers.yaml
        └── joint_trajectory_controller.yaml
```

## Key Differences from Original

### ROS2 → ROS Noetic Adaptations

1. **Message Synchronization**: 
   - ROS2: `message_filters.ApproximateTimeSynchronizer`
   - ROS Noetic: Same API, different import structure

2. **TF Integration**:
   - ROS2: `tf2_ros.Buffer` and `TransformListener`  
   - ROS Noetic: Compatible API with `tf2_ros`

3. **Action Clients**:
   - ROS2: `rclpy.action.ActionClient`
   - ROS Noetic: `actionlib.SimpleActionClient`

4. **Service Calls**:
   - ROS2: Async service calls with futures
   - ROS Noetic: Synchronous `rospy.ServiceProxy`

### Environment Features

- **Cylindrical Workspace**: Mathematical sampling within 0.18m radius cylinder
- **4-DOF Adaptation**: Reduced from 6-DOF with proper joint limits
- **State Vector**: 15-dimensional [joints(4) + velocities(4) + end_effector(3) + target(3) + distance(1)]
- **Action Space**: 4-dimensional continuous control

## Troubleshooting

### Common Issues

1. **TF Lookup Failures**:
   - Ensure robot model is properly loaded in Gazebo
   - Check TF tree with: `rosrun tf view_frames`
   - Fallback forward kinematics handles temporary failures

2. **Target Spawning Errors**:
   - Verify Gazebo model path: `echo $GAZEBO_MODEL_PATH`
   - Check target_sphere model files exist
   - Ensure Gazebo services are available

3. **Controller Issues**:
   - Verify controllers are loaded: `rosservice call /controller_manager/list_controllers`
   - Check joint limits in URDF match action bounds
   - Ensure trajectory controller is properly configured

### Debug Commands
```bash
# Check ROS topics
rostopic list

# Monitor joint states
rostopic echo /joint_states

# Check TF transforms  
rosrun tf tf_echo world link_4_1

# List Gazebo models
rosservice call /gazebo/get_world_properties

# Check controller status
rosservice call /controller_manager/list_controllers
```

## Training Configuration

Default MBPO parameters (in `train_gazebo_mbpo_visual.py`):
- **Episodes**: 100 (initial testing)
- **Steps per Episode**: 50  
- **State Dimension**: 15
- **Action Dimension**: 4
- **Model Training Frequency**: Every 250 steps
- **Batch Size**: 256
- **Success Threshold**: 2cm distance to target

## Monitoring Training

The training script provides real-time logging:
- Episode rewards and lengths
- Success rate tracking  
- Distance to target statistics
- Model training progress
- Policy update status

Visualization in Gazebo shows:
- Robot movement in real-time
- Target sphere positioning
- Workspace boundaries
- End-effector trajectory

## Future Improvements

1. **Multi-Target Training**: Support for multiple simultaneous targets
2. **Obstacle Avoidance**: Add workspace obstacles for complex navigation
3. **Curriculum Learning**: Progressive difficulty increase
4. **Distributed Training**: Multi-environment parallel training
5. **Real Robot Transfer**: Sim-to-real deployment preparation

---

**Note**: This system closely follows the architecture of the original robotic_arm_environment project, adapted for ROS Noetic with enhanced visual simulation capabilities.