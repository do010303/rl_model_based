# Gazebo MBPO Integration

This document describes the integration between the MBPO (Model-Based Policy Optimization) reinforcement learning system and Gazebo simulation for the 4-DOF robot arm.

## Overview

The integration allows training RL agents directly on the Gazebo-simulated robot, providing:

- **Realistic Physics**: Full Gazebo physics simulation with joint dynamics, friction, and collision detection
- **Direct Control**: Individual joint position controllers for fast, precise control
- **ROS Integration**: Seamless communication between Python RL code and Gazebo via ROS
- **Automated Management**: Automatic Gazebo startup/shutdown and workspace management

## Architecture

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   MBPO Trainer      │    │  Robot Control       │    │   Gazebo Simulation │
│                     │    │  Interface (ROS)     │    │                     │
│ - GazeboRobot4DOFEnv├────┤                      ├────┤ - 4-DOF Robot       │
│ - DDPG Agent        │    │ - Joint Commands     │    │ - Position Controllers│
│ - Dynamics Models   │    │ - State Feedback     │    │ - Physics Engine    │
│ - Replay Buffer     │    │ - Reset Service      │    │ - Real-time Sim     │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
```

## Key Components

### 1. Gazebo Environment (`environments/gazebo_robot_4dof_env.py`)

Enhanced version of `Robot4DOFEnv` that communicates with Gazebo:

- **ROS Integration**: Publishes joint commands, subscribes to joint states
- **Automatic Fallback**: Falls back to internal simulation if ROS unavailable
- **Real-time Control**: 20Hz control loop with Gazebo simulation
- **State Synchronization**: Real-time joint state feedback from Gazebo

```python
# Usage
from environments.gazebo_robot_4dof_env import GazeboRobot4DOFEnv

env = GazeboRobot4DOFEnv(config={
    'use_ros': True,
    'max_steps': 200,
    'success_distance': 0.02
})
```

### 2. Robot Control Interface (`robot_ws/src/new_robot_arm_urdf/scripts/robot_control_interface.py`)

ROS node that bridges Python and Gazebo:

- **Joint Commands**: Receives position commands from RL agent
- **State Publishing**: Publishes joint states and end-effector pose
- **Reset Service**: Provides robot reset functionality
- **Forward Kinematics**: Computes end-effector position

### 3. Gazebo MBPO Trainer (`gazebo_mbpo_trainer.py`)

Complete training system with Gazebo integration:

- **Automatic Gazebo Management**: Starts/stops Gazebo simulation
- **Workspace Building**: Builds ROS workspace automatically
- **Training Loop**: Full MBPO training with model learning and policy optimization
- **Checkpointing**: Saves models and training statistics

```python
# Usage
from gazebo_mbpo_trainer import GazeboMBPOTrainer

trainer = GazeboMBPOTrainer(config={
    'num_epochs': 100,
    'headless': True,
    'auto_manage_gazebo': True
})
trainer.train()
```

### 4. ROS Control Configuration

**Individual Position Controllers** (`config/control.yaml`):
```yaml
joint1_position_controller:
  type: position_controllers/JointPositionController
  joint: Joint_1
  pid: {p: 1000.0, i: 0.1, d: 100.0}
```

**URDF Transmissions** for all 4 joints with proper hardware interfaces.

**Optimized Launch File** (`launch/rl_training.launch`):
- Headless mode for training
- Fast simulation parameters
- Automatic controller spawning

## Usage

### Quick Start

1. **Test Integration**:
```bash
cd /home/ducanh/rl_model_based
python scripts/test_gazebo_integration.py
```

2. **Test Gazebo Connection**:
```bash
python examples/train_gazebo_mbpo.py --mode test
```

3. **Start Training**:
```bash
# Headless training (fastest)
python examples/train_gazebo_mbpo.py --headless --epochs 100

# With GUI for debugging
python examples/train_gazebo_mbpo.py --no-headless --epochs 10
```

### Advanced Usage

**Custom Configuration**:
```bash
# Create custom config
cat > my_config.json << EOF
{
  "num_epochs": 200,
  "headless": true,
  "env_config": {
    "success_distance": 0.015,
    "workspace_radius": 0.12
  },
  "agent_config": {
    "lr_actor": 0.0003,
    "noise_std": 0.15
  }
}
EOF

# Train with custom config
python examples/train_gazebo_mbpo.py --config my_config.json
```

**Direct Trainer Usage**:
```python
from gazebo_mbpo_trainer import GazeboMBPOTrainer

config = {
    "num_epochs": 50,
    "steps_per_epoch": 150,
    "headless": True,
    "env_config": {"success_distance": 0.02}
}

trainer = GazeboMBPOTrainer(config)
success = trainer.train()
```

### Manual Gazebo Management

If you prefer to manage Gazebo manually:

```bash
# Terminal 1: Start Gazebo
cd /home/ducanh/rl_model_based/robot_ws
source devel/setup.bash
roslaunch new_robot_arm_urdf rl_training.launch

# Terminal 2: Run training
python examples/train_gazebo_mbpo.py --config '{"auto_manage_gazebo": false}'
```

## Configuration Options

### Training Parameters
- `num_epochs`: Number of training epochs (default: 100)
- `steps_per_epoch`: Environment steps per epoch (default: 200)
- `num_trains_per_epoch`: Policy training steps per epoch (default: 4)
- `batch_size`: Training batch size (default: 64)

### Model-Based Parameters
- `num_dynamics_models`: Ensemble size (default: 5)
- `rollout_length`: Model rollout horizon (default: 5)
- `num_model_rollouts`: Synthetic rollouts per epoch (default: 400)

### Environment Parameters
- `max_steps`: Max steps per episode (default: 200)
- `success_distance`: Goal achievement threshold in meters (default: 0.02)
- `workspace_radius`: Reachable workspace radius (default: 0.15)
- `dense_reward`: Use dense vs sparse rewards (default: true)

### Gazebo Parameters
- `headless`: Run without GUI (default: true)
- `auto_manage_gazebo`: Automatic Gazebo lifecycle (default: true)
- `real_time_factor`: Simulation speed (0 = maximum) (default: 0)

## Troubleshooting

### Common Issues

**1. Gazebo Won't Start**
```bash
# Check ROS installation
roscore &
# Should start without errors

# Check workspace
cd /home/ducanh/rl_model_based/robot_ws
catkin_make
source devel/setup.bash
roslaunch new_robot_arm_urdf rl_training.launch gui:=true
```

**2. Joint Controllers Not Loading**
```bash
# Check controller manager
rosservice call /controller_manager/list_controllers

# Restart controllers
rosservice call /controller_manager/reload_controller_libraries
```

**3. ROS Communication Issues**
```bash
# Check topics
rostopic list
rostopic echo /joint_states

# Check services
rosservice list | grep robot
```

**4. Python Import Errors**
```bash
# Activate virtual environment
source .venv/bin/activate

# Install missing packages
pip install numpy gymnasium tensorflow rospy
```

### Performance Optimization

**For Faster Training**:
- Set `headless: true`
- Use `real_time_factor: 0` (maximum speed)
- Reduce `max_step_size` for stability
- Decrease `steps_per_epoch` for quick iterations

**For Better Learning**:
- Increase `num_dynamics_models` (ensemble size)
- Tune PID controller gains in `control.yaml`
- Adjust `success_distance` based on task difficulty
- Use `dense_reward: true` for faster convergence

## File Structure

```
/home/ducanh/rl_model_based/
├── environments/
│   ├── robot_4dof_env.py          # Original environment
│   └── gazebo_robot_4dof_env.py   # Gazebo-integrated environment
├── gazebo_mbpo_trainer.py         # Main Gazebo trainer
├── examples/
│   └── train_gazebo_mbpo.py       # Training examples
├── scripts/
│   └── test_gazebo_integration.py # Integration tests
├── robot_ws/                      # ROS workspace
│   └── src/new_robot_arm_urdf/
│       ├── config/control.yaml    # Controller configuration
│       ├── launch/rl_training.launch # Training launch file
│       ├── scripts/robot_control_interface.py # ROS bridge
│       └── urdf/New.xacro         # Robot description
└── checkpoints/gazebo_mbpo/       # Saved models
```

## Next Steps

1. **Run Integration Tests**: Verify all components work
2. **Start with Short Training**: Test with 10-20 epochs
3. **Monitor Performance**: Check training plots and success rates
4. **Optimize Parameters**: Tune for your specific task
5. **Scale Up**: Increase epochs and complexity

## Support

For issues or questions:
1. Run the integration tests first
2. Check Gazebo and ROS logs
3. Verify workspace builds correctly
4. Test individual components separately