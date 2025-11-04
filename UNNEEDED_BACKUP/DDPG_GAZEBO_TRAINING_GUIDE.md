# DDPG RL Training for Gazebo 4DOF Robot - Quick Start Guide

## 🎯 Overview

This integration brings the DDPG (Deep Deterministic Policy Gradient) agent from your backup RL project into the working Gazebo environment with **optimized fast training**:

- **Episode Length**: Only 5 actions per episode (~17.5 seconds)
- **Action Speed**: 3.5s per action (same as test_simple_movement.py)
- **State Space**: 14D [4 joint angles, 4 velocities, 3 ee_pos, 3 target_pos]
- **Persistent Subscriber**: Instant joint state reads (no overhead)

## 📁 Project Structure

```
rl_model_based/
├── agents/
│   ├── __init__.py
│   └── ddpg_gazebo.py          # DDPG agent adapted for Gazebo
├── utils/
│   ├── __init__.py
│   └── networks.py             # Actor/Critic neural networks
├── replay_memory/
│   ├── __init__.py
│   └── ReplayBuffer.py         # Experience replay buffer
├── checkpoints/
│   └── ddpg_gazebo/            # Model checkpoints saved here
├── train_gazebo_ddpg_short.py  # Main training script
└── robot_ws/
    └── src/new_robot_arm_urdf/scripts/
        ├── main_rl_environment_noetic.py  # Gazebo environment
        └── test_simple_movement.py        # Verified working test
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# TensorFlow 2.x for DDPG neural networks
pip3 install tensorflow==2.12.0  # Or compatible version

# Other dependencies (should already be installed)
pip3 install numpy matplotlib
```

### 2. Launch Gazebo Environment

Open **Terminal 1**:
```bash
cd ~/rl_model_based/robot_ws
source devel/setup.bash
roslaunch new_robot_arm_urdf test_rl_environment.launch
```

Wait until Gazebo loads and you see:
- Robot arm at home position
- Target sphere (small red ball) visible

### 3. Start Training

Open **Terminal 2**:
```bash
cd ~/rl_model_based
source robot_ws/devel/setup.bash
python3 train_gazebo_ddpg_short.py
```

## 📊 Training Configuration

Located at the top of `train_gazebo_ddpg_short.py`:

```python
# Episode settings
NUM_EPISODES = 500              # Total training episodes
MAX_STEPS_PER_EPISODE = 5       # Only 5 actions per episode
LEARNING_STARTS = 50            # Start learning after 50 episodes

# Training settings
OPT_STEPS_PER_EPISODE = 40      # Gradient updates per episode
SAVE_INTERVAL = 25              # Save checkpoints every 25 episodes
EVAL_INTERVAL = 10              # Evaluate (no noise) every 10 episodes

# Reward settings
GOAL_THRESHOLD = 0.02           # 2cm success threshold
DISTANCE_WEIGHT = 10.0          # Penalty for distance
SUCCESS_BONUS = 50.0            # Bonus for reaching goal
STEP_PENALTY = 0.1              # Time penalty per step

# Action timing (from test_simple_movement.py)
TRAJECTORY_TIME = 3.0           # Trajectory execution
BUFFER_TIME = 0.5               # Buffer time
ACTION_WAIT_TIME = 3.5          # Total: 3.5s per action
```

## 📈 Expected Training Progress

### Episode Timeline
```
Episode time = 5 actions × 3.5s = 17.5 seconds
500 episodes × 17.5s = ~145 minutes (2.4 hours total)
```

### Training Stages

**Stage 1: Exploration (Episodes 1-100)**
- Random exploration with high noise
- Learning joint limits and workspace
- Success rate: 0-10%
- Avg reward: Very negative (-50 to -20)

**Stage 2: Learning (Episodes 100-300)**
- Agent discovers reward structure
- Starts approaching target consistently
- Success rate: 10-40%
- Avg reward: Improving (-20 to 0)

**Stage 3: Optimization (Episodes 300-500)**
- Fine-tuning policy for precision
- Higher success rate on goal reaching
- Success rate: 40-70% (goal: >50%)
- Avg reward: Positive (0 to +30)

## 📁 Outputs

The training script generates:

1. **Model Checkpoints** (`checkpoints/ddpg_gazebo/`):
   - `actor.h5`, `critic.h5` - Best models (highest avg reward)
   - `actor_ep25.h5`, `critic_ep25.h5` - Periodic checkpoints
   - `target_actor.h5`, `target_critic.h5` - Target networks

2. **Training Results** (`training_results_YYYYMMDD_HHMMSS.pkl`):
   - Episode rewards, distances, successes
   - Actor/Critic losses
   - Training configuration

3. **Training Plot** (`training_plot_YYYYMMDD_HHMMSS.png`):
   - Episode rewards with moving average
   - Minimum distance to goal
   - Success rate over time
   - Training losses

## 🔧 Hyperparameter Tuning

### If training is too slow:
```python
NUM_EPISODES = 300              # Reduce total episodes
MAX_STEPS_PER_EPISODE = 3       # Reduce to 3 actions (10.5s/episode)
OPT_STEPS_PER_EPISODE = 20      # Less gradient updates
```

### If success rate is too low:
```python
GOAL_THRESHOLD = 0.03           # Increase to 3cm (easier)
SUCCESS_BONUS = 100.0           # Bigger reward for success
DISTANCE_WEIGHT = 15.0          # Stronger penalty for distance
```

### If robot is too conservative:
```python
# In agents/ddpg_gazebo.py:
NOISE_FACTOR = 0.2              # Increase exploration (default: 0.15)
```

### If learning is unstable:
```python
# In agents/ddpg_gazebo.py:
ALPHA = 5e-5                    # Lower actor learning rate
BETA = 5e-4                     # Lower critic learning rate
```

## 🎮 Testing Trained Agent

After training, test the learned policy:

```python
#!/usr/bin/env python3
"""Test trained DDPG agent"""
import rospy
import numpy as np
from agents.ddpg_gazebo import DDPGAgentGazebo
from robot_ws.src.new_robot_arm_urdf.scripts.main_rl_environment_noetic import RLEnvironmentNoetic

rospy.init_node('test_ddpg')

# Load environment
env = RLEnvironmentNoetic(max_episode_steps=10)
rospy.sleep(3.0)

# Load trained agent
agent = DDPGAgentGazebo(state_dim=14, n_actions=4)
agent.load_models()  # Load best model

# Test 10 episodes
for ep in range(10):
    state = env.get_state()
    env.reset_environment()
    
    for step in range(10):
        action = agent.choose_action(state, evaluate=True)  # No noise
        
        # Execute action
        joint_pos = denormalize_action(action)
        env.move_to_joint_positions(joint_pos)
        rospy.sleep(3.5)
        
        # Get next state
        state = env.get_state()
        
        # Calculate distance
        distance = env.get_distance_to_goal()
        print(f"Ep {ep+1}, Step {step+1}: distance = {distance:.4f}m")
        
        if distance < 0.02:
            print(f"✅ SUCCESS in {step+1} steps!")
            break
    
    rospy.sleep(2.0)
```

## 📚 Key Differences from Backup Project

| Aspect | Backup Project | Gazebo Integration |
|--------|----------------|-------------------|
| Environment | Custom DrawingArm4DoF | ROS Noetic Gazebo |
| Episode Length | 100 steps | **5 steps** (17.5s) |
| Action Time | Instant | **3.5s** (realistic) |
| State Space | 12D (planar) | **14D** (3D + velocities) |
| Joint States | Simulated | **Real /joint_states** topic |
| Subscriber | N/A | **Persistent** (instant reads) |
| Target | Random 2D | **3D drawing surface** |

## 🐛 Troubleshooting

### Error: "Could not get joint states"
**Solution**: Make sure `/joint_states` topic is being published:
```bash
rostopic hz /joint_states  # Should show ~100Hz
```

### Error: "Import tensorflow could not be resolved"
**Solution**: Install TensorFlow:
```bash
pip3 install tensorflow
```

### Training too slow (>20s per episode)
**Problem**: Action execution taking too long
**Solution**: Check trajectory time in `main_rl_environment_noetic.py`:
```python
# Should be 3.0s, not 5.0s
point.time_from_start = rospy.Duration(3.0)
```

### Robot oscillates indefinitely
**Expected**: This is normal due to PID gains
**Impact**: None - positions still accurate (<0.002 rad error)
**Note**: The 3.5s wait (3s trajectory + 0.5s buffer) is optimized for this

### Success rate stuck at 0%
**Possible causes**:
1. Goal threshold too tight → Increase `GOAL_THRESHOLD` to 0.03
2. Not enough exploration → Increase `NOISE_FACTOR` to 0.25
3. Learning not started → Wait until episode 50 (LEARNING_STARTS)

## 📖 Next Steps

1. **Monitor Training**: Watch the terminal output and Gazebo visualization
2. **Adjust Hyperparameters**: Tune based on training progress
3. **Evaluate Checkpoints**: Test models at different training stages
4. **Scale Up**: Increase episodes to 1000+ for better convergence
5. **Advanced Rewards**: Implement shaped rewards for faster learning

## 🎯 Success Criteria

A well-trained agent should achieve:
- **Success Rate**: >50% (reaching within 2cm)
- **Avg Reward**: >0 (positive over 100 episodes)
- **Min Distance**: <0.03m average (3cm)
- **Consistency**: Low variance in performance

## 💡 Tips

1. **Let it run**: First 100 episodes are mostly exploration
2. **Check Gazebo**: Watch robot movements to debug behavior
3. **Save often**: Checkpoints every 25 episodes prevent data loss
4. **Tune rewards**: Most important hyperparameters for success
5. **Use plots**: Visualize training progress to identify issues

---

**Created**: November 3, 2025  
**Based on**: test_simple_movement.py (verified working)  
**Training time**: ~2.4 hours for 500 episodes  
**Success target**: >50% success rate @ 2cm threshold
