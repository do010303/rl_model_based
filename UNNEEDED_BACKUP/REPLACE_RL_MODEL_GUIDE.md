# Replacing RL Model - Simple Integration Guide

## ✅ Current State
You have a **working RL training system**:
- ✅ `train_robot.py` - Training loop (random actions currently)
- ✅ `main_rl_environment_noetic.py` - Gazebo environment 
- ✅ `test_simple_movement.py` - Verified 3.5s action timing
- ✅ `ddpg_4dof_noetic.py` - Existing DDPG (PyTorch)

## 🎯 Goal
Replace the DDPG agent with the one from `Robotarm-RL-backup-20251103_172859/`

## 📋 Option 1: Quick Integration (Recommended)

Just update `train_robot.py` to use the backup DDPG:

### Step 1: Copy Backup Files
```bash
cd ~/rl_model_based
cp -r Robotarm-RL-backup-20251103_172859/agents robot_ws/src/new_robot_arm_urdf/scripts/
cp -r Robotarm-RL-backup-20251103_172859/utils robot_ws/src/new_robot_arm_urdf/scripts/
cp -r Robotarm-RL-backup-20251103_172859/replay_memory robot_ws/src/new_robot_arm_urdf/scripts/
```

### Step 2: Update train_robot.py

Replace the random action section with DDPG:

```python
#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import rospy
import time
import numpy as np
from main_rl_environment_noetic import RLEnvironmentNoetic
from agents.ddpg import DDPGAgent  # ← Add this import

def train_robot(num_episodes=500, max_steps_per_episode=5):  # ← Short episodes
    """Train with DDPG agent"""
    
    rospy.init_node('robot_ddpg_trainer', anonymous=True)
    
    # Create environment
    env = RLEnvironmentNoetic(max_episode_steps=max_steps_per_episode)
    time.sleep(3)
    
    # Create DDPG agent (TensorFlow from backup)
    state_dim = 10  # [3 ee_pos + 4 joints + 3 target]
    agent = DDPGAgent(env=None, input_dims=state_dim)  # ← Create agent
    agent.n_actions = 4
    agent.max_action = 1.0
    agent.min_action = -1.0
    
    print(f"🤖 Training with DDPG for {num_episodes} episodes")
    print(f"📊 Episode length: {max_steps_per_episode} actions (~{max_steps_per_episode*3.5:.1f}s)")
    
    # Training loop
    for episode in range(num_episodes):
        env.reset_environment()
        observation = env.get_state()
        episode_reward = 0
        
        for step in range(max_steps_per_episode):
            # Get action from DDPG agent
            action = agent.choose_action(observation, evaluate=False)  # ← DDPG chooses action
            
            # Denormalize action to joint positions
            joint_limits_low = np.array([-np.pi, -np.pi/2, -np.pi/2, -np.pi/2])
            joint_limits_high = np.array([np.pi, np.pi/2, np.pi/2, np.pi/2])
            joint_pos = joint_limits_low + (action + 1) * 0.5 * (joint_limits_high - joint_limits_low)
            
            # Execute action (3.5s timing from test_simple_movement.py)
            env.move_to_joint_positions(joint_pos)
            time.sleep(3.5)  # ← Verified timing
            
            # Get next state and reward
            next_observation = env.get_state()
            reward, done = env.calculate_reward()
            
            # Store experience
            agent.remember(observation, action, reward, next_observation, done)  # ← Store
            
            observation = next_observation
            episode_reward += reward
            
            if done:
                break
        
        # Learn from experience (after each episode)
        for _ in range(40):  # 40 gradient updates per episode
            actor_loss, critic_loss = agent.learn()  # ← Learning
        
        # Log progress
        distance = env.get_distance_to_goal()
        print(f"Episode {episode+1}/{num_episodes}: reward={episode_reward:.2f}, distance={distance:.4f}m")
        
        # Save checkpoint every 25 episodes
        if (episode + 1) % 25 == 0:
            agent.save_models()
            print(f"💾 Checkpoint saved at episode {episode+1}")

if __name__ == '__main__':
    train_robot(num_episodes=500, max_steps_per_episode=5)
```

### Step 3: Run Training

```bash
# Terminal 1: Launch Gazebo
cd ~/rl_model_based/robot_ws
source devel/setup.bash
roslaunch new_robot_arm_urdf robot_4dof_rl_gazebo.launch

# Terminal 2: Start training
cd ~/rl_model_based/robot_ws/src/new_robot_arm_urdf/scripts
source ~/rl_model_based/robot_ws/devel/setup.bash
python3 train_robot.py
```

## 📋 Option 2: Minimal Changes

Keep everything as-is, just replace the action selection in `train_robot.py`:

```python
# Add at top of train_robot.py:
sys.path.append('/home/ducanh/rl_model_based')
from agents.ddpg_gazebo import DDPGAgentGazebo

# In train_robot() function, before training loop:
agent = DDPGAgentGazebo(state_dim=10, n_actions=4)

# In episode loop, replace:
# action = env.generate_random_action()
# With:
action = agent.choose_action(observation, evaluate=False)

# After getting reward, add:
agent.remember(observation, action, reward, next_observation, done)

# After episode ends, add:
for _ in range(40):
    agent.learn()
```

## 🎯 Key Points

1. **State Space**: 10D `[3 ee_pos, 4 joints, 3 target_pos]`
2. **Action Space**: 4D normalized `[-1, 1]` → denormalize to joint limits
3. **Action Timing**: 3.5s per action (3s trajectory + 0.5s buffer)
4. **Episode Length**: 5 actions = 17.5s per episode
5. **Learning**: 40 gradient updates after each episode
6. **Total Time**: 500 episodes × 17.5s = ~2.4 hours

## 🔧 What Changed

**Before** (in your current `train_robot.py`):
```python
action = env.generate_random_action()  # Random exploration
env.execute_action(action)             # Old method
```

**After** (with DDPG):
```python
action = agent.choose_action(state, evaluate=False)  # DDPG policy
joint_pos = denormalize(action)                      # Convert to radians
env.move_to_joint_positions(joint_pos)               # New fast method (3.5s)
time.sleep(3.5)                                      # Verified timing
```

## ✅ Verification

After starting training, you should see:
```
🤖 Training with DDPG for 500 episodes
📊 Episode length: 5 actions (~17.5s)
✅ DDPG Agent initialized:
   State dim: 10, Actions: 4
   
Episode 1/500: reward=-15.34, distance=0.1234m
Episode 2/500: reward=-12.87, distance=0.0987m
...
💾 Checkpoint saved at episode 25
```

---

**Simple Answer**: Just update `train_robot.py` to import and use the backup DDPG agent instead of random actions!
