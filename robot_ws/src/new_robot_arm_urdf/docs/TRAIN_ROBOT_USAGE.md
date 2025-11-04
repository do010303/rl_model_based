# Training Robot Script Usage Guide

## Overview
`train_robot.py` - Interactive DDPG training script with manual testing mode for 4DOF Gazebo robot.

## Features
✅ **Manual Test Mode** - Test joint angles before training (like `test_simple_movement.py`)  
✅ **RL Training Mode** - Train DDPG agent with customizable episodes/steps  
✅ **Fast Action Execution** - 3.5s per action (3s trajectory + 0.5s buffer)  
✅ **Checkpoint Management** - Auto-creates checkpoint directory  
✅ **Interactive Configuration** - Set episodes and steps at runtime  

## Usage

### 1. Start Gazebo and Robot
```bash
cd ~/rl_model_based/robot_ws
source devel/setup.bash
roslaunch new_robot_arm_urdf gazebo.launch
```

### 2. Run Training Script
```bash
cd ~/rl_model_based/robot_ws/src/new_robot_arm_urdf/scripts
python3 train_robot.py
```

### 3. Choose Mode

**Option 1: Manual Test Mode**
```
🎮 TRAINING MENU
======================================================================
1. Manual Test Mode - Test joint angles manually
2. RL Training Mode - Train DDPG agent
======================================================================
Choose mode (1 or 2): 1
```

Then enter joint angles (radians):
```
Enter 4 joint angles in radians (space-separated):
Example: 0.1 0 0 0
Joint angles: 0.5 0 0 0
```

The robot will execute the movement and show results:
```
RESULTS:
======================================================================
Distance to goal: 0.1234m (12.34cm)
Reward: -1.23
Success: ❌ NO
Execution time: 3.5s
======================================================================
```

**Option 2: RL Training Mode**
```
Choose mode (1 or 2): 2
```

Configure training parameters:
```
⚙️  RL TRAINING CONFIGURATION
======================================================================
Number of episodes (default 500): 10
Steps per episode (default 5): 3
```

Training will start:
```
✅ Configuration:
   Episodes: 10
   Steps per episode: 3
   Estimated time: ~1.8 minutes
======================================================================
📍 Episode 1/10
   🎯 Step 1/3: Executing action...
      ✓ distance=0.1234m, reward=-1.23, done=False
   🎯 Step 2/3: Executing action...
      ✓ distance=0.0987m, reward=-0.98, done=False
   ...
```

## Fixed Issues from Previous Version

### 1. ✅ Robot Not Moving (Episode Ending Early)
**Problem:** Episode finished after 1 step instead of 5  
**Cause:** `done` was set to `True` when `episode_steps >= MAX_STEPS_PER_EPISODE` in reward function  
**Fix:** Moved max steps check to training loop, `done` only set on goal success

### 2. ✅ Missing Checkpoint Directory
**Problem:** `FileNotFoundError: checkpoints/ddpg_gazebo/actor.h5`  
**Cause:** Directory didn't exist  
**Fix:** Auto-create directory:
```python
checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', 'ddpg_gazebo')
os.makedirs(checkpoint_dir, exist_ok=True)
```

### 3. ✅ No Manual Testing
**Problem:** No way to test movements before training  
**Fix:** Added interactive menu with manual test mode

### 4. ✅ Fixed Configuration
**Problem:** Episodes/steps hardcoded as constants  
**Fix:** Interactive input at runtime

## File Locations

**Training Script:**
```
/home/ducanh/rl_model_based/robot_ws/src/new_robot_arm_urdf/scripts/train_robot.py
```

**Checkpoints:**
```
/home/ducanh/rl_model_based/robot_ws/src/new_robot_arm_urdf/scripts/checkpoints/ddpg_gazebo/
├── actor.h5          # Best actor network
├── critic.h5         # Best critic network
├── target_actor.h5   # Best target actor
├── target_critic.h5  # Best target critic
├── actor_ep025.h5    # Checkpoint at episode 25
└── ...
```

**Training Results:**
```
/home/ducanh/rl_model_based/robot_ws/src/new_robot_am_urdf/scripts/
├── training_results_YYYYMMDD_HHMMSS.pkl  # Training metrics
└── training_plot_YYYYMMDD_HHMMSS.png     # Training plots
```

## Training Parameters

**Default Configuration:**
- Episodes: 500
- Steps per episode: 5
- Action time: 3.5s (3s trajectory + 0.5s buffer)
- Episode time: ~17.5s
- Goal threshold: 0.02m (2cm)
- Learning starts: After 50 episodes
- Optimization steps: 40 per episode
- Save interval: Every 25 episodes

**Reward Components:**
- Distance penalty: `-10.0 * distance`
- Step penalty: `-0.1`
- Success bonus: `+50.0` (when distance < 2cm)

## Tips

1. **Test First:** Always run manual test mode to verify robot movement before training
2. **Start Small:** Begin with 10 episodes and 3 steps to test the pipeline
3. **Monitor Distance:** Watch the distance values - should decrease over time
4. **Check Logs:** ROS logs show detailed step-by-step execution
5. **Interrupt Safely:** Press Ctrl+C to stop training (model will save)

## Troubleshooting

**Robot doesn't move:**
- Check Gazebo is running
- Verify `/joint_trajectory_controller/follow_joint_trajectory` action server is active
- Test with manual mode first

**Episode ends after 1 step:**
- This is FIXED in current version
- Check you're using the updated `train_robot.py`

**FileNotFoundError for checkpoints:**
- This is FIXED - directory auto-created
- Check file permissions if still failing

**Import errors:**
- Make sure you're in the correct directory
- Verify `/home/ducanh/rl_model_based/agents/` exists

## Next Steps

After successful training:
1. Check training plots in `training_plot_*.png`
2. Review metrics in `training_results_*.pkl`
3. Test trained model with evaluation script
4. Adjust hyperparameters if needed (in script constants)
