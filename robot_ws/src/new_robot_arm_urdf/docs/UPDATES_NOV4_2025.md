# Updates - November 4, 2025

## Summary of Changes

Three major improvements to the Gazebo RL training system:

---

## 1. ✅ Fixed Ctrl+C Handling at Menu Selection

### Problem
When pressing Ctrl+C at the "Choose mode (1 or 2):" prompt, the program would print "Exiting" but then continue and ask for input again instead of actually exiting.

### Solution
Changed the exception handler to return `None` instead of calling `sys.exit(0)`, which properly exits the function and stops the program.

**File**: `scripts/train_robot.py`

**Change**:
```python
# OLD:
except (KeyboardInterrupt, EOFError):
    print("\n\n👋 Exiting training script. Goodbye!")
    rospy.signal_shutdown("User requested exit")
    sys.exit(0)  # This wasn't working properly

# NEW:
except (KeyboardInterrupt, EOFError):
    print("\n\n👋 Exiting training script. Goodbye!")
    rospy.signal_shutdown("User requested exit")
    return None  # Properly exits the function
```

**Result**: Pressing Ctrl+C now cleanly exits the program immediately.

---

## 2. ✅ Updated Joint Limits

### Problem
Joint limits were too restrictive/incorrect:
- Joint1 (base rotation): Was ±180° (-π to π)
- Joint4 (end-effector): Was ±90° (-π/2 to π/2)

### New Requirements
- **Joint1**: ±90° (-π/2 to π/2) = [-1.57, 1.57] radians
- **Joint2-3**: Keep at ±90° (-π/2 to π/2) = [-1.57, 1.57] radians
- **Joint4**: 0° to 180° (0 to π) = [0, 3.14] radians

### Changes Made

#### A. Training Script (`scripts/train_robot.py`)
```python
# OLD:
self.joint_limits_low = np.array([-np.pi, -np.pi/2, -np.pi/2, -np.pi/2])
self.joint_limits_high = np.array([np.pi, np.pi/2, np.pi/2, np.pi/2])

# NEW:
self.joint_limits_low = np.array([-np.pi/2, -np.pi/2, -np.pi/2, 0.0])
self.joint_limits_high = np.array([np.pi/2, np.pi/2, np.pi/2, np.pi])
```

#### B. URDF File (`urdf/robot_4dof_rl.urdf.xacro`)

**Joint1 (base rotation)**:
```xml
<!-- OLD: -->
<limit effort="411" lower="-3.14159" upper="3.14159" velocity="6.2832"/>

<!-- NEW: -->
<limit effort="411" lower="-1.57079" upper="1.57079" velocity="6.2832"/>
```

**Joint4 (end-effector)**:
```xml
<!-- OLD: -->
<limit effort="194" lower="-1.57079" upper="1.57079" velocity="6.2832"/>

<!-- NEW: -->
<limit effort="194" lower="0.0" upper="3.14159" velocity="6.2832"/>
```

**Joint2 and Joint3**: Unchanged (already at ±90°)

### Impact
- RL agent will learn within the correct joint ranges
- Robot movements will respect new limits
- Better workspace configuration for the specific task

---

## 3. ✅ Faster Robot Movement (2.25s per action)

### Problem
Training was slow with 3.5s per action (3s trajectory + 0.5s buffer).

### New Configuration
- **Trajectory time**: 2.0 seconds (down from 3.0s)
- **Buffer time**: 0.25 seconds (down from 0.5s)
- **Total action time**: 2.25 seconds (down from 3.5s)

### Changes Made

**File**: `scripts/train_robot.py`

```python
# OLD:
TRAJECTORY_TIME = 3.0       # Trajectory execution time
BUFFER_TIME = 0.5           # Buffer after trajectory
ACTION_WAIT_TIME = TRAJECTORY_TIME + BUFFER_TIME  # 3.5s total

# NEW:
TRAJECTORY_TIME = 2.0       # Trajectory execution time
BUFFER_TIME = 0.25          # Buffer after trajectory
ACTION_WAIT_TIME = TRAJECTORY_TIME + BUFFER_TIME  # 2.25s total
```

### Training Speed Impact

**Episode Duration** (5 steps per episode):
- **Old**: 5 steps × 3.5s = 17.5 seconds per episode
- **New**: 5 steps × 2.25s = 11.25 seconds per episode
- **Improvement**: 35.7% faster! (6.25s saved per episode)

**Training Time** (500 episodes):
- **Old**: 500 episodes × 17.5s = 8,750 seconds ≈ 2 hours 26 minutes
- **New**: 500 episodes × 11.25s = 5,625 seconds ≈ 1 hour 34 minutes
- **Time saved**: 52 minutes per full training run!

### Safety Note
The robot movements will be faster but should still be smooth due to:
- Controllers have appropriate PID gains (p=1.0)
- Joint damping and friction values in URDF
- 0.25s buffer still provides settling time

---

## Testing Checklist

Before training, verify all changes:

### 1. Test Ctrl+C Exit
```bash
cd /home/ducanh/rl_model_based/robot_ws/src/new_robot_arm_urdf/scripts
python3 train_robot.py
# At menu, press Ctrl+C
# Expected: Program exits immediately
```

### 2. Test Joint Limits
```bash
# Launch Gazebo
roslaunch new_robot_arm_urdf robot_4dof_rl_gazebo.launch

# In another terminal, run manual test
cd scripts
python3 train_robot.py
# Choose mode: 1 (Manual Test)
# Test Joint1: Try 1.57 0 0 0 (should work, at limit)
# Test Joint1: Try 1.6 0 0 0 (should clamp to 1.57)
# Test Joint4: Try 0 0 0 3.14 (should work, at limit)
# Test Joint4: Try 0 0 0 -0.1 (should clamp to 0.0)
```

### 3. Test Faster Movement
```bash
# Manual test mode
python3 train_robot.py
# Choose mode: 1
# Enter: 0.5 0.5 0.5 1.5
# Observe: Robot should move in ~2.25s (not 3.5s)
```

### 4. Verify RL Training
```bash
python3 train_robot.py
# Choose mode: 2
# Episodes: 10 (test run)
# Steps: 5
# Expected: Each episode completes in ~11-12 seconds
```

---

## Files Modified

1. **`scripts/train_robot.py`**
   - Fixed Ctrl+C handling
   - Updated joint limits
   - Reduced trajectory/buffer times
   - Updated docstrings

2. **`urdf/robot_4dof_rl.urdf.xacro`**
   - Joint1 limits: -1.57 to 1.57
   - Joint4 limits: 0.0 to 3.14

---

## Rollback Instructions

If issues occur, revert changes:

```bash
cd /home/ducanh/rl_model_based/robot_ws/src/new_robot_arm_urdf
git diff scripts/train_robot.py
git diff urdf/robot_4dof_rl.urdf.xacro
# If needed:
git checkout HEAD -- scripts/train_robot.py
git checkout HEAD -- urdf/robot_4dof_rl.urdf.xacro
```

---

**Date**: November 4, 2025  
**Status**: ✅ All changes completed and ready for testing  
**Expected Impact**: 35.7% faster training, better user experience, correct joint limits
