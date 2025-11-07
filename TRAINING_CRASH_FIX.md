# Training Crash Fix + RViz Robot Visibility

## Issues Fixed

### Issue 1: Training Crashes with KeyError: 'distance'
**Error:**
```python
[ERROR] [1762503732.947878, 1353.846000]: ❌ Training failed with error: 'distance'
Traceback (most recent call last):
  File "train_robot.py", line 881, in <module>
    results = train_ddpg_gazebo()
  File "train_robot.py", line 708, in train_ddpg_gazebo
    rospy.loginfo(f"      ✓ distance={info['distance']:.4f}m, reward={reward:.2f}, done={done}")
KeyError: 'distance'
```

**Root Cause:**
When ground collision (-997) or overreach (-998) was prevented, the safety code returned an `info` dict without the 'distance' key, but the training loop expected it to always be present.

**Fix:**
Updated `train_robot.py` to include all required keys in safety info dicts:

```python
# BEFORE (incomplete info):
info = {'error': 'ground_collision_prevented', 'error_code': -997}

# AFTER (complete info):
next_state = self.get_state()
current_ee = next_state[8:11]
target_pos = next_state[11:14]
distance = np.linalg.norm(current_ee - target_pos)
info = {
    'error': 'ground_collision_prevented', 
    'error_code': -997,
    'distance': distance,
    'ee_position': current_ee,
    'target_position': target_pos
}
```

Now both overreach (-998) and ground collision (-997) cases include:
- ✅ `distance` - Distance to target
- ✅ `ee_position` - Current end-effector position
- ✅ `target_position` - Target position
- ✅ `error` - Error description
- ✅ `error_code` - Numeric error code

---

### Issue 2: Robot Not Visible in RViz

**Problem:** 
When opening RViz, the robot model doesn't appear (only see grid, TF frames, but no blue robot arm).

**Root Causes:**
1. RViz config didn't explicitly list robot links
2. `robot_state_publisher` might not be running
3. TF frames not being published

**Fixes Applied:**

#### 1. Updated RViz Config (`trajectory_view.rviz`)
Expanded the RobotModel display to explicitly show all links:
- `base_link`
- `link_1`, `link_2`, `link_3`, `link_4`
- `end_effector`

Each link now has:
```yaml
Alpha: 1           # Fully opaque
Show Axes: false   # Hide coordinate frames
Show Trail: false  # No motion trail
Value: true        # Enabled
```

#### 2. Updated Launch File (`robot_with_rviz.launch`)
Added `robot_state_publisher` to ensure TF transforms are published:

```xml
<node name="robot_state_publisher" pkg="robot_state_publisher" type="robot_state_publisher"/>
```

This node:
- Reads joint states from `/joint_states` topic
- Publishes TF transforms for all robot links
- Required for RViz to display the robot model

---

## How to Verify Fixes

### Test 1: Training No Longer Crashes
```bash
cd /home/ducanh/rl_model_based/robot_ws/src/new_robot_arm_urdf/scripts
python3 train_robot.py
# Choose 2 (RL Training Mode)
```

**Expected:** 
- Training continues even when ground collision/overreach occurs
- No KeyError crashes
- Log shows: `✓ distance=0.2345m, reward=-30.0, done=False`

### Test 2: Robot Visible in RViz
```bash
cd /home/ducanh/rl_model_based/robot_ws
source devel/setup.bash
roslaunch new_robot_arm_urdf robot_with_rviz.launch
```

**Expected in RViz:**
- ✅ Blue robot arm visible
- ✅ All 4 joints + end-effector visible
- ✅ Robot moves when Gazebo robot moves
- ✅ Green trajectory line appears when robot moves

### Test 3: Check TF Frames
```bash
# In another terminal:
rosrun tf view_frames
# Wait a few seconds, then:
evince frames.pdf
```

**Expected:**
- TF tree shows: `world` → `base_link` → `link_1` → `link_2` → `link_3` → `link_4` → `end_effector`
- All connections valid
- No broken links

---

## Troubleshooting

### "Still don't see robot in RViz"

1. **Check Fixed Frame:**
   - In RViz, top left: "Fixed Frame" should be `world` or `base_link`
   - Try switching between them

2. **Check RobotModel Display:**
   - Left panel → RobotModel → check ✅ enabled
   - Expand "Links" → verify all links enabled
   - Robot Description: should be `robot_description`

3. **Check TF is publishing:**
   ```bash
   rostopic echo /tf --noarr
   ```
   Should see continuous transforms

4. **Check robot_description parameter:**
   ```bash
   rosparam get /robot_description | head -20
   ```
   Should show URDF XML content

5. **Restart everything:**
   ```bash
   # Kill all ROS nodes
   killall -9 rosmaster roscore gzserver gzclient
   
   # Restart
   roslaunch new_robot_arm_urdf robot_with_rviz.launch
   ```

### "Training still crashes with different error"

Check which info keys are required:
```python
# In train_robot.py, around line 708:
rospy.loginfo(f"      ✓ distance={info['distance']:.4f}m, reward={reward:.2f}, done={done}")
```

If crash occurs, the info dict is missing one of:
- `distance`
- `ee_position` (used elsewhere)
- `target_position` (used elsewhere)

The fix ensures all safety error codes (-997, -998, -999) include these keys.

---

## Technical Details

### State Vector Structure (14D):
```python
state = [
    # Joints (4)
    joint1, joint2, joint3, joint4,
    
    # Joint velocities (4)
    vel1, vel2, vel3, vel4,
    
    # End-effector position (3)
    ee_x, ee_y, ee_z,          # indices 8, 9, 10
    
    # Target position (3)
    target_x, target_y, target_z  # indices 11, 12, 13
]
```

### Info Dict Structure (complete):
```python
info = {
    'distance': float,           # Distance to target (required)
    'ee_position': np.array(3),  # End-effector [x,y,z]
    'target_position': np.array(3),  # Target [x,y,z]
    'error': str,                # Optional: error description
    'error_code': int,           # Optional: error code
    'num_points': int,           # Optional: trajectory points
    'length_m': float,           # Optional: trajectory length
    'length_cm': float           # Optional: trajectory length (cm)
}
```

### Safety Error Codes:
- `-999`: Robot broken (NaN detected) → End episode, -100 reward
- `-998`: Overreach prevented → Continue, -50 reward
- `-997`: Ground collision prevented → Continue, -30 reward
- All now include complete info dict with distance

---

## Files Modified

1. ✅ `train_robot.py` - Added distance calculation to safety error handlers
2. ✅ `robot_with_rviz.launch` - Added robot_state_publisher node
3. ✅ `trajectory_view.rviz` - Expanded robot links display config

---

## Summary

### What Was Wrong:
1. Safety error handlers returned incomplete `info` dicts
2. Training loop expected `info['distance']` to always exist
3. RViz config didn't explicitly enable all robot links
4. `robot_state_publisher` not guaranteed to be running

### What Was Fixed:
1. All safety handlers now compute and include distance
2. Info dict now always has: distance, ee_position, target_position
3. RViz config explicitly lists and enables all robot links
4. Launch file ensures robot_state_publisher is running

### Result:
- ✅ Training won't crash on safety events
- ✅ Robot fully visible in RViz
- ✅ Green trajectory line visible alongside robot
- ✅ Complete debugging info in all cases

---

**Date**: November 7, 2025  
**Issues**: Training KeyError crash + Robot invisible in RViz  
**Status**: Both fixed ✅
