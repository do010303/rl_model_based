# Trajectory Drawing Position Fix

## Issue: Drawing Line Not Following End-Effector

### Problem Description
After setting Joint4 home position to 90° (π/2 radians), the trajectory line was being drawn at the **wrong position** - it appeared at the old end-effector location (when Joint4 was at 0°) instead of the current location.

**User observation:**
> "the drawing line start with the previous position of end eff, but now i set the home position at 90 (for link 4) now, which the drawing line was not following the end eff of the robot at all"

### Visual Example
```
Joint4 = 0° (old home):
    Link4 ─────► End-effector (old position)
    Trajectory drawn here ✓ (correct)

Joint4 = 90° (new home):
         Link4
           │
           │
           ▼
    End-effector (new position)
    
But trajectory was still drawn at old position ✗ (wrong!)
```

---

## Root Cause

### The Bug (in `main_rl_environment_noetic.py`, line 353-359):

```python
# WRONG - Simple offset addition without rotation!
offset = np.array([0.001137, 0.01875, 0.077946])

self.robot_x = pos.x + offset[0]  # ← Treats offset as world coordinates
self.robot_y = pos.y + offset[1]
self.robot_z = pos.z + offset[2]
```

### Why This Was Wrong:

The end-effector offset `[0.001137, 0.01875, 0.077946]` is defined in **link_4's local coordinate frame** (from the URDF).

When you simply add this offset to link_4's world position, you're assuming link_4 has **zero rotation** (i.e., aligned with world axes).

**When Joint4 = 0°:**
- Link4 aligned with world frame
- Simple addition works ✓

**When Joint4 = 90°:**
- Link4 rotated 90° around Y-axis
- The offset vector also needs to be rotated!
- Simple addition gives wrong position ✗

### Example Calculation:

```
Link4 at origin [0, 0, 0], Joint4 = 90°:

Offset in local frame: [0.001, 0.019, 0.078]

When Joint4 = 0° (no rotation):
  World offset = [0.001, 0.019, 0.078]
  End-effector = [0.001, 0.019, 0.078] ✓

When Joint4 = 90° (rotated around Y):
  WRONG (old code): [0.001, 0.019, 0.078] ✗
  CORRECT: rotate offset by 90° around Y
           = [0.078, 0.019, -0.001] ✓
```

The Z-component becomes X, and X becomes -Z (90° rotation around Y-axis).

---

## The Fix

### Implementation (line 346-375):

```python
# Get link_4 position and orientation from Gazebo
pos = link_states.pose[idx].position
ori = link_states.pose[idx].orientation  # Quaternion (x, y, z, w)

# End-effector offset in link_4's LOCAL frame (from URDF)
offset_local = np.array([0.001137, 0.01875, 0.077946])

# Convert quaternion to rotation matrix
qx, qy, qz, qw = ori.x, ori.y, ori.z, ori.w

R = np.array([
    [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
    [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
    [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
])

# Rotate the offset from link_4's local frame to world frame
offset_world = R @ offset_local

# Correct end-effector position!
self.robot_x = pos.x + offset_world[0]
self.robot_y = pos.y + offset_world[1]
self.robot_z = pos.z + offset_world[2]
```

### What This Does:

1. **Get link_4's orientation** as a quaternion from Gazebo
2. **Convert quaternion to rotation matrix** (3x3 matrix `R`)
3. **Rotate the offset vector** from link_4's local frame to world frame
4. **Add rotated offset** to link_4's position

Now the end-effector position is **always correct**, regardless of Joint4's angle!

---

## Mathematical Details

### Quaternion to Rotation Matrix Formula:

Given quaternion `q = (qx, qy, qz, qw)`:

```
R = | 1-2(qy²+qz²)   2(qxqy-qzqw)   2(qxqz+qyqw) |
    | 2(qxqy+qzqw)   1-2(qx²+qz²)   2(qyqz-qxqw) |
    | 2(qxqz-qyqw)   2(qyqz+qxqw)   1-2(qx²+qy²) |
```

This converts the quaternion rotation into a 3x3 rotation matrix.

### Frame Transformation:

```
offset_world = R × offset_local

where:
- offset_local: [x, y, z] in link_4's coordinate frame
- R: Rotation matrix (link_4 → world)
- offset_world: [x', y', z'] in world coordinate frame
```

### Example (Joint4 = 90°):

```
Joint angles: [0, 0, 0, π/2]
Link4 rotated 90° around Y-axis

Offset_local: [0.001137, 0.01875, 0.077946]

R (90° around Y) ≈ | 0  0  1 |
                   | 0  1  0 |
                   |-1  0  0 |

Offset_world = R @ offset_local
             ≈ [0.077946, 0.01875, -0.001137]

Z-component → X-component
X-component → -Z-component
Y-component → Y-component (unchanged, rotation around Y)
```

---

## Verification

### Before Fix:
```bash
# Start simulation
roslaunch new_robot_arm_urdf robot_with_rviz.launch

# Run manual test
python3 train_robot.py
# Choose 1 (Manual Test)

# Move to home: 0 0 0 1.571 (90°)
# Look at RViz:
# - Robot at one position
# - Trajectory line at different position ✗
```

### After Fix:
```bash
# Same commands
# Look at RViz:
# - Robot end-effector and trajectory line ALIGNED ✓
# - Green line starts exactly at robot tip ✓
# - Line follows robot as it moves ✓
```

### Quick Test Script:

```python
# Test FK vs Gazebo with rotation
from fk_ik_utils import fk
import numpy as np

# Home position with Joint4 = 90°
joints_home = np.array([0.0, 0.0, 0.0, np.pi/2])

# FK should give correct position
ee_x, ee_y, ee_z = fk(joints_home)
print(f"FK End-effector: [{ee_x:.4f}, {ee_y:.4f}, {ee_z:.4f}]")

# Compare with what Gazebo reports
# Should match after the fix!
```

---

## Impact

### What Was Affected:
- ✅ Trajectory visualization in RViz
- ✅ Distance calculations to target
- ✅ Reward computation (based on EE-to-target distance)
- ✅ State vector (includes EE position)
- ✅ Training data accuracy

### Why It Matters:

**Before fix:**
- RL agent learned from **incorrect end-effector positions**
- Rewards calculated from **wrong distances**
- Visualization misleading (trajectory not at robot tip)
- Would have prevented successful training!

**After fix:**
- ✅ RL agent sees **correct end-effector positions**
- ✅ Rewards calculated from **accurate distances**
- ✅ Visualization matches reality
- ✅ Training can succeed!

---

## Related Code

### Where End-Effector Position Is Used:

1. **State Vector** (line ~420):
   ```python
   state[8:11] = [robot_x, robot_y, robot_z]  # EE position
   ```

2. **Reward Calculation** (line ~600):
   ```python
   end_effector_pos = np.array([self.robot_x, self.robot_y, self.robot_z])
   distance = np.linalg.norm(end_effector_pos - target_pos)
   ```

3. **Trajectory Drawing** (line ~390):
   ```python
   current_pos = np.array([self.robot_x, self.robot_y, self.robot_z])
   self.trajectory_drawer.add_point_array(current_pos)
   ```

4. **FK Fallback** (line ~375):
   ```python
   ee_x, ee_y, ee_z = fk(self.joint_positions)
   self.robot_x = ee_x  # Already correct (FK handles rotation)
   ```

All of these now use the **correctly rotated** end-effector position!

---

## Technical Notes

### Why FK Fallback Worked:

The FK function in `fk_ik_utils.py` **already properly handles rotation**:

```python
R4 = R3 @ rot_y(joint_angles[3])  # Apply Joint4 rotation
p4 = p3 + R4 @ ee_offset           # Rotate offset by accumulated rotation
```

So when Gazebo data wasn't available, FK gave correct positions. But when using Gazebo's link_4 position, we had to manually rotate the offset.

### Why Use Gazebo Position?

Gazebo's link states are preferred because:
- ✅ Accounts for actual physics simulation
- ✅ Includes joint compliance, friction
- ✅ Reflects real robot behavior
- ✅ More accurate than ideal FK

But we must **always rotate local-frame vectors** when converting to world frame!

---

## Files Modified

- ✅ `main_rl_environment_noetic.py` (lines 346-375)
  - Replaced simple offset addition with proper quaternion rotation
  - Removed TODO comment (now implemented!)

---

## Summary

### The Problem:
End-effector offset was treated as world coordinates instead of link_4's local coordinates.

### The Solution:
Rotate the offset vector by link_4's orientation before adding to link_4's position.

### The Formula:
```
EE_world = Link4_pos_world + R × EE_offset_local

where R = quaternion_to_rotation_matrix(Link4_orientation)
```

### The Result:
- ✅ Trajectory line now correctly follows robot tip
- ✅ Works for ANY Joint4 angle (0°, 90°, or anything else)
- ✅ Accurate RL training data
- ✅ Correct visualization in RViz

---

**Date**: November 7, 2025  
**Issue**: Trajectory line not following end-effector after Joint4 home = 90°  
**Root Cause**: Missing rotation transformation for offset vector  
**Status**: Fixed ✅
