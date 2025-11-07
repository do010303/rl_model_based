# Target Distance Updated to 15cm

## Changes Made

Updated the target sphere and drawing surface distance from **7.5cm** to **15cm** from the robot base.

## Files Modified

### 1. ✅ `robot_ws/src/new_robot_arm_urdf/worlds/rl_training_world.world`
**Drawing surface position:**
```xml
<!-- BEFORE: -->
<pose frame=''>0.075 0 0.12 0 1.57079 0</pose>  <!-- 7.5cm from base -->

<!-- AFTER: -->
<pose frame=''>0.15 0 0.12 0 1.57079 0</pose>  <!-- 15cm from base -->
```

### 2. ✅ `robot_ws/src/new_robot_arm_urdf/launch/robot_4dof_rl_gazebo.launch`
**Sphere spawn position:**
```xml
<!-- BEFORE: -->
-x 0.075 -y 0.0 -z 0.14

<!-- AFTER: -->
-x 0.15 -y 0.0 -z 0.14
```

### 3. ✅ `robot_ws/src/new_robot_arm_urdf/scripts/main_rl_environment_noetic.py`
**Multiple locations updated:**

a) **Configuration constants (line 4):**
```python
# BEFORE:
SURFACE_X = 0.075  # 7.5cm from robot base

# AFTER:
SURFACE_X = 0.15  # 15cm from robot base
```

b) **Target randomization (line 107):**
```python
# BEFORE:
drawing_surface_x = 0.075   # 7.5cm

# AFTER:
drawing_surface_x = 0.15   # 15cm
```

c) **Reset target position (line 733):**
```python
# BEFORE:
surface_x = 0.075 - 0.008  # 6.7cm

# AFTER:
surface_x = 0.15 - 0.008  # 14.2cm
```

## What You Need to Do

### ⚠️ MUST RESTART GAZEBO

The world file changes require a Gazebo restart:

```bash
# Stop current Gazebo (Ctrl+C in the terminal running launch)

# Then restart:
cd /home/ducanh/rl_model_based/robot_ws
source devel/setup.bash
roslaunch new_robot_arm_urdf robot_4dof_rl_gazebo.launch
```

### Expected Results

After restart, you should see:
- ✅ Drawing surface at X=15cm (further from robot base)
- ✅ Red sphere (1cm radius) sitting ON the white surface
- ✅ Robot has more space to maneuver
- ✅ Targets at a comfortable reaching distance

## Why 15cm is Better

### Advantages:
- **More workspace**: Robot can reach targets without being too cramped
- **Better visualization**: Easier to see robot movements
- **Less collision risk**: More clearance between robot and surface
- **Original design**: This was the original distance before optimization attempts

### Reachability:
Based on previous workspace analysis:
- At X=0.075m (7.5cm): ~4.6% coverage (too close, limited configurations)
- At X=0.10m (10cm): ~3.6% coverage
- At X=0.15m (15cm): ~2.8% coverage (but more comfortable to reach)

**Note**: Lower coverage % is acceptable because:
- RL learns non-random policies (much better than random sampling)
- 15cm provides better working space for robot movements
- More natural reaching distance for the robot arm

## Technical Details

### All Files Now Synchronized:
```
World file:      Surface at X=0.15m ✅
Launch file:     Sphere at X=0.15m ✅
Environment:     SURFACE_X = 0.15m ✅
Target spawn:    X=0.15m ± 0.01m ✅
```

### Coordinate System:
```
Surface Position:  X=0.15m, Y=0.0m, Z=0.12m
Sphere Center:     X=0.142m (15cm - 8mm offset), Y=random, Z=random
Sphere Range:      Y∈[-14, +14]cm, Z∈[5, 22]cm
Sphere Radius:     1cm
```

### Why 8mm Offset?
The sphere is spawned 8mm (0.008m) in front of the surface so when the robot's end-effector touches the sphere, it's not colliding with the surface itself. This gives a small safety clearance.

---
**Date**: November 5, 2025  
**Change**: Target distance updated from 7.5cm → 15cm for better workspace
