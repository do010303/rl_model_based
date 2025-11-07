# Gazebo Trajectory Visualization Guide

## Overview

Now you can see the robot's trajectory line **DIRECTLY in Gazebo** (not just RViz)!

The system uses **two parallel trajectory drawers**:
1. **RViz Drawer** - Fast, smooth lines using visualization_msgs/Marker
2. **Gazebo Drawer** - Visible in Gazebo using spawned cylinder models

## How It Works

### RViz Trajectory (Original)
- Uses `visualization_msgs/Marker` with LINE_STRIP
- Fast and lightweight
- Only visible in RViz
- Green smooth line

### Gazebo Trajectory (NEW!)
- Uses Gazebo's `spawn_sdf_model` service
- Each line segment is a thin cylinder
- Visible directly in Gazebo simulation
- Slightly slower (spawns actual models)
- Green cylinders connecting trajectory points

## Technical Details

### Gazebo Line Drawing Method

Each line segment between two trajectory points becomes a **cylinder**:

1. **Calculate midpoint** between two consecutive points
2. **Calculate cylinder length** (distance between points)
3. **Calculate orientation** to align cylinder with line direction
4. **Spawn cylinder model** in Gazebo with correct pose
5. **Color and size** match the trajectory appearance

### Cylinder Parameters

- **Radius**: 1.5mm (3mm diameter line)
- **Color**: Green (RGB: 0, 1, 0)
- **Material**: Slightly emissive for visibility
- **Static**: Yes (doesn't fall or collide)

### Performance

- **RViz drawer**: Updates instantly, no limit on points
- **Gazebo drawer**: ~0.1s per segment (spawning overhead)
- **Recommendation**: Use both for best experience
  - Watch training in Gazebo (realistic)
  - Analyze trajectory in RViz (smooth, detailed)

## Files Changed

### New Files

1. **`gazebo_trajectory_drawer.py`** - Gazebo trajectory drawing system
   - `GazeboTrajectoryDrawer` class
   - Cylinder spawning logic
   - Pose calculation (position + quaternion rotation)
   - Clear functionality

2. **`test_gazebo_trajectory.py`** - Test script for Gazebo drawer
   - Draws spiral pattern
   - Verifies Gazebo visibility

### Modified Files

1. **`main_rl_environment_noetic.py`**
   - Import `GazeboTrajectoryDrawer`
   - Create `self.gazebo_drawer` instance
   - Add points to both drawers
   - Clear both drawers on reset

## Usage

### Automatic (During Training)

The trajectory is drawn automatically when training:

```bash
# 1. Start Gazebo
roslaunch new_robot_arm_urdf robot_4dof_rl_gazebo.launch

# 2. Run training (in another terminal)
cd /home/ducanh/rl_model_based/robot_ws/src/new_robot_arm_urdf/scripts
python3 train_robot.py
```

**What you'll see**:
- Green cylinders appearing in Gazebo as robot moves
- Same trajectory also visible in RViz (if running)
- Trajectory clears at start of each episode

### Manual Test

Test Gazebo drawing with spiral pattern:

```bash
# Terminal 1: Start Gazebo
roslaunch new_robot_arm_urdf robot_4dof_rl_gazebo.launch

# Terminal 2: Run test script
cd /home/ducanh/rl_model_based/robot_ws/src/new_robot_arm_urdf/scripts
python3 test_gazebo_trajectory.py
```

**Expected output**:
- Spiral pattern drawn with green cylinders
- Visible directly in Gazebo window
- Auto-clears after 15 seconds

## Advantages & Limitations

### Advantages ✅

- **Visible in Gazebo** - No need to run RViz
- **Integrated with simulation** - True 3D perspective
- **Persistent** - Stays visible until cleared
- **Accurate positioning** - Uses same coordinates as RViz

### Limitations ⚠️

- **Slower than RViz** - Each segment spawns a model (~0.1s)
- **More memory** - Each cylinder is a Gazebo model
- **Can accumulate** - Long episodes create many cylinders
- **No curve smoothing** - Straight segments between points

### When to Use Each

| Scenario | Use RViz | Use Gazebo | Use Both |
|----------|----------|------------|----------|
| Fast training | ✅ | ❌ | ❌ |
| Visual debugging | ✅ | ✅ | ✅ |
| Demo/presentation | ❌ | ✅ | ✅ |
| Analyzing trajectory | ✅ | ❌ | ❌ |
| Short episodes | ✅ | ✅ | ✅ |

## Configuration

### Change Line Color

Edit in `main_rl_environment_noetic.py`:

```python
# Line ~220
self.gazebo_drawer = GazeboTrajectoryDrawer(
    color='red',        # 'green', 'blue', 'yellow', 'cyan', 'magenta', 'white', 'orange'
    line_width=0.003    # Thickness in meters (3mm default)
)
```

### Change Line Thickness

```python
self.gazebo_drawer = GazeboTrajectoryDrawer(
    color='green',
    line_width=0.005    # Thicker line (5mm)
)
```

### Disable Gazebo Drawing (Keep Only RViz)

Comment out in `main_rl_environment_noetic.py`:

```python
# Line ~423, ~430
# self.gazebo_drawer.add_point_array(current_pos)

# Line ~820
# self.gazebo_drawer.clear()
```

## Troubleshooting

### Problem: No cylinders appear in Gazebo

**Check**:
1. Is Gazebo running? (`roslaunch ... robot_4dof_rl_gazebo.launch`)
2. Check terminal for spawn errors
3. Try test script: `python3 test_gazebo_trajectory.py`

### Problem: Cylinders appear at wrong positions

**Likely cause**: Coordinate frame mismatch

**Fix**: Verify `frame_id='world'` in spawning (already set correctly)

### Problem: Too many cylinders, Gazebo slows down

**Solution**: Reduce trajectory sampling or clear more frequently

```python
# In main_rl_environment_noetic.py, line ~419
min_movement = 0.005  # Increase from 0.002 to 0.005 (less points)
```

### Problem: Cylinders don't clear

**Fix**: Check Gazebo services are available

```bash
rosservice list | grep delete_model
# Should show: /gazebo/delete_model
```

## Comparison: RViz vs Gazebo Drawing

| Feature | RViz Marker | Gazebo Cylinders |
|---------|-------------|------------------|
| **Visibility** | RViz only | Gazebo + RViz |
| **Speed** | Instant | ~0.1s per segment |
| **Memory** | Low | Medium (spawned models) |
| **Smoothness** | Very smooth | Segmented |
| **Color** | Full RGBA | RGB + emissive |
| **Clearing** | Instant | ~0.01s per segment |
| **Best for** | Analysis | Demonstration |

## Example Output

When running training with both drawers:

```
🤖 Initializing Visual RL Environment for 4DOF Robot...
🎨 Trajectory drawer initialized!  (RViz)
🎨 Gazebo trajectory drawer initialized!  (Gazebo)

Episode 1/1000:
  Drawing trajectory... (RViz + Gazebo)
  Steps: 45/200
  ✅ Goal reached! Distance: 0.018m
  
🧹 Clearing trajectory (RViz + Gazebo)...
✨ Trajectory cleared!
🧹 Clearing 89 trajectory segments... (Gazebo)
✨ Trajectory cleared!
```

## Summary

**Before**: Trajectory only visible in RViz ❌  
**Now**: Trajectory visible in **both** RViz AND Gazebo ✅

- **RViz**: Fast, smooth, detailed analysis
- **Gazebo**: Realistic, integrated with simulation, great for demos
- **Both**: Best of both worlds!

---

**Created**: November 7, 2025  
**Feature**: Gazebo trajectory visualization  
**Files**: `gazebo_trajectory_drawer.py`, `test_gazebo_trajectory.py`  
**Status**: ✅ Ready to use
