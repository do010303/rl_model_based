# Trajectory Drawing Feature 🎨

## Overview

The robot end-effector now leaves a visual trail as it moves, similar to drawing with a pen! This helps visualize the robot's learning progress and movement patterns.

## Features

✅ **Real-time Visualization**: Green line follows the end-effector in Gazebo/RViz
✅ **Automatic Drawing**: Line appears as robot moves (minimum 2mm movement threshold)
✅ **Manual Clearing**: Clear trajectory with simple commands
✅ **Trajectory Statistics**: Shows path length and number of points
✅ **Episode Auto-Clear**: Automatically clears between RL training episodes

## How It Works

### Technical Details

**Visualization Method**: ROS `visualization_msgs/Marker` (LINE_STRIP type)
**Update Frequency**: Every time end-effector moves ≥2mm
**Color**: Green (`rgb(0, 1, 0)`)
**Line Width**: 3mm (0.003m)
**Frame**: `world` (Gazebo global frame)

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ RLEnvironmentNoetic                                         │
│  │                                                           │
│  ├─ _update_end_effector_position()                        │
│  │   └─> _update_trajectory_drawing()                      │
│  │        └─> trajectory_drawer.add_point()                │
│  │             └─> Publishes to /visualization_marker      │
│  │                                                           │
│  ├─ clear_trajectory()                                     │
│  │   └─> trajectory_drawer.clear()                         │
│  │                                                           │
│  └─ get_trajectory_info()                                  │
│      └─> Returns: num_points, length_m, length_cm          │
└─────────────────────────────────────────────────────────────┘
```

## Usage

### Manual Test Mode

When in manual test mode, you can:

1. **Move the robot** - Type joint angles:
   ```
   Joint angles: 0.5 0.5 0.5 0.5
   ```
   ↳ End-effector draws a green line as it moves

2. **Clear the drawing** - Type any of these commands:
   ```
   Joint angles: clear
   Joint angles: c
   Joint angles: erase
   Joint angles: reset
   ```
   ↳ Green line disappears

3. **View trajectory info** - After each movement:
   ```
   🎨 Trajectory: 245 points, 15.32cm total path
   ```

### RL Training Mode

**Automatic behavior during training:**

- ✅ Draws trajectory during each episode
- ✅ Shows trajectory stats in episode summary
- ✅ **Automatically clears** at the end of each episode
- ✅ Fresh start for next episode

**Example output:**
```
📊 Episode 5 Summary:
   Total reward: -12.50
   Min distance: 0.0234m (2.34cm)
   Success: ❌ NO
   Episode time: 8.5s
   🎨 Trajectory: 127 points, 18.45cm total path
   
🧹 Trajectory cleared!
```

## Examples

### Example 1: Drawing a Circle

```python
# In manual mode, send these commands in sequence:
0.0 0.5 0.5 0.0
0.5 0.5 0.5 0.0
1.0 0.5 0.5 0.0
1.5 0.5 0.5 0.0
# ... continue around
# Result: Green circle in Gazebo!
```

### Example 2: Tracking Learning Progress

```
Episode 1: 🎨 Trajectory: 523 points, 45.23cm (random exploration)
Episode 10: 🎨 Trajectory: 234 points, 23.12cm (learning paths)
Episode 50: 🎨 Trajectory: 89 points, 8.34cm (efficient movement!)
```

## API Reference

### Environment Methods

```python
# Clear the trajectory drawing
env.clear_trajectory()

# Get trajectory statistics
info = env.get_trajectory_info()
# Returns:
# {
#     'num_points': 127,
#     'length_m': 0.1845,
#     'length_cm': 18.45
# }
```

### TrajectoryDrawer Class

Located in `trajectory_drawer.py`:

```python
from trajectory_drawer import TrajectoryDrawer

# Create drawer
drawer = TrajectoryDrawer(
    frame_id='world',      # Reference frame
    color='green',         # Line color
    line_width=0.003       # 3mm thickness
)

# Add points
drawer.add_point(x, y, z)
drawer.add_point_array([x, y, z])

# Clear drawing
drawer.clear()

# Get info
num_points = drawer.get_num_points()
length = drawer.get_trajectory_length()  # in meters

# Change color
drawer.change_color('blue')

# Save/load trajectory
drawer.save_trajectory('path.npy')
drawer.load_trajectory('path.npy')
```

## Visualization

### In Gazebo

The green line appears automatically in the Gazebo scene:
- **Location**: Main 3D viewport
- **Visibility**: Always visible when points exist
- **Persistence**: Remains until cleared

### In RViz (Optional)

To visualize in RViz:

1. Launch RViz:
   ```bash
   rosrun rviz rviz
   ```

2. Add marker display:
   - Click "Add" → "By topic"
   - Select `/visualization_marker`
   - Click OK

3. Set Fixed Frame: `world`

## Configuration

### Change Line Color

In `main_rl_environment_noetic.py`, line ~140:
```python
self.trajectory_drawer = TrajectoryDrawer(
    color='blue',        # Change color here!
    line_width=0.005     # Change thickness here!
)
```

Available colors:
- `'red'`, `'green'`, `'blue'`, `'yellow'`
- `'cyan'`, `'magenta'`, `'white'`, `'orange'`

### Change Movement Threshold

In `_update_trajectory_drawing()` method:
```python
min_movement = 0.002  # 2mm - change this value
```

- **Higher** → Fewer points, less detailed
- **Lower** → More points, smoother line

## Troubleshooting

### Problem: No line appears

**Solutions:**
1. Check ROS topic:
   ```bash
   rostopic echo /visualization_marker
   ```
   Should show marker messages

2. Verify Gazebo is running:
   ```bash
   rosnode list | grep gazebo
   ```

3. Move robot more than 2mm (threshold)

### Problem: Line appears in wrong place

**Check:**
- Frame ID is `'world'`
- End-effector position is correct (see `END_EFFECTOR_DEFINITION.md`)

### Problem: Clear command not working

**Check:**
- Type exactly: `clear`, `c`, `erase`, or `reset` (case-insensitive)
- No extra characters or spaces
- In manual test mode only

## Performance Notes

### Memory Usage

- Each point: ~24 bytes
- 1000 points ≈ 24 KB
- Typical episode: 100-500 points
- **Impact**: Negligible

### CPU Usage

- Marker publishing: ~0.1% CPU per update
- Line rendering: Handled by Gazebo/RViz
- **Impact**: Minimal

### Network Bandwidth

- Marker message: ~1-2 KB
- Update frequency: ~5-10 Hz (depending on robot speed)
- **Bandwidth**: ~10-20 KB/s
- **Impact**: Negligible on local machine

## Advanced Usage

### Save Trajectory for Analysis

```python
# After an episode
env.trajectory_drawer.save_trajectory('episode_5_success.npy')

# Later, load and analyze
import numpy as np
points = np.load('episode_5_success.npy')
print(f"Trajectory shape: {points.shape}")
print(f"First point: {points[0]}")
print(f"Last point: {points[-1]}")
```

### Multiple Colored Trajectories

```python
# Create multiple drawers for different purposes
exploration_drawer = TrajectoryDrawer(color='yellow')
success_drawer = TrajectoryDrawer(color='green')
failure_drawer = TrajectoryDrawer(color='red')

# Use appropriate drawer based on episode outcome
if success:
    env.trajectory_drawer = success_drawer
else:
    env.trajectory_drawer = failure_drawer
```

## Future Enhancements

Potential improvements:

- [ ] Color changes based on reward (green=good, red=bad)
- [ ] Trajectory fading over time
- [ ] Multiple trajectory layers (last 5 episodes)
- [ ] Heatmap visualization of frequently visited areas
- [ ] 3D trajectory replay at different speeds

## Files Modified

- ✅ `trajectory_drawer.py` - New file, drawing system
- ✅ `main_rl_environment_noetic.py` - Integrated trajectory tracking
- ✅ `train_robot.py` - Added clear commands and info display

## Testing

### Test the Drawing System

```bash
cd ~/rl_model_based/robot_ws/src/new_robot_arm_urdf/scripts

# Test standalone trajectory drawer
python3 trajectory_drawer.py
# Should draw a circle in RViz/Gazebo

# Test with robot
python3 train_robot.py
# Choose mode 1 (Manual Test)
# Move robot and watch the green line!
```

### Verify Clearing Works

```bash
# In manual mode:
0.5 0.5 0.5 0.5   # Move robot (line appears)
0 0 0 0           # Move back (longer line)
clear             # Line disappears!
```

## Credits

- **ROS visualization_msgs**: Line rendering
- **Marker type**: LINE_STRIP (efficient multi-segment lines)
- **Integration**: Seamless with existing RL environment

## Date Added

November 3, 2025

---

**Enjoy watching your robot draw! 🎨🤖**
