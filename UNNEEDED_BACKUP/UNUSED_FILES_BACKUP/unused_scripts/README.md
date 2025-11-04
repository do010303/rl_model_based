# Unused Scripts Archive

This folder contains older/obsolete scripts that have been replaced by newer implementations but are kept for reference.

## Files in this folder:

### `ddpg_4dof_noetic.py`
- **Status**: Replaced
- **Replaced by**: `/home/ducanh/rl_model_based/agents/ddpg_gazebo.py`
- **Reason**: Standalone DDPG implementation, now using centralized agent in agents/ folder

### `rl_environment_noetic.py`
- **Status**: Replaced
- **Replaced by**: `main_rl_environment_noetic.py`
- **Reason**: Old environment implementation, replaced by enhanced version with FK, trajectory drawing, and better state tracking

### `robot_control_interface.py`
- **Status**: Obsolete
- **Reason**: Functionality integrated into `main_rl_environment_noetic.py`

### `training_interface.py`
- **Status**: Obsolete
- **Reason**: Visual training interface, functionality now in `main_rl_environment_noetic.py` and `trajectory_drawer.py`

### `test_environment.py`
- **Status**: Obsolete
- **Replaced by**: `test_simple_movement.py`
- **Reason**: Broken import paths, replaced by working test script

---

## Active Scripts (in parent directory):

These are the ONLY scripts you should use for training:

1. **`train_robot.py`** - Main training script with manual test mode
2. **`main_rl_environment_noetic.py`** - Complete RL environment with all features
3. **`test_simple_movement.py`** - Movement validation and testing
4. **`fk_ik_utils.py`** - Forward/Inverse kinematics utilities
5. **`trajectory_drawer.py`** - Visual trajectory drawing system

---

**Date archived**: November 4, 2025  
**Note**: These files can be safely deleted if disk space is needed, but kept for reference/comparison.
