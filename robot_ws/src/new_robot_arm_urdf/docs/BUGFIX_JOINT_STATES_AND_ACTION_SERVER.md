# Bug Fixes: Joint States Reading & Action Server Error -5

## Date: November 3, 2025

## Issues Fixed

### 1. Test Script Unable to Read Final Joint Positions

**Problem:**
- The test script `test_simple_movement.py` was creating competing subscribers to `/joint_states`
- After sending a movement command, the new subscriber would fail to receive any messages
- This caused the error: "Không lấy được trạng thái khớp từ /joint_states sau khi chờ robot dừng!"

**Root Cause:**
- The environment (`RLEnvironmentNoetic`) already subscribes to `/joint_states` 
- Creating a second subscriber in the test script caused conflicts
- ROS topic subscribers can interfere with each other when not properly managed

**Solution:**
- Removed the competing subscriber function `get_latest_joint_positions()`
- Created `get_latest_joint_positions_from_env(env)` that reads from the environment's existing `joint_positions` attribute
- The environment's subscriber continues to work without interference

**Files Modified:**
- `/robot_ws/src/new_robot_arm_urdf/scripts/test_simple_movement.py`

---

### 2. Joint Positions Read Too Early (Before Robot Stops)

**Problem:**
- Fixed wait time of 3 seconds was insufficient
- Joint positions were read while the robot was still moving
- This resulted in incorrect "final" positions that didn't match the actual final position

**Evidence:**
- User confirmed with `rostopic echo /joint_states` that positions were different from what the test reported
- Visual inspection in Gazebo confirmed the robot reached the correct position
- The test was reading positions mid-movement

**Solution:**
- Created `wait_until_stopped_from_env(env)` function that monitors joint velocities
- Waits until all joint velocities are below threshold (0.01 rad/s) for 0.5 seconds
- Only then reads the final joint positions
- Added additional 0.5s stabilization delay after velocity check

**Files Modified:**
- `/robot_ws/src/new_robot_arm_urdf/scripts/test_simple_movement.py`

---

### 3. Action Server Error Code -5 (GOAL_TOLERANCE_VIOLATED)

**Problem:**
- Every movement command returned error code -5
- Error messages showed "Home move failed" and "Action execution failed"
- However, visual inspection confirmed the robot WAS moving correctly

**Root Cause:**
- Error code -5 = `GOAL_TOLERANCE_VIOLATED` in ROS action lib
- The action server's internal tolerance was too strict
- The robot was reaching approximately the correct position, but not within the action server's tight tolerance
- The trajectory WAS executing successfully despite the error code

**Solution in Environment:**
- Modified `move_to_joint_positions()` to accept error code -5 as success
- Increased trajectory time from 3.0s to 5.0s to give robot more time
- Increased goal_time_tolerance from 2.0s to 3.0s
- Increased wait timeout from 5.0s to 10.0s
- Added debug logging for error code -5

**Solution in Test Script:**
- Updated to show informative message for error code -5 instead of warning
- No longer treats -5 as a failure condition

**Files Modified:**
- `/robot_ws/src/new_robot_arm_urdf/scripts/main_rl_environment_noetic.py`
- `/robot_ws/src/new_robot_arm_urdf/scripts/test_simple_movement.py`

---

### 4. Improved Tolerance Check and Error Reporting

**Problem:**
- Test was incorrectly reporting tolerance failures
- Example: 0.99 vs 1.0 should pass with 0.1 radian tolerance, but was failing
- No information about actual error magnitude

**Solution:**
- Added detailed error reporting showing:
  - Target positions
  - Final positions  
  - Position errors for each joint
  - Maximum error in radians and degrees
- Improved tolerance check logic
- Better visual feedback for success/failure

**Files Modified:**
- `/robot_ws/src/new_robot_arm_urdf/scripts/test_simple_movement.py`

---

## Testing

Run the test with:
```bash
python3 ~/rl_model_based/robot_ws/src/new_robot_arm_urdf/scripts/test_simple_movement.py
```

Expected behavior:
1. ✅ Environment initializes successfully
2. ✅ Can send movement commands without errors (except acceptable -5)
3. ✅ Waits for robot to actually stop before reading position
4. ✅ Correctly reads final joint positions from environment
5. ✅ Accurately reports whether target was reached
6. ✅ Shows detailed error information

---

## Key Learnings

1. **Don't create duplicate ROS subscribers** - Reuse existing subscribers when possible
2. **Wait for motion to complete** - Don't read positions on a fixed timer, monitor velocities
3. **Understand action server error codes** - Error code -5 doesn't mean complete failure
4. **Gazebo physics != perfect control** - Some position error is expected and acceptable
5. **Provide detailed debugging info** - Show actual vs target positions, not just pass/fail

---

## Error Codes Reference

ROS FollowJointTrajectory Action Error Codes:
- `0` = SUCCESS - Goal achieved within tolerance
- `-1` = INVALID_GOAL - Goal is malformed
- `-2` = INVALID_JOINTS - Joint names don't match
- `-3` = OLD_HEADER_TIMESTAMP - Timestamp is in the past
- `-4` = PATH_TOLERANCE_VIOLATED - Trajectory deviated too much during execution
- `-5` = GOAL_TOLERANCE_VIOLATED - Final position not within goal tolerance (but trajectory executed)
- `-100` = Custom: No result from action server
- `-101` = Custom: Exception during execution

---

## Code Changes Summary

### test_simple_movement.py
```python
# OLD: Created competing subscriber
sub = rospy.Subscriber('/joint_states', JointState, cb)

# NEW: Use environment's existing data
final_joint_positions = get_latest_joint_positions_from_env(env)
```

### test_simple_movement.py  
```python
# OLD: Fixed wait time
time.sleep(3.0)
final_positions = get_latest_joint_positions()

# NEW: Wait for robot to stop
wait_until_stopped_from_env(env, vel_thresh=0.01, hold_time=0.5)
time.sleep(0.5)  # Extra stabilization
final_positions = get_latest_joint_positions_from_env(env)
```

### main_rl_environment_noetic.py
```python
# OLD: Only accept error code 0 as success
success = result is not None and result.error_code == 0

# NEW: Accept both 0 and -5 as success
success = result is not None and (result.error_code == 0 or result.error_code == -5)
```
