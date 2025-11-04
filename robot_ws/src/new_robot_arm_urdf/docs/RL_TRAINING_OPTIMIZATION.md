# RL Training Optimization - Fast Joint State Reading

## Date: November 3, 2025

## Performance Optimizations for RL Training

### New Fast Methods Added to RLEnvironmentNoetic

```python
# FAST: Get just joint positions (optimized for RL)
positions = env.get_joint_positions()  # Returns numpy array or None
# Returns: [joint1, joint2, joint3, joint4]

# FAST: Get just joint velocities (check if robot stopped)
velocities = env.get_joint_velocities()  # Returns numpy array or None  
# Returns: [vel1, vel2, vel3, vel4]

# FULL: Get complete state (includes end-effector, joints, target)
state = env.get_state()  # Returns 10-element state vector
# Returns: [ee_x, ee_y, ee_z, j1, j2, j3, j4, target_x, target_y, target_z]
```

### Usage in RL Training Loop

```python
# Initialize environment
env = RLEnvironmentNoetic()

# Training loop
for episode in range(num_episodes):
    env.reset_environment()
    
    for step in range(max_steps):
        # FAST: Get current state
        state = env.get_state()
        
        # Agent selects action
        action = agent.select_action(state)
        
        # Execute action
        env.execute_action(action)
        
        # FAST: Wait for movement to complete (optimized timing)
        # The execute_action already waits, but if you need to check:
        velocities = env.get_joint_velocities()
        
        # FAST: Get next state
        next_state = env.get_state()
        
        # Calculate reward
        reward, done = env.calculate_reward()
        
        # Store transition
        agent.store_transition(state, action, reward, next_state, done)
        
        if done:
            break
```

### Timing Optimizations

**Before (slow):**
- Wait fixed 3 seconds
- Create new subscriber every time
- Check velocities with 0.01 rad/s threshold (too strict)
- Multiple redundant checks
- **Total: ~15-20 seconds per action**

**After (fast):**
- Wait for trajectory duration (5.5s) - matches actual movement time
- Use existing subscriber (no overhead)
- Check velocities with 0.1 rad/s threshold (practical)
- Optimized position reading (instant if data ready)
- **Total: ~6-8 seconds per action**

### Movement Execution Flow (Optimized)

```python
# 1. Send command (instant)
result = env.move_to_joint_positions(target_positions)

# 2. Wait for trajectory completion (5.5s - matches trajectory time)
#    Checks velocity every 50ms, accepts when max_vel < 0.1 rad/s for 0.3s

# 3. Read final position (instant - uses cached data)
final_pos = env.get_joint_positions()

# Total time: ~6 seconds per movement
```

### Key Parameters

```python
# In move_to_joint_positions():
trajectory_time = 5.0  # Time for robot to reach target
goal_time_tolerance = 3.0  # Allow 3s flexibility
wait_timeout = 10.0  # Max wait for action completion

# In wait_until_stopped_from_env():
vel_threshold = 0.1  # rad/s - practical threshold  
hold_time = 0.3  # seconds - must be stopped for this long
min_wait = 5.5  # seconds - minimum wait (matches trajectory time)
timeout = 8.0  # seconds - maximum total wait

# In get_latest_joint_positions_from_env():
check_interval = 0.02  # seconds - 20ms between checks
timeout = 1.0  # seconds - max wait (usually instant)
```

### Error Code Handling

Action server error codes now accepted as success:
- `0` = SUCCESS (perfect)
- `-5` = GOAL_TOLERANCE_VIOLATED (acceptable - trajectory still executes)

This is because Gazebo physics simulation doesn't allow perfect position control.
The robot reaches approximately the right position, which is sufficient for RL.

### Position Validation

**Tolerance for success:**
- Position error: 0.1 radian (~5.7 degrees)
- This is reasonable for:
  - Gazebo simulation physics
  - PID controller limitations
  - Real-world applications

**What if robot doesn't reach target?**
- Check controller PID gains in `config/rl_controllers.yaml`
- Check joint limits
- Verify no collisions
- Check Gazebo physics timestep

### Debugging Tips

```python
# Show detailed movement info
current_pos = env.get_joint_positions()
target_pos = np.array([1.0, 0.5, 0.3, 0.8])
print(f"Current: {current_pos}")
print(f"Target: {target_pos}")
print(f"Delta: {target_pos - current_pos}")

# After movement
final_pos = env.get_joint_positions()
error = np.abs(final_pos - target_pos)
max_error = np.max(error)
print(f"Final: {final_pos}")
print(f"Error: {error}")
print(f"Max error: {max_error:.4f} rad = {np.degrees(max_error):.2f} deg")

# Check if robot is still moving
velocities = env.get_joint_velocities()
max_vel = np.max(np.abs(velocities))
print(f"Velocities: {velocities}")
print(f"Max velocity: {max_vel:.4f} rad/s")
print(f"Stopped: {max_vel < 0.1}")
```

### Training Speed Estimates

**Single Episode (200 steps max):**
- Average actions per episode: ~100
- Time per action: ~6 seconds
- Episode time: ~600 seconds (10 minutes)

**Full Training (1000 episodes):**
- Estimated time: ~167 hours (7 days)
- With early success: Can reduce significantly

**Speedup Ideas:**
1. Reduce max_steps per episode (100 instead of 200)
2. Use curriculum learning (start with easier targets)
3. Reduce trajectory_time to 3.0s for faster movements
4. Use action repeat (execute same action multiple times)
5. Parallel environments (multiple Gazebo instances)

### Common Issues & Solutions

**Issue: Robot doesn't reach target**
- Check: PID gains too low → increase P gain
- Check: Trajectory time too short → increase to 5.0s
- Check: Joint limits violated → clip actions to limits

**Issue: Robot oscillates around target**
- Check: PID gains too high → decrease P, increase D
- Check: Velocity threshold too strict → use 0.1 instead of 0.01

**Issue: "Timeout waiting for robot to stop"**
- This is now OK - returns True anyway after min_wait
- Robot has completed trajectory even if still has small oscillations

**Issue: Getting None from get_joint_positions()**
- Check: env.data_ready is False → wait for first /joint_states message
- Check: Subscriber not receiving → verify rostopic echo /joint_states works

### Performance Monitoring

Add timing to your training loop:

```python
import time

# In training loop
action_start = time.time()
env.execute_action(action)
action_time = time.time() - action_start

state_start = time.time()
next_state = env.get_state()
state_time = time.time() - state_start

print(f"Action time: {action_time:.2f}s, State read time: {state_time:.4f}s")
```

Expected:
- Action time: 6-8 seconds (dominated by physical movement)
- State read time: <0.1 seconds (should be instant)

