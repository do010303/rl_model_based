import time
from sensor_msgs.msg import JointState
import threading
import sys
import os
import rospy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main_rl_environment_noetic import RLEnvironmentNoetic

def wait_until_reached(env, target, tol=0.1, timeout=20, vel_thresh=0.005):
    start = time.time()
    while time.time() - start < timeout:
        state = env.get_state()
        if state is not None:
            joints = np.array(state)[3:7]
            vels = np.array(state)[7:11] if len(state) >= 11 else np.zeros(4)
            print(f"  [DEBUG] joints: {joints}, vels: {vels}")
            pos_ok = np.allclose(joints, target, atol=tol)
            vel_ok = np.all(np.abs(vels) < vel_thresh)
            if pos_ok and vel_ok:
                time.sleep(1.0)
                return True
        time.sleep(0.2)
    # After timeout, do a final check after a short sleep
    time.sleep(1.0)
    return False

def get_user_joint_angles():
    print("Nhập 4 góc khớp (đơn vị radian, cách nhau bởi dấu cách):")
    while True:
        try:
            angles = input("Joint angles (ví dụ: 0.1 0 0 0): ").strip().split()
            if len(angles) == 0:
                print("[INFO] Test cancelled by user (empty input)")
                raise KeyboardInterrupt
            if len(angles) != 4:
                print("Vui lòng nhập đúng 4 giá trị!")
                continue
            angles = [float(a) for a in angles]
            return np.array(angles)
        except KeyboardInterrupt:
            print("\n[INFO] Test cancelled by user (Ctrl+C)")
            raise
        except Exception:
            print("Giá trị không hợp lệ, thử lại!")

def wait_for_joint_states(timeout=5.0):
    """Wait for /joint_states to be published and return the first message."""
    result = {}
    def cb(msg):
        result['msg'] = msg
    sub = rospy.Subscriber('/joint_states', JointState, cb)
    start = time.time()
    while 'msg' not in result and (time.time() - start < timeout):
        rospy.sleep(0.05)
    sub.unregister()
    return result.get('msg', None)

# Global subscriber for joint states (persistent)
_joint_state_data = {'positions': None, 'velocities': None, 'timestamp': 0}
_joint_state_sub = None

def _joint_state_callback(msg):
    """Persistent callback for /joint_states - updates global data"""
    global _joint_state_data
    try:
        positions = []
        velocities = []
        for i in range(1, 5):
            joint_name = f'Joint{i}'
            if joint_name in msg.name:
                idx = msg.name.index(joint_name)
                positions.append(msg.position[idx])
                velocities.append(msg.velocity[idx] if len(msg.velocity) > idx else 0.0)
        
        if len(positions) == 4:
            _joint_state_data['positions'] = np.array(positions)
            _joint_state_data['velocities'] = np.array(velocities)
            _joint_state_data['timestamp'] = time.time()
    except Exception as e:
        rospy.logerr_throttle(1.0, f"Error parsing /joint_states: {e}")

def init_joint_state_subscriber():
    """Initialize the persistent /joint_states subscriber"""
    global _joint_state_sub
    if _joint_state_sub is None:
        _joint_state_sub = rospy.Subscriber('/joint_states', JointState, _joint_state_callback, queue_size=1)
        rospy.sleep(0.2)  # Give subscriber time to connect

def get_joint_positions_direct(timeout=0.5):
    """
    Get joint positions DIRECTLY from /joint_states using persistent subscriber.
    This is FAST because it just reads cached data from the callback.
    """
    global _joint_state_data
    
    start = time.time()
    while time.time() - start < timeout:
        if _joint_state_data['positions'] is not None:
            age = time.time() - _joint_state_data['timestamp']
            if age < 0.5:  # Data is fresh (less than 0.5s old)
                return _joint_state_data['positions'].copy(), _joint_state_data['velocities'].copy()
        rospy.sleep(0.01)
    
    # Timeout
    if _joint_state_data['positions'] is not None:
        age = time.time() - _joint_state_data['timestamp']
        rospy.logwarn(f"Using stale joint_states data (age: {age:.2f}s)")
        return _joint_state_data['positions'].copy(), _joint_state_data['velocities'].copy()
    
    return None, None

def wait_until_stopped_direct(vel_thresh=0.02, hold_time=1.0, timeout=15):
    """
    Wait until robot stops by monitoring /joint_states DIRECTLY.
    Uses persistent subscriber for speed.
    """
    print(f"[DEBUG] Waiting for robot to stop (vel < {vel_thresh} rad/s for {hold_time}s)...")
    
    start = time.time()
    last_below_time = None
    check_count = 0
    last_print = 0
    
    while time.time() - start < timeout:
        positions, velocities = get_joint_positions_direct(timeout=0.1)
        
        if velocities is not None:
            max_vel = np.max(np.abs(velocities))
            
            check_count += 1
            # Print every 0.5s instead of every 10 checks
            if time.time() - last_print >= 0.5:
                print(f"[DEBUG] t={time.time()-start:.1f}s, max_vel={max_vel:.4f} rad/s")
                last_print = time.time()
            
            if max_vel < vel_thresh:
                if last_below_time is None:
                    last_below_time = time.time()
                    print(f"[DEBUG] Velocity below threshold! Waiting {hold_time}s to confirm...")
                elif time.time() - last_below_time >= hold_time:
                    total_time = time.time() - start
                    print(f"[DEBUG] ✓ Robot stopped! (t={total_time:.1f}s, vel={max_vel:.4f} rad/s)")
                    return True
            else:
                if last_below_time is not None:
                    print(f"[DEBUG] Velocity increased to {max_vel:.4f} rad/s, resetting timer...")
                last_below_time = None
        
        rospy.sleep(0.05)
    
    # Timeout
    positions, velocities = get_joint_positions_direct(timeout=0.1)
    if velocities is not None:
        max_vel = np.max(np.abs(velocities))
        print(f"[WARN] ⚠️ Timeout after {timeout}s! Robot still at {max_vel:.4f} rad/s")
        print(f"[WARN] Robot is OSCILLATING - PID gains need tuning!")
    return False
    """Wait until all joint velocities are below threshold for hold_time seconds, using environment data."""
    start = time.time()
    last_below = None
    
    # First, wait at least for the trajectory duration (5 seconds as set in move_to_joint_positions)
    min_wait = 5.5  # Slightly more than trajectory time
    
    while time.time() - start < min_wait:
        rospy.sleep(0.1)
    
    print(f"[DEBUG] Trajectory time complete ({min_wait}s), now checking if robot has stopped...")
    
    # Now check if velocities have settled - MUST actually stop, not timeout
    check_count = 0
    while time.time() - start < timeout:
        # Use the environment's existing joint_velocities data
        if hasattr(env, 'joint_velocities') and len(env.joint_velocities) == 4:
            vels = np.array(env.joint_velocities)
            max_vel = np.max(np.abs(vels))
            
            check_count += 1
            if check_count % 10 == 0:  # Print every 10 checks (~0.5s)
                print(f"[DEBUG] Max velocity: {max_vel:.4f} rad/s (threshold: {vel_thresh})")
            
            # Check if all velocities are below threshold
            if max_vel < vel_thresh:
                if last_below is None:
                    last_below = time.time()
                    print(f"[DEBUG] Velocity below threshold, starting hold timer...")
                elif time.time() - last_below >= hold_time:
                    total_time = time.time() - start
                    print(f"[DEBUG] ✓ Robot stopped after {total_time:.2f}s (max vel: {max_vel:.4f} rad/s)")
                    return True
            else:
                if last_below is not None:
                    print(f"[DEBUG] Velocity exceeded threshold again: {max_vel:.4f} rad/s")
                last_below = None
        
        rospy.sleep(0.05)
    
    # Timeout - robot didn't stop properly
    if hasattr(env, 'joint_velocities') and len(env.joint_velocities) == 4:
        vels = np.array(env.joint_velocities)
        max_vel = np.max(np.abs(vels))
        print(f"[WARN] ⚠️ Timeout after {timeout}s, robot still moving at {max_vel:.4f} rad/s!")
    return False  # Return FALSE on timeout - this is important!

def get_latest_joint_positions_from_env(env, timeout=1.0, force_fresh=False):
    """Get the latest joint positions from the environment's existing subscriber - optimized for speed."""
    start = time.time()
    
    if force_fresh:
        # Force wait for a NEW message by checking timestamp or waiting briefly
        print(f"[DEBUG] Forcing fresh joint state read...")
        rospy.sleep(0.2)  # Wait for at least 2 new messages at 100Hz publish rate
    
    # Use the new fast method if available
    if hasattr(env, 'get_joint_positions'):
        positions = env.get_joint_positions()
        if positions is not None:
            print(f"[DEBUG] Got joint positions: {positions}")
            return positions
    
    # Fallback: direct attribute access
    for attempt in range(int(timeout / 0.02)):  # Check every 20ms
        if hasattr(env, 'joint_positions') and hasattr(env, 'data_ready'):
            if env.data_ready and len(env.joint_positions) == 4:
                current_positions = np.array(env.joint_positions)
                
                # Quick validation - just check for NaN
                if not np.any(np.isnan(current_positions)):
                    elapsed = time.time() - start
                    print(f"[DEBUG] Got joint positions: {current_positions}")
                    return current_positions
        
        rospy.sleep(0.02)  # 20ms sleep for fast checking
    
    # Fallback: return whatever we have
    if hasattr(env, 'joint_positions') and len(env.joint_positions) == 4:
        print(f"[WARN] Returning positions after timeout, may not be fully updated")
        return np.array(env.joint_positions)
    
    print(f"[ERROR] Failed to get valid joint positions after {timeout}s timeout")
    return None

def test_simple_movement():
    """Test very simple robot movements to debug error code -4."""
    print("Testing simple robot movements...")
    try:
        # Initialize ROS node
        rospy.init_node('simple_movement_test', anonymous=True)
        print("✓ ROS node initialized")

        # Initialize persistent /joint_states subscriber
        print("Setting up /joint_states subscriber...")
        init_joint_state_subscriber()
        print("✓ /joint_states subscriber ready")

        # Confirm /joint_states is being published
        print("Waiting for /joint_states topic...")
        js_msg = wait_for_joint_states(timeout=5.0)
        if js_msg is not None:
            print(f"✓ /joint_states received: joints = {js_msg.name}")
        else:
            print("✗ Did not receive /joint_states! Check controllers.")
            return False

        # Create environment
        env = RLEnvironmentNoetic()
        print("✓ Environment created")

        # Wait for initialization
        print("Waiting for environment to initialize...")
        rospy.sleep(3)

        # Reset environment
        print("Resetting environment...")
        success = env.reset_environment()
        print(f"✓ Environment reset successful: {success}")

        # Print joint limits for reference
        joint_limits_low = getattr(env, 'joint_limits_low', None)
        joint_limits_high = getattr(env, 'joint_limits_high', None)
        if joint_limits_low is not None and joint_limits_high is not None:
            print("Joint limits (radian):")
            for i, (lo, hi) in enumerate(zip(joint_limits_low, joint_limits_high)):
                print(f"  Joint {i+1}: [{lo}, {hi}]")
        else:
            print("(Could not read joint limits from environment)")

        print("\nTest nhập góc thủ công cho robot (Ctrl+C để thoát)...")
        while True:
            try:
                movement = get_user_joint_angles()
                print(f"\n{'='*60}")
                print(f"Testing Movement to: {movement}")
                print(f"{'='*60}")

                # Send action to move robot
                print(f"\n[1/3] Sending trajectory goal to robot...")
                result = env.move_to_joint_positions(movement)
                if not result['success']:
                    print(f"      ❌ Action failed with error code: {result['error_code']}")
                elif result['error_code'] == -5:
                    print(f"      ✓ Trajectory sent (error -5 is acceptable)")
                else:
                    print(f"      ✓ Trajectory sent successfully")

                # Wait for trajectory to complete (3s trajectory + 0.5s buffer)
                print(f"\n[2/3] Waiting for trajectory to complete...")
                trajectory_time = 3.0  # Must match time in move_to_joint_positions
                buffer_time = 0.5
                total_wait = trajectory_time + buffer_time
                
                print(f"      Waiting {total_wait}s (trajectory: {trajectory_time}s + buffer: {buffer_time}s)")
                rospy.sleep(total_wait)
                print(f"      ✓ Trajectory complete")
                
                # Read final position DIRECTLY from /joint_states
                print(f"\n[3/3] Reading final position from /joint_states...")
                final_joint_positions, final_velocities = get_joint_positions_direct(timeout=1.0)
                
                if final_joint_positions is None:
                    print(f"      ❌ ERROR: Could not read final joint positions!")
                    print(f"      Check that /joint_states is being published.")
                    if hasattr(env, 'joint_positions'):
                        print(f"      Environment has joint_positions: {env.joint_positions}")
                    return False
                
                # Calculate errors
                error = np.abs(final_joint_positions - movement)
                max_error = np.max(error)
                max_vel = np.max(np.abs(final_velocities)) if final_velocities is not None else 0.0
                tolerance = 0.1  # 0.1 radian = ~5.7 degrees
                
                # Display results
                print(f"\n{'='*60}")
                print(f"RESULTS:")
                print(f"{'='*60}")
                print(f"Target positions:  {movement}")
                print(f"Final positions:   {final_joint_positions}")
                print(f"Position errors:   {error}")
                if max_vel > 0.02:
                    print(f"Final velocities:  {final_velocities} (STILL MOVING!)")
                else:
                    print(f"Final velocities:  {final_velocities}")
                print(f"Max velocity:      {max_vel:.4f} rad/s")
                print(f"Max error:         {max_error:.4f} rad = {np.degrees(max_error):.2f}°")
                print(f"Tolerance:         {tolerance} rad = {np.degrees(tolerance):.2f}°")
                
                # Check tolerance
                if np.allclose(final_joint_positions, movement, atol=tolerance):
                    print(f"\n✅ SUCCESS! Robot reached target within tolerance")
                    if max_vel > 0.05:
                        print(f"⚠️  NOTE: Robot still oscillating at {max_vel:.4f} rad/s")
                        print(f"   This is normal with current PID gains")
                        print(f"   Position is acceptable for RL training")
                else:
                    print(f"\n❌ FAILED! Position error exceeds tolerance")
                    print(f"   Possible causes:")
                    print(f"   - Trajectory time too short (try increasing from 3s to 5s)")
                    print(f"   - Joint limits violated")
                    print(f"   - Controller malfunction")
                
                print(f"{'='*60}\n")
            except KeyboardInterrupt:
                print("\n[INFO] Test cancelled by user (Ctrl+C)")
                break
            except Exception as e:
                print(f"✗ Test failed with error: {e}")
                import traceback
                traceback.print_exc()
                return False
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    return True

if __name__ == "__main__":
    test_simple_movement()
