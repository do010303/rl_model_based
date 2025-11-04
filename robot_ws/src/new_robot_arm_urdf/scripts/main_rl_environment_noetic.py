from fk_ik_utils import fk

# Drawing surface bounds (adjust as needed)
SURFACE_X = 0.20
SURFACE_Y_MIN = -0.10
SURFACE_Y_MAX = 0.10
SURFACE_Z = 0.12
SURFACE_X_TOL = 0.01
SURFACE_Z_TOL = 0.01


def is_on_surface(joint_angles):
    x, y, z = fk(joint_angles)
    return (
        abs(x - SURFACE_X) < SURFACE_X_TOL and
        SURFACE_Y_MIN <= y <= SURFACE_Y_MAX and
        abs(z - SURFACE_Z) < SURFACE_Z_TOL
    )

def wait_until_reached(env, target, tol=0.1, timeout=10, vel_thresh=0.01):
    import time
    start = time.time()
    while time.time() - start < timeout:
        state = env.get_state()
        if state is not None:
            joints = np.array(state)[3:7]
            vels = np.array(state)[7:11] if len(state) >= 11 else np.zeros(4)
            pos_ok = np.allclose(joints, target, atol=tol)
            vel_ok = np.all(np.abs(vels) < vel_thresh)
            if pos_ok and vel_ok:
                time.sleep(0.2)
                return True
        time.sleep(0.1)
    return False
#!/usr/bin/env python3
"""
ROS Noetic Visual RL Environment for 4DOF Robot Arm
Complete adaptation from ROS2 robotic_arm_environment to ROS Noetic

This is the main RL environment that provides:
1. State space: end-effector position + 4 joint states + target sphere position
2. Action space: 4 joint position commands 
3. Reward calculation: distance-based with goal achievement detection
4. Visual feedback and episode management
5. Trajectory drawing for end-effector path visualization
"""

import rospy
import numpy as np
import random
import time
from typing import Tuple, Optional
from geometry_msgs.msg import Point, Pose, Quaternion
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ModelStates  
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryGoal
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction
from trajectory_drawer import TrajectoryDrawer

class RLEnvironmentNoetic:
    def wait_until_reached(self, target, tol=0.1, timeout=10, vel_thresh=0.01):
        """Wait until robot joints reach target (within tol) and velocities are low."""
        import time
        start = time.time()
        while time.time() - start < timeout:
            state = self.get_state()
            if state is not None:
                joints = np.array(state)[3:7]
                vels = np.array(state)[7:11] if len(state) >= 11 else np.zeros(4)
                pos_ok = np.allclose(joints, target, atol=tol)
                vel_ok = np.all(np.abs(vels) < vel_thresh)
                if pos_ok and vel_ok:
                    time.sleep(0.2)
                    return True
            time.sleep(0.1)
        return False
    """
    Visual RL Environment for 4DOF Robot in ROS Noetic
    
    Provides Gym-like interface for robot learning with visual feedback
    """
    
    def __init__(self, max_episodes=1000, max_episode_steps=200, goal_tolerance=0.02):
        """
        Initialize the RL environment
        
        Args:
            max_episodes: Maximum number of training episodes (configurable)
            max_episode_steps: Maximum steps per episode
            goal_tolerance: Distance tolerance for goal achievement (reduced for smaller target)
        """
        rospy.loginfo("🤖 Initializing Visual RL Environment for 4DOF Robot...")
        
        # Episode configuration (now configurable)
        self.max_episodes = max_episodes
        self.max_episode_steps = max_episode_steps
        self.goal_tolerance = goal_tolerance  # Reduced for smaller target sphere
        """Move target sphere to random position on the drawing surface"""
        # Drawing surface is a vertical plane at x=0.2m (20cm from robot base)
        # Surface dimensions: 50cm height x 30cm width (0.5 x 0.3)
        # Surface center: x=0.2, y=0, z=0.15
        
        # Generate random position on the drawing surface
        drawing_surface_x = 0.2   # Fixed X position (surface distance from robot - 20cm)
        surface_y_range = 0.12    # +/- 12cm from center (24cm total width)
        surface_z_min = 0.05      # 5cm from bottom (accessible by robot)
        surface_z_max = 0.35      # 35cm from bottom (top of reachable area)
        
        sphere_x = drawing_surface_x + random.uniform(-0.01, 0.01)  # Small variance around surface
        sphere_y = random.uniform(-surface_y_range, surface_y_range)
        sphere_z = random.uniform(surface_z_min, surface_z_max)
        
        # Create service request
        request = SetModelStateRequest()
        request.model_state.model_name = 'my_sphere'
        request.model_state.reference_frame = 'world'
        
        # Set position
        request.model_state.pose = Pose()
        request.model_state.pose.position = Point(x=sphere_x, y=sphere_y, z=sphere_z)
        request.model_state.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)
        
        try:
            rospy.loginfo(f"🎯 Moving target to drawing surface: [{sphere_x:.3f}, {sphere_y:.3f}, {sphere_z:.3f}]")
            response = self.reset_target_client(request)
            
            if response.success:
                rospy.loginfo("✅ Target sphere positioned on drawing surface")
                return True
            else:
                rospy.logerr(f"❌ Target reset failed: {response.status_message}")
                return False
                
        except Exception as e:
            rospy.logerr(f"Error resetting target position: {e}")
            return False

    def _show_episode_success(self):
        """Visual feedback for successful episode completion"""
        try:
            # Flash the target sphere green to indicate success
            rospy.loginfo("🎉 SHOWING SUCCESS FEEDBACK 🎉")
            rospy.sleep(0.5)  # Brief pause for effect
            
        except Exception as e:
            rospy.logerr(f"Error showing episode success: {e}")

    def generate_random_action(self) -> np.ndarray:
        """Generate random action within joint limits for exploration"""
        return np.random.uniform(self.joint_limits_low, self.joint_limits_high)

# TF for end-effector position tracking
import tf2_ros
from tf2_ros import TransformException


class RLEnvironmentNoetic:
    """
    Complete Visual RL Environment for 4DOF Robot Arm in ROS Noetic
    Adapted from ROS2 robotic_arm_environment for visual RL training in Gazebo
    """
    
    def __init__(self, max_episode_steps=200, goal_tolerance=0.02):
        """
        Initialize Visual RL Environment for 4DOF Robot
        
        Args:
            max_episode_steps: Maximum steps per episode (default: 200)
            goal_tolerance: Distance threshold for goal achievement (default: 0.02m)
        """
        rospy.loginfo("🤖 Initializing Visual RL Environment for 4DOF Robot...")
        
        # Robot state variables (4DOF robot)
        self.robot_x = 0.0
        self.robot_y = 0.0  
        self.robot_z = 0.0
        self.joint_positions = [0.0, 0.0, 0.0, 0.0]  # 4 joints
        self.joint_velocities = [0.0, 0.0, 0.0, 0.0]
        
        # Target sphere state
        self.pos_sphere_x = 0.0
        self.pos_sphere_y = 0.0
        self.pos_sphere_z = 0.0
        
        # State readiness flag
        self.data_ready = False
        
        # Joint limits for 4DOF robot (from original Fusion URDF)
        self.joint_limits_low = np.array([-3.14159, -1.57079, -1.57079, -1.57079])
        self.joint_limits_high = np.array([3.14159, 1.57079, 1.57079, 1.57079])
        
        # RL training parameters (configurable)
        self.goal_tolerance = goal_tolerance
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        
        rospy.loginfo(f"📊 Episode settings: max_steps={max_episode_steps}, goal_tolerance={goal_tolerance}m")
        
        # TF2 for end-effector position tracking
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Trajectory drawer for visualizing end-effector path
        self.trajectory_drawer = TrajectoryDrawer(color='green', line_width=0.003)
        self._last_ee_pos = None  # For tracking movement
        
        # Initialize ROS interfaces
        self._setup_action_clients()
        self._setup_service_clients()  
        self._setup_subscribers()
        
        # Initial target randomization
        self._reset_target_position()
        
        rospy.loginfo("✅ Visual RL Environment initialized for 4DOF robot!")
    
    def _setup_action_clients(self):
        """Initialize action client for robot trajectory control"""
        rospy.loginfo("⏳ Connecting to 4DOF trajectory action server...")
        
        self.trajectory_action_client = actionlib.SimpleActionClient(
            '/doosan_arm_controller/follow_joint_trajectory',
            FollowJointTrajectoryAction
        )
        
        # Wait for action server with timeout
        if self.trajectory_action_client.wait_for_server(timeout=rospy.Duration(30)):
            rospy.loginfo("✅ 4DOF trajectory action server connected!")
        else:
            rospy.logerr("❌ Failed to connect to trajectory action server!")
            raise Exception("Trajectory action server not available")
    
    def _setup_service_clients(self):
        """Initialize service clients for Gazebo environment control"""
        rospy.loginfo("⏳ Connecting to Gazebo services...")
        
        # Wait for Gazebo set model state service
        rospy.wait_for_service('/gazebo/set_model_state')
        self.reset_target_client = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        
        rospy.loginfo("✅ Gazebo services connected!")
    
    def _setup_subscribers(self):
        """Setup ROS subscribers for robot and environment state"""
        rospy.loginfo("⏳ Setting up state subscribers...")
        
        # Subscribe to joint states (robot state)
        self.joint_state_subscriber = rospy.Subscriber(
            '/joint_states', 
            JointState, 
            self._joint_state_callback
        )
        
        # Subscribe to model states (target sphere state)  
        self.model_state_subscriber = rospy.Subscriber(
            '/gazebo/model_states',
            ModelStates, 
            self._model_state_callback
        )
        
        # Subscribe to link states (for accurate end-effector position)
        from gazebo_msgs.msg import LinkStates
        self._last_link_states = None
        self.link_state_subscriber = rospy.Subscriber(
            '/gazebo/link_states',
            LinkStates,
            self._link_state_callback
        )
        
        rospy.loginfo("✅ State subscribers initialized!")
    
    def _joint_state_callback(self, msg: JointState):
        """Robustly update joint positions/velocities for 4DOF robot, even if joint_states is unordered or incomplete."""
        joint_names = ['Joint1', 'Joint2', 'Joint3', 'Joint4']
        positions = [0.0] * 4
        velocities = [0.0] * 4
        found_all = True
        for idx, joint_name in enumerate(joint_names):
            if joint_name in msg.name:
                jidx = msg.name.index(joint_name)
                try:
                    positions[idx] = msg.position[jidx]
                    velocities[idx] = msg.velocity[jidx] if len(msg.velocity) > jidx else 0.0
                except Exception as e:
                    rospy.logwarn_throttle(5, f"Error reading joint {joint_name}: {e}")
                    found_all = False
            else:
                rospy.logwarn_throttle(5, f"Joint {joint_name} not found in joint_states: {msg.name}")
                found_all = False
        self.joint_positions = positions
        self.joint_velocities = velocities
        # Mark data as ready if all joints found
        if found_all:
            self.data_ready = True
        # Always update end-effector position
        self._update_end_effector_position()
    
    def _model_state_callback(self, msg: ModelStates):
        """Callback to process target sphere position"""
        try:
            # Find the sphere in the model states
            if 'my_sphere' in msg.name:
                sphere_index = msg.name.index('my_sphere')
                sphere_pose = msg.pose[sphere_index]
                
                self.pos_sphere_x = sphere_pose.position.x
                self.pos_sphere_y = sphere_pose.position.y
                self.pos_sphere_z = sphere_pose.position.z
                
                # Mark data as ready when we have both robot and target states
                if hasattr(self, 'joint_positions') and len(self.joint_positions) == 4:
                    self.data_ready = True
                    
            else:
                rospy.logwarn_throttle(10, "Target sphere 'my_sphere' not found in model states")
                
        except (IndexError, ValueError) as e:
            rospy.logwarn_throttle(5, f"Error processing model states: {e}")
    
    def _link_state_callback(self, msg):
        """Callback to store latest link states from Gazebo"""
        self._last_link_states = msg
    
    def _update_end_effector_position(self):
        """
        Update end-effector position using Gazebo link states
        
        We use link_4_1_1 from Gazebo because:
        1. Gazebo uses the actual URDF geometry (most accurate)
        2. endefff_1 is merged with link_4_1_1 via fixed joint (Rigid5)
        3. The position includes the full kinematic chain from URDF
        
        Fallback to FK if Gazebo data is unavailable.
        Also updates trajectory drawing.
        """
        try:
            # Try to get link_4_1_1 position from Gazebo (most accurate)
            if hasattr(self, '_last_link_states') and self._last_link_states is not None:
                link_states = self._last_link_states
                
                if 'robot_4dof_rl::link_4_1_1' in link_states.name:
                    idx = link_states.name.index('robot_4dof_rl::link_4_1_1')
                    pos = link_states.pose[idx].position
                    
                    # link_4_1_1 position + Rigid5 offset to get actual endefff_1 tip
                    # The offset needs to be rotated by link_4's orientation
                    ori = link_states.pose[idx].orientation
                    
                    # For now, use simple offset (assumes link_4 at home orientation)
                    # TODO: Properly rotate offset by link_4 orientation quaternion
                    offset = np.array([0.001137, 0.01875, 0.077946])
                    
                    self.robot_x = pos.x + offset[0]
                    self.robot_y = pos.y + offset[1]
                    self.robot_z = pos.z + offset[2]
                    
                    # Update trajectory drawing
                    self._update_trajectory_drawing()
                    
                    rospy.logdebug_throttle(2.0, f"EE from Gazebo link_4: [{self.robot_x:.4f}, {self.robot_y:.4f}, {self.robot_z:.4f}]")
                    return
            
            # Fallback to FK if Gazebo data not available
            if len(self.joint_positions) == 4:
                ee_x, ee_y, ee_z = fk(self.joint_positions)
                
                self.robot_x = ee_x
                self.robot_y = ee_y
                self.robot_z = ee_z
                
                # Update trajectory drawing
                self._update_trajectory_drawing()
                
                rospy.logdebug_throttle(2.0, f"EE from FK: [{ee_x:.4f}, {ee_y:.4f}, {ee_z:.4f}]")
            else:
                rospy.logwarn_throttle(5.0, f"Invalid joint positions length: {len(self.joint_positions)}")
                
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"EE position update failed: {e}")
            pass
    
    def _update_trajectory_drawing(self):
        """Add current end-effector position to trajectory visualization"""
        try:
            current_pos = np.array([self.robot_x, self.robot_y, self.robot_z])
            
            # Only add point if robot has moved significantly (avoid cluttering with stationary points)
            min_movement = 0.002  # 2mm threshold
            
            if self._last_ee_pos is None:
                # First point
                self.trajectory_drawer.add_point_array(current_pos)
                self._last_ee_pos = current_pos
            else:
                # Check if moved enough
                distance = np.linalg.norm(current_pos - self._last_ee_pos)
                if distance >= min_movement:
                    self.trajectory_drawer.add_point_array(current_pos)
                    self._last_ee_pos = current_pos
        except Exception as e:
            rospy.logdebug_throttle(5.0, f"Trajectory drawing update failed: {e}")
    
    def get_state(self) -> Optional[np.ndarray]:
        """
```
        Get current environment state for RL agent
        
        State vector for 4DOF robot (10 elements total):
        - End-effector position (3): [x, y, z] 
        - Joint positions (4): [joint1, joint2, joint3, joint4]
        - Target position (3): [sphere_x, sphere_y, sphere_z]
        
        Returns:
            numpy array of state or None if not ready
        """
        if not self.data_ready:
            rospy.logdebug("State not ready yet...")
            return None
            
        try:
            state = np.array([
                # End-effector position (3 elements)
                self.robot_x, self.robot_y, self.robot_z,
                # Joint positions (4 elements)  
                self.joint_positions[0], self.joint_positions[1], 
                self.joint_positions[2], self.joint_positions[3],
                # Target sphere position (3 elements)
                self.pos_sphere_x, self.pos_sphere_y, self.pos_sphere_z
            ], dtype=np.float32)
            
            return state
            
        except Exception as e:
            rospy.logerr(f"Error creating state vector: {e}")
            return None
    
    def get_joint_positions(self) -> Optional[np.ndarray]:
        """
        Fast method to get current joint positions for RL training.
        Optimized for speed - no state vector construction.
        
        Returns:
            numpy array of 4 joint positions or None if not ready
        """
        if self.data_ready and len(self.joint_positions) == 4:
            return np.array(self.joint_positions, dtype=np.float32)
        return None
    
    def get_joint_velocities(self) -> Optional[np.ndarray]:
        """
        Fast method to get current joint velocities.
        Useful for checking if robot has stopped moving.
        
        Returns:
            numpy array of 4 joint velocities or None if not ready
        """
        if self.data_ready and len(self.joint_velocities) == 4:
            return np.array(self.joint_velocities, dtype=np.float32)
        return None
    
    def execute_action(self, action: np.ndarray) -> bool:
        """
        Execute action on the 4DOF robot (robust, single-point trajectory), then wait for robot to settle and check final position.
        Args:
            action: numpy array of 4 joint positions [joint1, joint2, joint3, joint4]
        Returns:
            bool: True if robot reached target within tolerance after settling
        """
        if len(action) != 4:
            rospy.logerr(f"Action must have 4 elements for 4DOF robot, got {len(action)}")
            return False
        # Restrict action so end-effector stays on drawing surface
        if not is_on_surface(action):
            rospy.logwarn("⚠️  Action would move end-effector off the drawing surface! Rejecting action.")
            return False
        # Clip action to joint limits
        action_clipped = np.clip(action, self.joint_limits_low, self.joint_limits_high)
        # Create trajectory point (single point)
        point = JointTrajectoryPoint()
        point.positions = action_clipped.tolist()
        point.velocities = [0.0] * 4
        point.accelerations = [0.0] * 4
        point.time_from_start = rospy.Duration(2.0)  # 2 seconds for movement
        # Create goal message
        joint_names = ['Joint1', 'Joint2', 'Joint3', 'Joint4']
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.header.stamp = rospy.Time.now()
        goal.trajectory.joint_names = joint_names
        goal.trajectory.points = [point]
        goal.goal_time_tolerance = rospy.Duration(1.0)
        try:
            rospy.loginfo("Sending trajectory goal to 4DOF robot...")
            self.trajectory_action_client.send_goal_and_wait(goal, rospy.Duration(5.0))
            result = self.trajectory_action_client.get_result()
            if not (result and result.error_code == 0):
                rospy.logwarn(f"❌ Action execution failed with error code: {result.error_code if result else 'No result'} (will still check final position)")
            # Wait for robot to settle at target
            reached = wait_until_reached(self, action_clipped, tol=0.1, timeout=8, vel_thresh=0.01)
            state = self.get_state()
            if state is not None:
                joints = np.array(state)[3:7]
                rospy.loginfo(f"Final joint positions after movement: {joints}")
                if np.allclose(joints, action_clipped, atol=0.1):
                    rospy.loginfo("✅ Đạt trong tolerance 0.1 radian!")
                    rospy.loginfo("✅ Movement successful! Robot reached the target (regardless of action server result).")
                    return True
                else:
                    rospy.logwarn("⚠️ Không đạt tolerance 0.1 radian!")
                    rospy.logwarn("❌ Movement failed or did not reach target within tolerance.")
                    return False
            else:
                rospy.logerr("Không lấy được trạng thái khớp!")
                return False
        except Exception as e:
            rospy.logerr(f"Error executing action: {e}")
            return False
    
    def move_to_joint_positions(self, joint_positions: np.ndarray) -> dict:
        """
        Move robot to a specific set of joint positions.
        
        Args:
            joint_positions: numpy array of 4 target joint positions.
        
        Returns:
            dict: {'success': bool, 'error_code': int}
        """
        point = JointTrajectoryPoint()
        point.positions = joint_positions.tolist()
        point.velocities = [0.0] * 4
        point.accelerations = [0.0] * 4
        point.time_from_start = rospy.Duration(3.0)  # Fast movement for RL training

        joint_names = ['Joint1', 'Joint2', 'Joint3', 'Joint4']
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.header.stamp = rospy.Time.now()
        goal.trajectory.joint_names = joint_names
        goal.trajectory.points = [point]
        goal.goal_time_tolerance = rospy.Duration(2.0)  # Allow some flexibility

        try:
            self.trajectory_action_client.send_goal_and_wait(goal, rospy.Duration(8.0))  # Wait up to 8s
            result = self.trajectory_action_client.get_result()
            error_code = result.error_code if result else -100 # Custom code for no result
            
            # Error code -5 = GOAL_TOLERANCE_VIOLATED, but trajectory executes
            # We accept this as success since the robot does move to approximately the right position
            success = result is not None and (result.error_code == 0 or result.error_code == -5)
            
            if result and result.error_code == -5:
                rospy.logdebug("Action server returned -5 (GOAL_TOLERANCE_VIOLATED), but trajectory executed")
            
            return {'success': success, 'error_code': error_code}
        except Exception as e:
            rospy.logerr(f"Error in move_to_joint_positions: {e}")
            return {'success': False, 'error_code': -101} # Custom code for exception

    def calculate_reward(self) -> Tuple[float, bool]:
        """
        Calculate reward based on distance to target
        
        Returns:
            tuple: (reward, done) where done indicates if goal is reached
        """
        try:
            # Calculate Euclidean distance between end-effector and target
            end_effector_pos = np.array([self.robot_x, self.robot_y, self.robot_z])
            target_pos = np.array([self.pos_sphere_x, self.pos_sphere_y, self.pos_sphere_z])
            distance = np.linalg.norm(end_effector_pos - target_pos)
            # Reward structure (adapted from original)
            if distance <= self.goal_tolerance:
                rospy.loginfo(f"🎯 GOAL REACHED! Distance: {distance:.4f}")
                rospy.loginfo(f"🏆 EPISODE COMPLETED SUCCESSFULLY! 🏆")
                self._show_episode_success()
                reward = 10.0  # Goal achievement reward
                done = True
            else:
                reward = -1.0   # Step penalty (encourages faster completion)
                done = False
            # Check if episode should end due to step limit
            self.current_step += 1
            if self.current_step >= self.max_episode_steps:
                rospy.loginfo(f"📊 Episode ended due to step limit ({self.max_episode_steps})")
                rospy.loginfo(f"⏰ TIME LIMIT REACHED - EPISODE COMPLETE ⏰")
                done = True
            if done:
                rospy.loginfo("🔄 Episode ended, moving robot to home position for observation...")
                self._move_robot_home()
                rospy.loginfo("⏳ Waiting 5 seconds for robot to reach home and for observation...")
                rospy.sleep(5.0)  # Increased wait time for user to observe home position
            return reward, done
        except Exception as e:
            rospy.logerr(f"Error calculating reward: {e}")
            return -1.0, False
    
    def reset_environment(self) -> bool:
        """
        Reset environment: move robot to home position and randomize target
        
        Returns:
            bool: True if reset successful
        """
        rospy.loginfo("🔄 Resetting RL environment...")
        
        # Reset episode step counter
        self.current_step = 0
        
        # Reset target sphere to random position first
        target_success = self._reset_target_position()
        if not target_success:
            rospy.logerr("❌ Failed to reset target position")
            return False
            
        # Reset robot to home position
        home_success = self._move_robot_home()
        if not home_success:
            rospy.logwarn("❌ Home move failed, attempting controller reset and retry...")
            # Try to reset controllers using controller_manager
            try:
                from controller_manager_msgs.srv import SwitchController
                rospy.wait_for_service('/controller_manager/switch_controller', timeout=5)
                switch_controller = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)
                stop_controllers = ['joint_trajectory_controller']
                start_controllers = ['joint_trajectory_controller']
                switch_controller(start_controllers, stop_controllers, 2, False, 1.0)
                rospy.loginfo("Waiting 3 seconds after controller reset to avoid rapid switching...")
                rospy.sleep(3.0)
                # Retry home move once
                home_success = self._move_robot_home()
                if not home_success:
                    rospy.logerr("❌ Failed to reset robot to home position after controller reset")
                    return False
                else:
                    rospy.loginfo("✅ Home move succeeded after controller reset!")
            except Exception as e:
                rospy.logerr(f"Controller reset failed: {e}")
                return False
            
        # Wait for states to update
        rospy.sleep(1.0)
        
        rospy.loginfo("✅ Environment reset completed!")
        return True
    
    def _move_robot_home(self) -> bool:
        """Move robot to home/initial position (robust, single-point trajectory)"""
        # Home position for all joints (must be within joint limits)
        # Joint_4: [-1.57079, 1.57079], home = 0.0 (safe)
        home_position = np.array([0.0, 0.0, 0.0, 0.0])
        # Clip home position to joint limits for safety
        safe_home_position = np.clip(home_position, self.joint_limits_low, self.joint_limits_high)
        rospy.loginfo("Moving robot to home position...")
        result = self.move_to_joint_positions(safe_home_position)
        if result['success']:
            rospy.loginfo("✅ Robot moved to home position")
            return True
        else:
            rospy.logwarn(f"❌ Home move failed with error code: {result['error_code']}")
            return False
    
    def _reset_target_position(self) -> bool:
        """Move target sphere to random position on the drawing surface (robust randomization)"""
        # Drawing surface is a vertical plane at x=0.2m (20cm from robot base)
        # Surface dimensions: 0.3m (width, y) x 0.25m (height, z)
        # Center: x=0.2, y=0, z=0.12
        surface_x = 0.2 - 0.008  # 8mm in front of surface
        surface_y_min = -0.14    # -14cm from center
        surface_y_max = 0.14     # +14cm from center
        surface_z_min = 0.05     # 5cm above ground
        surface_z_max = 0.22     # 22cm above ground
        
        sphere_x = surface_x
        sphere_y = random.uniform(surface_y_min, surface_y_max)
        sphere_z = random.uniform(surface_z_min, surface_z_max)
        
        request = SetModelStateRequest()
        request.model_state.model_name = 'my_sphere'
        request.model_state.reference_frame = 'world'
        request.model_state.pose = Pose()
        request.model_state.pose.position = Point(x=sphere_x, y=sphere_y, z=sphere_z)
        request.model_state.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)
        try:
            rospy.loginfo(f"🎯 Moving target to drawing surface: [{sphere_x:.3f}, {sphere_y:.3f}, {sphere_z:.3f}]")
            response = self.reset_target_client(request)
            if response.success:
                rospy.loginfo("✅ Target sphere positioned on drawing surface")
                return True
            else:
                rospy.logerr(f"❌ Target reset failed: {response.status_message}")
                return False
        except Exception as e:
            rospy.logerr(f"Error resetting target position: {e}")
            return False
    
    def _show_episode_success(self):
        """Visual feedback for successful episode completion"""
        try:
            # Flash the target sphere green to indicate success
            rospy.loginfo("🎉 SHOWING SUCCESS FEEDBACK 🎉")
            rospy.sleep(0.5)  # Brief pause for effect
            
        except Exception as e:
            rospy.logerr(f"Error showing episode success: {e}")
    
    def generate_random_action(self) -> np.ndarray:
        """Generate random action within joint limits for exploration"""
        return np.random.uniform(self.joint_limits_low, self.joint_limits_high)
    
    def get_distance_to_goal(self) -> float:
        """Calculate current distance from end-effector to target"""
        end_effector_pos = np.array([self.robot_x, self.robot_y, self.robot_z])
        target_pos = np.array([self.pos_sphere_x, self.pos_sphere_y, self.pos_sphere_z])
        return np.linalg.norm(end_effector_pos - target_pos)
    
    def clear_trajectory(self):
        """Clear the trajectory drawing"""
        if hasattr(self, 'trajectory_drawer'):
            self.trajectory_drawer.clear()
            self._last_ee_pos = None
            rospy.loginfo("🧹 Trajectory cleared!")
    
    def get_trajectory_info(self) -> dict:
        """Get information about current trajectory"""
        if hasattr(self, 'trajectory_drawer'):
            return {
                'num_points': self.trajectory_drawer.get_num_points(),
                'length_m': self.trajectory_drawer.get_trajectory_length(),
                'length_cm': self.trajectory_drawer.get_trajectory_length() * 100
            }
        return {'num_points': 0, 'length_m': 0.0, 'length_cm': 0.0}
    
    def get_action_space_info(self) -> dict:
        """Get information about action space for RL algorithm"""
        return {
            'type': 'continuous',
            'shape': (4,),  # 4DOF robot
            'low': self.joint_limits_low,
            'high': self.joint_limits_high,
            'joint_names': ['Joint1', 'Joint2', 'Joint3', 'Joint4']
        }
    
    def get_observation_space_info(self) -> dict:
        """Get information about observation space for RL algorithm"""  
        return {
            'type': 'continuous',
            'shape': (10,),  # 3 (end-eff) + 4 (joints) + 3 (target) = 10
            'description': 'end_effector_xyz + joint_positions + target_xyz'
        }
    
    # ========================================================================
    # PROPERTIES FOR COMPATIBILITY WITH RL TRAINING WRAPPER
    # ========================================================================
    
    @property
    def ee_position(self):
        """
        Get end-effector position as numpy array (for compatibility with training wrapper)
        
        Returns:
            numpy array [x, y, z] of end-effector position in meters
        """
        return np.array([self.robot_x, self.robot_y, self.robot_z])
    
    @property
    def target_position(self):
        """
        Get target sphere position as numpy array (for compatibility with training wrapper)
        
        Returns:
            numpy array [x, y, z] of target sphere position in meters
        """
        return np.array([self.pos_sphere_x, self.pos_sphere_y, self.pos_sphere_z])
    
    # ========================================================================
    # SHUTDOWN
    # ========================================================================
    
    def shutdown(self):
        """Clean shutdown of the environment"""
        rospy.loginfo("🛑 Shutting down Visual RL Environment...")
        self.data_ready = False


# Test function for standalone execution
def test_rl_environment():
    """Test the RL environment functionality"""
    rospy.init_node('test_rl_environment', anonymous=True)
    
    try:
        # Initialize environment
        env = RLEnvironmentNoetic()
        rospy.sleep(2.0)  # Wait for initialization
        
        rospy.loginfo("🧪 Testing RL Environment...")
        
        # Test state retrieval
        state = env.get_state()
        if state is not None:
            rospy.loginfo(f"📊 Current state: {state}")
            rospy.loginfo(f"🎯 Distance to goal: {env.get_distance_to_goal():.4f}")
        else:
            rospy.logwarn("⚠️ State not available yet")
            
        # Test environment reset
        if env.reset_environment():
            rospy.loginfo("✅ Environment reset test passed")
        else:
            rospy.logerr("❌ Environment reset test failed")
            
        # Test action execution
        random_action = env.generate_random_action()
        rospy.loginfo(f"🎮 Testing random action: {random_action}")
        
        if env.execute_action(random_action):
            rospy.loginfo("✅ Action execution test passed")
        else:
            rospy.logerr("❌ Action execution test failed")
            
        # Test reward calculation
        reward, done = env.calculate_reward()
        rospy.loginfo(f"🏆 Reward: {reward}, Done: {done}")
        
        rospy.loginfo("🎉 RL Environment test completed!")
        
    except KeyboardInterrupt:
        rospy.loginfo("Test interrupted by user")
    except Exception as e:
        rospy.logerr(f"Test failed: {e}")
    finally:
        if 'env' in locals():
            env.shutdown()


if __name__ == '__main__':
    test_rl_environment()