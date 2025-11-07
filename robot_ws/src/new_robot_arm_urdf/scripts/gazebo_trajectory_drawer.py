#!/usr/bin/env python3
"""
Gazebo Trajectory Drawing System for Robot End-Effector
Creates visual line markers DIRECTLY in Gazebo simulation (not just RViz)

This draws lines in Gazebo using individual cylinder models for each line segment
"""

import rospy
from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import Pose, Point, Quaternion
import numpy as np
import math


class GazeboTrajectoryDrawer:
    """
    Draws the end-effector trajectory as cylinders in Gazebo
    
    Features:
    - Visible directly in Gazebo simulation (not just RViz)
    - Each line segment is a thin cylinder
    - Can be cleared on demand
    - Customizable color and thickness
    """
    
    def __init__(self, color='green', line_width=0.002):
        """
        Initialize Gazebo trajectory drawer
        
        Args:
            color: Line color ('green', 'red', 'blue', 'yellow', etc.)
            line_width: Thickness of the line in meters (default: 2mm)
        """
        # Wait for Gazebo services
        rospy.loginfo("🎨 Waiting for Gazebo services...")
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        rospy.wait_for_service('/gazebo/delete_model')
        
        # Service proxies
        self.spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        
        # Trajectory points storage
        self.trajectory_points = []
        
        # Configuration
        self.line_width = line_width
        self.color = self._parse_color(color)
        
        # Model tracking
        self.segment_models = []  # List of spawned model names
        self.segment_counter = 0
        
        rospy.loginfo("🎨 Gazebo trajectory drawer initialized!")
    
    def _parse_color(self, color_name):
        """Convert color name to Gazebo material color"""
        colors = {
            'red': (1.0, 0.0, 0.0, 1.0),
            'green': (0.0, 1.0, 0.0, 1.0),
            'blue': (0.0, 0.0, 1.0, 1.0),
            'yellow': (1.0, 1.0, 0.0, 1.0),
            'cyan': (0.0, 1.0, 1.0, 1.0),
            'magenta': (1.0, 0.0, 1.0, 1.0),
            'white': (1.0, 1.0, 1.0, 1.0),
            'orange': (1.0, 0.5, 0.0, 1.0),
        }
        return colors.get(color_name.lower(), colors['green'])
    
    def _create_cylinder_sdf(self, length, radius, color):
        """
        Create SDF model string for a cylinder
        
        Args:
            length: Length of cylinder (line segment)
            radius: Radius of cylinder (line thickness)
            color: RGBA tuple (r, g, b, a)
        
        Returns:
            SDF model string
        """
        r, g, b, a = color
        
        sdf = f"""<?xml version='1.0'?>
<sdf version='1.6'>
  <model name='trajectory_segment'>
    <static>true</static>
    <link name='link'>
      <visual name='visual'>
        <geometry>
          <cylinder>
            <radius>{radius}</radius>
            <length>{length}</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>{r} {g} {b} {a}</ambient>
          <diffuse>{r} {g} {b} {a}</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
          <emissive>{r*0.3} {g*0.3} {b*0.3} 0</emissive>
        </material>
      </visual>
      <collision name='collision'>
        <geometry>
          <cylinder>
            <radius>{radius}</radius>
            <length>{length}</length>
          </cylinder>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>"""
        return sdf
    
    def _calculate_cylinder_pose(self, p1, p2):
        """
        Calculate pose (position + orientation) for cylinder connecting two points
        
        Args:
            p1: Start point [x, y, z]
            p2: End point [x, y, z]
        
        Returns:
            Pose object with position and orientation
        """
        # Midpoint (cylinder center)
        mid_x = (p1[0] + p2[0]) / 2.0
        mid_y = (p1[1] + p2[1]) / 2.0
        mid_z = (p1[2] + p2[2]) / 2.0
        
        # Direction vector
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dz = p2[2] - p1[2]
        
        # Length
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        if length < 1e-6:
            # Points too close, skip
            return None, 0.0
        
        # Normalize direction
        dx /= length
        dy /= length
        dz /= length
        
        # Calculate orientation quaternion
        # Cylinder's default axis is Z, we need to rotate it to align with [dx, dy, dz]
        
        # Use rotation from Z-axis [0,0,1] to direction [dx,dy,dz]
        # Axis of rotation: cross product
        ax = -dy  # [0,0,1] x [dx,dy,dz] = [-dy, dx, 0]
        ay = dx
        az = 0.0
        
        # Angle of rotation
        angle = math.acos(np.clip(dz, -1.0, 1.0))  # angle between [0,0,1] and [dx,dy,dz]
        
        # Convert axis-angle to quaternion
        if abs(angle) < 1e-6:
            # No rotation needed
            qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
        elif abs(angle - math.pi) < 1e-6:
            # 180 degree rotation - use any perpendicular axis
            qx, qy, qz, qw = 1.0, 0.0, 0.0, 0.0
        else:
            # Normalize axis
            axis_len = math.sqrt(ax*ax + ay*ay + az*az)
            if axis_len > 1e-6:
                ax /= axis_len
                ay /= axis_len
                az /= axis_len
            
            # Quaternion from axis-angle
            half_angle = angle / 2.0
            sin_half = math.sin(half_angle)
            qx = ax * sin_half
            qy = ay * sin_half
            qz = az * sin_half
            qw = math.cos(half_angle)
        
        # Create pose
        pose = Pose()
        pose.position.x = mid_x
        pose.position.y = mid_y
        pose.position.z = mid_z
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw
        
        return pose, length
    
    def add_point(self, x, y, z):
        """
        Add a point to the trajectory and draw line segment
        
        Args:
            x, y, z: 3D coordinates of the point
        """
        current_point = np.array([float(x), float(y), float(z)])
        
        # If we have a previous point, draw line segment
        if len(self.trajectory_points) > 0:
            prev_point = self.trajectory_points[-1]
            self._spawn_line_segment(prev_point, current_point)
        
        # Store point
        self.trajectory_points.append(current_point)
    
    def add_point_array(self, position):
        """
        Add a point from numpy array [x, y, z]
        
        Args:
            position: numpy array or list [x, y, z]
        """
        self.add_point(position[0], position[1], position[2])
    
    def _spawn_line_segment(self, p1, p2):
        """
        Spawn a cylinder in Gazebo connecting two points
        
        Args:
            p1: Start point [x, y, z]
            p2: End point [x, y, z]
        """
        try:
            # Calculate pose and length
            pose, length = self._calculate_cylinder_pose(p1, p2)
            
            if pose is None:
                return  # Points too close
            
            # Create cylinder SDF
            sdf = self._create_cylinder_sdf(
                length=length,
                radius=self.line_width / 2.0,
                color=self.color
            )
            
            # Generate unique model name
            model_name = f'trajectory_seg_{self.segment_counter}'
            self.segment_counter += 1
            
            # Spawn model in Gazebo
            self.spawn_model(
                model_name=model_name,
                model_xml=sdf,
                robot_namespace='',
                initial_pose=pose,
                reference_frame='world'
            )
            
            # Track model for cleanup
            self.segment_models.append(model_name)
            
        except Exception as e:
            rospy.logwarn(f"Failed to spawn line segment: {e}")
    
    def clear(self):
        """Clear all trajectory segments from Gazebo"""
        rospy.loginfo(f"🧹 Clearing {len(self.segment_models)} trajectory segments...")
        
        for model_name in self.segment_models:
            try:
                self.delete_model(model_name)
            except Exception as e:
                rospy.logwarn(f"Failed to delete {model_name}: {e}")
        
        # Reset tracking
        self.segment_models = []
        self.trajectory_points = []
        self.segment_counter = 0
        
        rospy.loginfo("✨ Trajectory cleared!")
    
    def get_point_count(self):
        """Return number of trajectory points"""
        return len(self.trajectory_points)


# Test code
if __name__ == '__main__':
    rospy.init_node('test_gazebo_trajectory_drawer')
    
    rospy.loginfo("Testing Gazebo Trajectory Drawer...")
    
    drawer = GazeboTrajectoryDrawer(color='green', line_width=0.003)
    
    # Draw a test trajectory (circle)
    rospy.loginfo("Drawing test circle...")
    num_points = 20
    radius = 0.05
    center_x = 0.15
    center_y = 0.0
    center_z = 0.15
    
    for i in range(num_points + 1):
        angle = 2 * np.pi * i / num_points
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        z = center_z
        
        drawer.add_point(x, y, z)
        rospy.sleep(0.1)
    
    rospy.loginfo(f"✅ Drew {drawer.get_point_count()} points")
    rospy.loginfo("Circle will stay for 10 seconds...")
    rospy.sleep(10)
    
    rospy.loginfo("Clearing trajectory...")
    drawer.clear()
    
    rospy.loginfo("Test complete!")
