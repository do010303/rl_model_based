#!/usr/bin/env python3
"""
Trajectory Drawing System for Robot End-Effector
Creates visual line markers in Gazebo to show robot movement path
"""

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import numpy as np


class TrajectoryDrawer:
    """
    Draws the end-effector trajectory as a line in Gazebo/RViz
    
    Features:
    - Accumulates end-effector positions over time
    - Visualizes as a continuous line
    - Can be cleared on demand
    - Customizable color and thickness
    """
    
    def __init__(self, frame_id='world', color='green', line_width=0.003):
        """
        Initialize trajectory drawer
        
        Args:
            frame_id: Reference frame for markers (default: 'world')
            color: Line color ('green', 'red', 'blue', 'yellow', etc.)
            line_width: Thickness of the line in meters (default: 3mm)
        """
        # Publisher for visualization markers
        self.marker_pub = rospy.Publisher(
            '/visualization_marker',
            Marker,
            queue_size=10
        )
        
        # Trajectory points storage
        self.trajectory_points = []
        
        # Configuration
        self.frame_id = frame_id
        self.line_width = line_width
        self.color = self._parse_color(color)
        
        # Marker ID (for deletion)
        self.marker_id = 0
        
        rospy.loginfo("🎨 Trajectory drawer initialized!")
    
    def _parse_color(self, color_name):
        """Convert color name to RGBA"""
        colors = {
            'red': ColorRGBA(1.0, 0.0, 0.0, 1.0),
            'green': ColorRGBA(0.0, 1.0, 0.0, 1.0),
            'blue': ColorRGBA(0.0, 0.0, 1.0, 1.0),
            'yellow': ColorRGBA(1.0, 1.0, 0.0, 1.0),
            'cyan': ColorRGBA(0.0, 1.0, 1.0, 1.0),
            'magenta': ColorRGBA(1.0, 0.0, 1.0, 1.0),
            'white': ColorRGBA(1.0, 1.0, 1.0, 1.0),
            'orange': ColorRGBA(1.0, 0.5, 0.0, 1.0),
        }
        return colors.get(color_name.lower(), colors['green'])
    
    def add_point(self, x, y, z):
        """
        Add a point to the trajectory
        
        Args:
            x, y, z: 3D coordinates of the point
        """
        point = Point()
        point.x = float(x)
        point.y = float(y)
        point.z = float(z)
        
        self.trajectory_points.append(point)
        
        # Update visualization
        self._publish_trajectory()
    
    def add_point_array(self, position):
        """
        Add a point from numpy array [x, y, z]
        
        Args:
            position: numpy array or list [x, y, z]
        """
        self.add_point(position[0], position[1], position[2])
    
    def _publish_trajectory(self):
        """Publish the trajectory as a LINE_STRIP marker"""
        if len(self.trajectory_points) < 2:
            return  # Need at least 2 points to draw a line
        
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "ee_trajectory"
        marker.id = self.marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # Line properties
        marker.scale.x = self.line_width  # Line width
        marker.color = self.color
        
        # Add all points
        marker.points = self.trajectory_points
        
        # Marker lifetime (0 = forever)
        marker.lifetime = rospy.Duration(0)
        
        # Publish
        self.marker_pub.publish(marker)
    
    def clear(self):
        """Clear the trajectory (erase the drawing)"""
        # Delete the marker
        delete_marker = Marker()
        delete_marker.header.frame_id = self.frame_id
        delete_marker.header.stamp = rospy.Time.now()
        delete_marker.ns = "ee_trajectory"
        delete_marker.id = self.marker_id
        delete_marker.action = Marker.DELETE
        
        self.marker_pub.publish(delete_marker)
        
        # Clear internal storage
        self.trajectory_points = []
        
        # Increment marker ID for next drawing
        self.marker_id += 1
        
        rospy.loginfo("🧹 Trajectory cleared!")
    
    def change_color(self, color_name):
        """
        Change the color of future trajectory lines
        
        Args:
            color_name: New color name ('red', 'blue', etc.)
        """
        self.color = self._parse_color(color_name)
        rospy.loginfo(f"🎨 Trajectory color changed to {color_name}")
    
    def get_trajectory_length(self):
        """
        Calculate the total length of the trajectory
        
        Returns:
            float: Total length in meters
        """
        if len(self.trajectory_points) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(self.trajectory_points)):
            p1 = self.trajectory_points[i-1]
            p2 = self.trajectory_points[i]
            
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            dz = p2.z - p1.z
            
            segment_length = np.sqrt(dx*dx + dy*dy + dz*dz)
            total_length += segment_length
        
        return total_length
    
    def get_num_points(self):
        """Get the number of points in the trajectory"""
        return len(self.trajectory_points)
    
    def save_trajectory(self, filename):
        """
        Save trajectory points to a file
        
        Args:
            filename: Path to save file
        """
        try:
            points_array = np.array([[p.x, p.y, p.z] for p in self.trajectory_points])
            np.save(filename, points_array)
            rospy.loginfo(f"💾 Trajectory saved to {filename}")
            return True
        except Exception as e:
            rospy.logerr(f"Failed to save trajectory: {e}")
            return False
    
    def load_trajectory(self, filename):
        """
        Load trajectory points from a file
        
        Args:
            filename: Path to load file
        """
        try:
            points_array = np.load(filename)
            self.trajectory_points = []
            for point in points_array:
                self.add_point(point[0], point[1], point[2])
            rospy.loginfo(f"📂 Trajectory loaded from {filename}")
            return True
        except Exception as e:
            rospy.logerr(f"Failed to load trajectory: {e}")
            return False


# Test function
def test_trajectory_drawer():
    """Test the trajectory drawer"""
    rospy.init_node('test_trajectory_drawer', anonymous=True)
    
    drawer = TrajectoryDrawer(color='green', line_width=0.005)
    
    rospy.loginfo("Drawing a test trajectory...")
    
    # Draw a simple circle
    import math
    radius = 0.1
    center_x, center_y, center_z = 0.2, 0.0, 0.15
    
    for i in range(100):
        angle = 2 * math.pi * i / 100
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        z = center_z
        
        drawer.add_point(x, y, z)
        rospy.sleep(0.05)
    
    rospy.loginfo(f"✅ Drew {drawer.get_num_points()} points")
    rospy.loginfo(f"📏 Trajectory length: {drawer.get_trajectory_length():.4f}m")
    
    rospy.sleep(3.0)
    
    rospy.loginfo("Clearing trajectory...")
    drawer.clear()
    
    rospy.sleep(1.0)
    rospy.loginfo("Test complete!")


if __name__ == '__main__':
    test_trajectory_drawer()
