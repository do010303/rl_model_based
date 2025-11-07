#!/usr/bin/env python3
"""
Gazebo Real-Time Trajectory Drawing using Visual Plugin
Creates trajectory lines in Gazebo in real-time (like RViz)

This uses Gazebo's visual marker system instead of spawning models
Much faster - updates in real-time!
"""

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import numpy as np


class GazeboRealtimeTrajectory:
    """
    Real-time trajectory drawing in Gazebo using visualization markers
    
    This works the SAME as RViz - publishes to /visualization_marker
    Gazebo's visualization plugin will render it in real-time!
    """
    
    def __init__(self, frame_id='world', color='green', line_width=0.003, namespace='gazebo_trajectory'):
        """
        Initialize real-time Gazebo trajectory drawer
        
        Args:
            frame_id: Reference frame (default: 'world')
            color: Line color
            line_width: Line thickness in meters
            namespace: Marker namespace (different from RViz to show both)
        """
        # Publisher - same topic as RViz but different namespace
        self.marker_pub = rospy.Publisher(
            '/visualization_marker',
            Marker,
            queue_size=10
        )
        
        # Trajectory points
        self.trajectory_points = []
        
        # Configuration
        self.frame_id = frame_id
        self.line_width = line_width
        self.namespace = namespace
        self.color = self._parse_color(color)
        
        # Marker ID
        self.marker_id = 1  # Different from RViz (0)
        
        rospy.loginfo(f"🎨 Gazebo real-time trajectory initialized (namespace: {namespace})")
    
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
        Add a point to the trajectory - updates in real-time!
        
        Args:
            x, y, z: 3D coordinates
        """
        point = Point()
        point.x = float(x)
        point.y = float(y)
        point.z = float(z)
        
        self.trajectory_points.append(point)
        
        # Update visualization immediately
        self._publish_trajectory()
    
    def add_point_array(self, position):
        """
        Add point from array [x, y, z]
        
        Args:
            position: numpy array or list [x, y, z]
        """
        self.add_point(position[0], position[1], position[2])
    
    def _publish_trajectory(self):
        """Publish trajectory as LINE_STRIP marker (same as RViz!)"""
        if len(self.trajectory_points) < 2:
            return
        
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = self.namespace
        marker.id = self.marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # Scale (line width)
        marker.scale.x = self.line_width
        
        # Color
        marker.color = self.color
        
        # Points
        marker.points = self.trajectory_points
        
        # Lifetime (0 = forever)
        marker.lifetime = rospy.Duration(0)
        
        # Publish
        self.marker_pub.publish(marker)
    
    def clear(self):
        """Clear trajectory"""
        # Send DELETE action
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = self.namespace
        marker.id = self.marker_id
        marker.action = Marker.DELETE
        
        self.marker_pub.publish(marker)
        
        # Reset points
        self.trajectory_points = []
        
        rospy.loginfo(f"🧹 Gazebo trajectory cleared ({self.namespace})")
    
    def get_point_count(self):
        """Return number of points"""
        return len(self.trajectory_points)


# Test code
if __name__ == '__main__':
    rospy.init_node('test_gazebo_realtime_trajectory')
    
    rospy.loginfo("Testing Gazebo Real-Time Trajectory...")
    
    drawer = GazeboRealtimeTrajectory(color='cyan', line_width=0.005, namespace='test_trajectory')
    
    # Draw test circle
    rospy.loginfo("Drawing test circle in Gazebo...")
    num_points = 50
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
        rospy.sleep(0.05)  # 50ms - real-time!
    
    rospy.loginfo(f"✅ Drew {drawer.get_point_count()} points in real-time!")
    rospy.loginfo("Circle visible in Gazebo for 10 seconds...")
    rospy.sleep(10)
    
    drawer.clear()
    rospy.loginfo("Test complete!")
