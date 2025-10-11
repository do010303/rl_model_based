#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import numpy as np

class TargetMarker:
    """Visualize the target position in RViz"""
    
    def __init__(self):
        rospy.init_node('target_marker', anonymous=True)
        
        # Publisher for target marker
        self.marker_pub = rospy.Publisher('/target_marker', Marker, queue_size=1)
        
        # Target position (should match your RL environment)
        self.target_pos = [0.2, 0.0, 0.3]
        
        # Timer for publishing marker
        self.timer = rospy.Timer(rospy.Duration(0.1), self.publish_marker)
        
        rospy.loginfo("Target marker publisher started")
        
    def publish_marker(self, event):
        """Publish target marker"""
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        
        marker.ns = "target"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # Set position
        marker.pose.position.x = self.target_pos[0]
        marker.pose.position.y = self.target_pos[1] 
        marker.pose.position.z = self.target_pos[2]
        marker.pose.orientation.w = 1.0
        
        # Set scale
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        
        # Set color (red)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        
        # Publish marker
        self.marker_pub.publish(marker)

if __name__ == '__main__':
    try:
        target_marker = TargetMarker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass