#!/usr/bin/env python3
"""
Test Gazebo Trajectory Drawing
Simple script to verify trajectory lines appear in Gazebo simulation
"""

import rospy
import numpy as np
from gazebo_trajectory_drawer import GazeboTrajectoryDrawer

def main():
    rospy.init_node('test_gazebo_trajectory')
    
    print("\n" + "="*60)
    print("🎨 Testing Gazebo Trajectory Drawer")
    print("="*60)
    
    # Create drawer
    print("\n📍 Creating Gazebo trajectory drawer...")
    drawer = GazeboTrajectoryDrawer(color='green', line_width=0.003)
    
    # Draw a test pattern - spiral
    print("\n🌀 Drawing spiral pattern in Gazebo...")
    print("   (Look at Gazebo window - green cylinders should appear)")
    
    num_points = 30
    for i in range(num_points):
        t = i / num_points
        angle = 4 * np.pi * t
        radius = 0.05 * (1 - t)
        
        x = 0.15 + radius * np.cos(angle)
        y = 0.0 + radius * np.sin(angle)
        z = 0.12 + t * 0.10
        
        drawer.add_point(x, y, z)
        
        if i % 5 == 0:
            print(f"   Point {i+1}/{num_points}: ({x:.3f}, {y:.3f}, {z:.3f})")
        
        rospy.sleep(0.15)
    
    print(f"\n✅ Drew {drawer.get_point_count()} points")
    print(f"   Spawned {len(drawer.segment_models)} cylinder segments")
    
    print("\n⏱️  Spiral will stay visible for 15 seconds...")
    print("   (Check Gazebo - you should see green cylinders forming a spiral)")
    rospy.sleep(15)
    
    # Clear
    print("\n🧹 Clearing trajectory from Gazebo...")
    drawer.clear()
    
    print("\n✅ Test complete!")
    print("="*60 + "\n")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
