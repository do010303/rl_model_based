#!/usr/bin/env python3
"""
Quick test to verify trajectory visualization is working
Run this with Gazebo + RViz open to see the green line appear
"""

import rospy
from trajectory_drawer import TrajectoryDrawer
import time

def test_trajectory_visualization():
    rospy.init_node('test_trajectory_viz', anonymous=True)
    
    print("=" * 70)
    print("TRAJECTORY VISUALIZATION TEST")
    print("=" * 70)
    print()
    print("📋 Instructions:")
    print("   1. Make sure Gazebo is running:")
    print("      roslaunch new_robot_arm_urdf robot_4dof_rl_gazebo.launch")
    print()
    print("   2. Open RViz in another terminal:")
    print("      rosrun rviz rviz -d src/new_robot_arm_urdf/rviz/trajectory_view.rviz")
    print()
    print("   3. Watch for a green spiral appearing in RViz!")
    print()
    print("=" * 70)
    
    # Wait for user
    input("Press Enter when RViz is open and ready...")
    
    # Create trajectory drawer
    print("\n🎨 Creating trajectory drawer...")
    drawer = TrajectoryDrawer(color='green', line_width=0.01)
    time.sleep(1)
    
    # Draw a test spiral
    print("✏️  Drawing test spiral...")
    import numpy as np
    
    num_points = 100
    for i in range(num_points):
        t = i / num_points * 4 * np.pi  # 2 full rotations
        radius = 0.1 * (1 - i / num_points)  # Spiral inward
        
        x = 0.15 + radius * np.cos(t)
        y = radius * np.sin(t)
        z = 0.15 + i * 0.001  # Rise slowly
        
        drawer.add_point(x, y, z)
        time.sleep(0.02)  # Small delay to see it draw
        
        if i % 20 == 0:
            print(f"   Points drawn: {i}/{num_points}")
    
    print(f"\n✅ Complete! Drew {num_points} points")
    print(f"📏 Trajectory length: {drawer.get_trajectory_length():.3f}m")
    print()
    print("🔍 Check RViz - you should see a green spiral!")
    print()
    print("Press Ctrl+C to exit...")
    
    rospy.spin()

if __name__ == '__main__':
    try:
        test_trajectory_visualization()
    except rospy.ROSInterruptException:
        pass
