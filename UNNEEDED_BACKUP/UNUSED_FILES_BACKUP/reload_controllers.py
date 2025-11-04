#!/usr/bin/env python3
"""
Reload Controllers Script
========================

Reload controllers with new parameters without restarting Gazebo
"""

import rospy
from controller_manager_msgs.srv import SwitchController, LoadController, UnloadController
import time

def reload_controllers():
    """Reload arm_controller with new parameters"""
    rospy.init_node('reload_controllers', anonymous=True)
    
    try:
        # Wait for controller manager services
        rospy.wait_for_service('/controller_manager/switch_controller', timeout=5.0)
        rospy.wait_for_service('/controller_manager/load_controller', timeout=5.0)
        rospy.wait_for_service('/controller_manager/unload_controller', timeout=5.0)
        
        # Service proxies
        switch_controller = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)
        load_controller = rospy.ServiceProxy('/controller_manager/load_controller', LoadController)
        unload_controller = rospy.ServiceProxy('/controller_manager/unload_controller', UnloadController)
        
        rospy.loginfo("🔄 Stopping arm_controller...")
        # Stop arm_controller
        result = switch_controller([], ['arm_controller'], 2, True, 0.0)
        if result.ok:
            rospy.loginfo("✅ arm_controller stopped")
        else:
            rospy.logwarn("❌ Failed to stop arm_controller")
            return
        
        time.sleep(1.0)
        
        rospy.loginfo("📤 Unloading arm_controller...")
        # Unload controller
        result = unload_controller('arm_controller')
        if result.ok:
            rospy.loginfo("✅ arm_controller unloaded")
        else:
            rospy.logwarn("❌ Failed to unload arm_controller")
            return
        
        time.sleep(1.0)
        
        rospy.loginfo("📥 Reloading arm_controller with new parameters...")
        # Reload controller (this will pick up new parameters)
        result = load_controller('arm_controller')
        if result.ok:
            rospy.loginfo("✅ arm_controller reloaded")
        else:
            rospy.logwarn("❌ Failed to reload arm_controller")
            return
        
        time.sleep(1.0)
        
        rospy.loginfo("▶️ Starting arm_controller...")
        # Start controller
        result = switch_controller(['arm_controller'], [], 2, True, 0.0)
        if result.ok:
            rospy.loginfo("✅ arm_controller started with new parameters!")
            rospy.loginfo("🎯 Controller reload completed successfully")
        else:
            rospy.logwarn("❌ Failed to start arm_controller")
    
    except Exception as e:
        rospy.logerr(f"❌ Controller reload failed: {e}")

if __name__ == '__main__':
    reload_controllers()