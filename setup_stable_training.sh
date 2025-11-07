#!/bin/bash
# Quick fix script for stable training configuration

echo "========================================================================"
echo "🔧 Setting up STABLE training configuration"
echo "========================================================================"

# Kill any running ROS/Gazebo processes
echo ""
echo "1️⃣  Killing any running ROS/Gazebo processes..."
killall -9 gzserver gzclient rosmaster roscore 2>/dev/null
sleep 2
echo "   ✅ Processes killed"

# Clean logs (optional)
echo ""
echo "2️⃣  Cleaning old logs..."
rm -rf ~/.ros/log/* 2>/dev/null
echo "   ✅ Logs cleared"

echo ""
echo "========================================================================"
echo "✅ READY TO START"
echo "========================================================================"
echo ""
echo "Run these commands in SEPARATE terminals:"
echo ""
echo "📋 Terminal 1 - Start Gazebo (HEADLESS - no GUI crashes!):"
echo "   cd ~/rl_model_based/robot_ws"
echo "   source devel/setup.bash"
echo "   roslaunch new_robot_arm_urdf robot_4dof_rl_gazebo.launch gui:=false"
echo ""
echo "📋 Terminal 2 - Start RViz (for trajectory visualization):"
echo "   cd ~/rl_model_based/robot_ws"
echo "   source devel/setup.bash"
echo "   rosrun rviz rviz -d ~/rl_model_based/robot_ws/src/new_robot_arm_urdf/rviz/trajectory_view.rviz"
echo ""
echo "📋 Terminal 3 - Run training:"
echo "   cd ~/rl_model_based/robot_ws/src/new_robot_arm_urdf/scripts"
echo "   python3 train_robot.py"
echo ""
echo "========================================================================"
echo "💡 TIPS:"
echo "   • gui:=false = No Gazebo GUI = No crashes!"
echo "   • RViz shows trajectory (smooth green line)"
echo "   • Start with small movements: 0.1 0 0 0"
echo "   • If robot breaks (NaN), run this script again"
echo "========================================================================"
