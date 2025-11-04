#!/bin/bash
# Helper script to run safety tests with correct environment setup

echo "=========================================="
echo "Robot Safety Feature Tests"
echo "=========================================="
echo ""
echo "⚠️  Make sure Gazebo is running first!"
echo "   Terminal 1: roslaunch new_robot_arm_urdf robot_4dof_rl_gazebo.launch"
echo ""
echo "Press Enter to continue or Ctrl+C to cancel..."
read

cd ~/rl_model_based/robot_ws
source devel/setup.bash

echo ""
echo "Running safety tests..."
echo ""

python3 src/new_robot_arm_urdf/scripts/test_safety_features.py
