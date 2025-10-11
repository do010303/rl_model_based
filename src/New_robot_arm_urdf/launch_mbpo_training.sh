#!/bin/bash

# MBPO Training Launcher for Robot Arm
# This script provides easy commands to train the robot arm using MBPO

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🤖 MBPO Robot Arm Training Launcher${NC}"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "scripts/mbpo_ros_trainer.py" ]; then
    echo -e "${RED}❌ Error: Please run this script from the robot arm directory${NC}"
    echo "Expected location: /home/ducanh/rl_model_based/src/New_robot_arm_urdf/"
    exit 1
fi

# Function to check ROS environment
check_ros_environment() {
    if [ -z "$ROS_DISTRO" ]; then
        echo -e "${YELLOW}⚠️  ROS environment not detected, sourcing...${NC}"
        source /opt/ros/noetic/setup.bash
        source /home/ducanh/rl_model_based/devel/setup.bash
    else
        echo -e "${GREEN}✅ ROS environment detected: $ROS_DISTRO${NC}"
    fi
}

# Function to kill existing Gazebo processes
cleanup_gazebo() {
    echo -e "${YELLOW}🧹 Cleaning up existing Gazebo processes...${NC}"
    pkill -f gzserver || true
    pkill -f gzclient || true
    pkill -f rosmaster || true
    sleep 2
}

# Function to show training options
show_menu() {
    echo ""
    echo "Available training options:"
    echo "1. Quick MBPO Training (100 episodes)"
    echo "2. Full MBPO Training (1000 episodes)" 
    echo "3. Extended MBPO Training (2000 episodes)"
    echo "4. Custom MBPO Training (specify parameters)"
    echo "5. Direct MBPO Script (advanced users)"
    echo "6. Stable Baselines3 Comparison"
    echo "7. Exit"
    echo ""
}

# Function to run MBPO training
run_mbpo_training() {
    local episodes=$1
    local max_steps=$2
    local ensemble_size=$3
    local description=$4
    
    echo -e "${GREEN}🚀 Starting $description${NC}"
    echo "Episodes: $episodes"
    echo "Max steps per episode: $max_steps" 
    echo "Dynamics ensemble size: $ensemble_size"
    echo ""
    
    cleanup_gazebo
    check_ros_environment
    
    # Run MBPO training
    python3 scripts/mbpo_ros_trainer.py \
        --episodes $episodes \
        --max-steps $max_steps \
        --ensemble-size $ensemble_size \
        --headless \
        --buffer-capacity 100000 \
        --model-train-freq 250 \
        --rollout-freq 100
}

# Function to run via main training script
run_via_main_script() {
    local timesteps=$1
    local description=$2
    
    echo -e "${GREEN}🚀 Starting $description${NC}"
    echo "Timesteps: $timesteps"
    echo ""
    
    cleanup_gazebo
    check_ros_environment
    
    # Run via main training script
    python3 scripts/train_robot_arm.py \
        --algorithm MBPO \
        --timesteps $timesteps \
        --mode train
}

# Main menu loop
while true; do
    show_menu
    read -p "Select an option (1-7): " choice
    
    case $choice in
        1)
            run_mbpo_training 100 200 3 "Quick MBPO Training"
            ;;
        2)
            run_mbpo_training 1000 200 3 "Full MBPO Training"
            ;;
        3)
            run_mbpo_training 2000 200 5 "Extended MBPO Training"
            ;;
        4)
            echo ""
            read -p "Enter number of episodes (default: 1000): " episodes
            episodes=${episodes:-1000}
            
            read -p "Enter max steps per episode (default: 200): " max_steps
            max_steps=${max_steps:-200}
            
            read -p "Enter ensemble size (default: 3): " ensemble_size
            ensemble_size=${ensemble_size:-3}
            
            run_mbpo_training $episodes $max_steps $ensemble_size "Custom MBPO Training"
            ;;
        5)
            echo -e "${BLUE}💡 Direct MBPO script usage:${NC}"
            echo "python3 scripts/mbpo_ros_trainer.py --help"
            echo ""
            echo "Example:"
            echo "python3 scripts/mbpo_ros_trainer.py --episodes 1000 --ensemble-size 5 --headless"
            echo ""
            ;;
        6)
            echo ""
            echo "Select comparison algorithm:"
            echo "1. SAC vs MBPO"
            echo "2. PPO vs MBPO" 
            echo "3. TD3 vs MBPO"
            read -p "Choose (1-3): " comp_choice
            
            case $comp_choice in
                1) algo="SAC" ;;
                2) algo="PPO" ;;
                3) algo="TD3" ;;
                *) echo "Invalid choice"; continue ;;
            esac
            
            echo -e "${GREEN}🔄 Running $algo for comparison${NC}"
            cleanup_gazebo
            check_ros_environment
            
            python3 scripts/train_robot_arm.py \
                --algorithm $algo \
                --timesteps 200000 \
                --mode train
            ;;
        7)
            echo -e "${GREEN}👋 Goodbye!${NC}"
            cleanup_gazebo
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Invalid option. Please select 1-7.${NC}"
            ;;
    esac
    
    echo ""
    echo -e "${YELLOW}Training completed! Check results in /home/ducanh/rl_model_based/results/${NC}"
    echo ""
    read -p "Press Enter to continue or Ctrl+C to exit..."
done