#!/usr/bin/env python3
"""
Robot Arm Visualization Script for 4-DOF Robot Arm
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.robot_4dof_env import Robot4DOFEnv
from agents.ddpg_agent import DDPGAgent

class RobotVisualizer:
    """3D Robot Arm Visualizer"""
    
    def __init__(self, env: Robot4DOFEnv, agent: Optional[DDPGAgent] = None):
        """
        Initialize robot visualizer.
        
        Args:
            env: Robot environment
            agent: Trained agent (optional)
        """
        self.env = env
        self.agent = agent
        
        # Setup figure
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Robot parameters
        self.link_lengths = [0.3, 0.25, 0.2, 0.15]  # 4 links
        self.joint_positions = []
        self.trajectory = []
        
    def forward_kinematics(self, joint_angles: np.ndarray) -> List[np.ndarray]:
        """
        Compute forward kinematics for visualization.
        
        Args:
            joint_angles: Array of joint angles [θ1, θ2, θ3, θ4]
            
        Returns:
            List of 3D positions for each joint
        """
        positions = [np.array([0, 0, 0])]  # Base position
        
        # Cumulative transformation
        x, y, z = 0, 0, 0
        cumulative_angle = 0
        
        for i, (angle, length) in enumerate(zip(joint_angles, self.link_lengths)):
            if i == 0:  # Base rotation (around z-axis)
                cumulative_angle = angle
                x = length * np.cos(cumulative_angle)
                y = length * np.sin(cumulative_angle)
                z = 0
            else:  # Subsequent joints
                cumulative_angle += angle
                x += length * np.cos(cumulative_angle)
                y += length * np.sin(cumulative_angle)
                z += 0  # Simplified 2D kinematics projected to 3D
                
            positions.append(np.array([x, y, z]))
            
        return positions
    
    def plot_robot_frame(self, joint_angles: np.ndarray, target_pos: np.ndarray, 
                        end_effector_path: List[np.ndarray] = None):
        """
        Plot single frame of robot visualization.
        
        Args:
            joint_angles: Current joint angles
            target_pos: Target position
            end_effector_path: Path traced by end effector
        """
        self.ax.clear()
        
        # Compute joint positions
        positions = self.forward_kinematics(joint_angles)
        
        # Plot robot links
        for i in range(len(positions) - 1):
            self.ax.plot3D([positions[i][0], positions[i+1][0]],
                          [positions[i][1], positions[i+1][1]], 
                          [positions[i][2], positions[i+1][2]], 
                          'b-', linewidth=3, marker='o', markersize=5)
        
        # Plot joints
        for i, pos in enumerate(positions):
            color = 'red' if i == 0 else 'blue'
            size = 8 if i == 0 else 6
            self.ax.scatter(pos[0], pos[1], pos[2], 
                           c=color, s=size**2, alpha=0.8)
        
        # Plot end effector
        end_pos = positions[-1]
        self.ax.scatter(end_pos[0], end_pos[1], end_pos[2], 
                       c='green', s=100, marker='*', label='End Effector')
        
        # Plot target
        self.ax.scatter(target_pos[0], target_pos[1], target_pos[2], 
                       c='red', s=150, marker='X', label='Target')
        
        # Plot trajectory
        if end_effector_path and len(end_effector_path) > 1:
            path = np.array(end_effector_path)
            self.ax.plot3D(path[:, 0], path[:, 1], path[:, 2], 
                          'g--', alpha=0.6, linewidth=1)
        
        # Set axis properties
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([-1, 1])
        self.ax.set_zlim([-0.5, 0.5])
        
        # Add distance info
        distance = np.linalg.norm(end_pos - target_pos)
        self.ax.text2D(0.05, 0.95, f'Distance: {distance:.3f}m', 
                      transform=self.ax.transAxes, fontsize=12,
                      bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        
        self.ax.legend()
        self.ax.set_title('4-DOF Robot Arm Visualization', fontsize=14, fontweight='bold')
    
    def animate_episode(self, episode_length: int = 200, save_gif: bool = False):
        """
        Animate one episode of robot movement.
        
        Args:
            episode_length: Number of steps to animate
            save_gif: Whether to save animation as GIF
        """
        print("🎬 Starting Robot Animation")
        print("=" * 50)
        
        # Reset environment
        state, info = self.env.reset()
        target_pos = info.get('target_position', np.array([0.5, 0.5, 0]))
        
        # Storage for animation
        joint_angles_history = []
        end_effector_path = []
        rewards = []
        
        for step in range(episode_length):
            # Get action from agent or random
            if self.agent is not None:
                action = self.agent.act(state, add_noise=False)
            else:
                action = self.env.action_space.sample()
            
            # Take step
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            # Store data
            current_angles = info.get('joint_angles', np.zeros(4))
            joint_angles_history.append(current_angles.copy())
            
            # Compute end effector position
            positions = self.forward_kinematics(current_angles)
            end_effector_path.append(positions[-1].copy())
            rewards.append(reward)
            
            # Update visualization
            if step % 10 == 0:  # Update every 10 steps for performance
                self.plot_robot_frame(current_angles, target_pos, end_effector_path)
                plt.pause(0.01)
                print(f"Step {step:3d} | Reward: {reward:6.1f} | "
                      f"Distance: {info.get('distance_to_target', 0):.3f}m")
            
            if terminated or truncated:
                print(f"Episode finished at step {step}")
                break
                
            state = next_state
        
        # Final visualization
        self.plot_robot_frame(joint_angles_history[-1], target_pos, end_effector_path)
        plt.show()
        
        # Print summary
        success = info.get('goal_reached', False)
        total_reward = sum(rewards)
        final_distance = info.get('distance_to_target', float('inf'))
        
        print(f"\n🎯 Episode Summary:")
        print(f"   Success: {'✅ Yes' if success else '❌ No'}")
        print(f"   Steps: {len(rewards)}")
        print(f"   Total Reward: {total_reward:.1f}")
        print(f"   Final Distance: {final_distance:.3f}m")
        
        return {
            'success': success,
            'steps': len(rewards),
            'total_reward': total_reward,
            'final_distance': final_distance,
            'joint_angles': joint_angles_history,
            'end_effector_path': end_effector_path
        }

def load_trained_agent(model_path: str, env: Robot4DOFEnv) -> Optional[DDPGAgent]:
    """
    Load a trained DDPG agent.
    
    Args:
        model_path: Path to saved model
        env: Environment instance
        
    Returns:
        Loaded agent or None if loading fails
    """
    try:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        agent = DDPGAgent(state_dim=state_dim, action_dim=action_dim)
        agent.load(model_path)
        print(f"✅ Loaded trained agent from {model_path}")
        return agent
    except Exception as e:
        print(f"❌ Failed to load agent: {e}")
        return None

def visualize_random_policy(episodes: int = 3):
    """Visualize robot with random policy."""
    print("🎲 Visualizing Random Policy")
    
    env = Robot4DOFEnv()
    visualizer = RobotVisualizer(env)
    
    for episode in range(episodes):
        print(f"\n🎯 Episode {episode + 1}/{episodes}")
        result = visualizer.animate_episode(episode_length=100)
    
    env.close()

def visualize_trained_policy(model_path: str = "checkpoints/curriculum_final", episodes: int = 3):
    """Visualize robot with trained policy."""
    print(f"🤖 Visualizing Trained Policy: {model_path}")
    
    env = Robot4DOFEnv()
    agent = load_trained_agent(model_path, env)
    
    if agent is None:
        print("⚠️ Using random policy instead")
        agent = None
    
    visualizer = RobotVisualizer(env, agent)
    
    for episode in range(episodes):
        print(f"\n🎯 Episode {episode + 1}/{episodes}")
        result = visualizer.animate_episode(episode_length=200)
    
    env.close()

def compare_policies():
    """Compare random vs trained policies side by side."""
    print("📊 Comparing Random vs Trained Policies")
    
    # Create subplots
    fig = plt.figure(figsize=(15, 6))
    
    # Random policy
    ax1 = fig.add_subplot(121, projection='3d')
    env1 = Robot4DOFEnv()
    viz1 = RobotVisualizer(env1)
    viz1.ax = ax1
    
    # Trained policy  
    ax2 = fig.add_subplot(122, projection='3d')
    env2 = Robot4DOFEnv()
    agent2 = load_trained_agent("checkpoints/curriculum_final", env2)
    viz2 = RobotVisualizer(env2, agent2)
    viz2.ax = ax2
    
    # Run episodes
    print("Running comparison...")
    
    # Random episode
    state1, info1 = env1.reset()
    target1 = info1.get('target_position', np.array([0.5, 0.5, 0]))
    path1 = []
    
    for step in range(50):
        action1 = env1.action_space.sample()
        state1, _, terminated1, truncated1, info1 = env1.step(action1)
        angles1 = info1.get('joint_angles', np.zeros(4))
        positions1 = viz1.forward_kinematics(angles1)
        path1.append(positions1[-1])
        if terminated1 or truncated1:
            break
    
    viz1.plot_robot_frame(angles1, target1, path1)
    ax1.set_title("Random Policy", fontsize=12, fontweight='bold')
    
    # Trained episode
    if agent2:
        state2, info2 = env2.reset()
        target2 = info2.get('target_position', np.array([0.5, 0.5, 0]))
        path2 = []
        
        for step in range(50):
            action2 = agent2.act(state2, add_noise=False)
            state2, _, terminated2, truncated2, info2 = env2.step(action2)
            angles2 = info2.get('joint_angles', np.zeros(4))
            positions2 = viz2.forward_kinematics(angles2)
            path2.append(positions2[-1])
            if terminated2 or truncated2:
                break
        
        viz2.plot_robot_frame(angles2, target2, path2)
        ax2.set_title("Trained Policy", fontsize=12, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 0.5, "No trained model available", 
                ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    plt.show()
    
    env1.close()
    env2.close()

def main():
    """Main visualization function."""
    print("🎨 4-DOF Robot Arm Visualization Tool")
    print("=" * 50)
    
    while True:
        print("\n📋 Select visualization option:")
        print("1. 🎲 Random Policy (No training)")
        print("2. 🤖 Trained Policy (Load saved model)")
        print("3. 📊 Compare Random vs Trained")
        print("4. 🚪 Exit")
        
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                episodes = int(input("Number of episodes (default 3): ") or "3")
                visualize_random_policy(episodes)
                
            elif choice == '2':
                model_path = input("Model path (default: checkpoints/curriculum_final): ").strip()
                if not model_path:
                    model_path = "checkpoints/curriculum_final"
                episodes = int(input("Number of episodes (default 3): ") or "3")
                visualize_trained_policy(model_path, episodes)
                
            elif choice == '3':
                compare_policies()
                
            elif choice == '4':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1-4.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()