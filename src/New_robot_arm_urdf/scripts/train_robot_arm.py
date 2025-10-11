#!/usr/bin/env python3

"""
RL Training Script for Robot Arm in Gazebo
This script supports multiple RL algorithms including MBPO integration
"""

import sys
import os
import numpy as np
import rospy
import subprocess
import signal
import time
import argparse
from robot_arm_env import RobotArmEnv

# Add parent RL project to Python path for MBPO integration
sys.path.insert(0, '/home/ducanh/rl_model_based')

class RLTrainer:
    """
    Reinforcement Learning trainer for robot arm
    Supports multiple RL algorithms and libraries
    """
    
    def __init__(self, algorithm='SAC', total_timesteps=100000):
        """
        Initialize trainer
        
        Args:
            algorithm (str): RL algorithm to use ('SAC', 'PPO', 'TD3', etc.)
            total_timesteps (int): Total training steps
        """
        self.algorithm = algorithm
        self.total_timesteps = total_timesteps
        self.gazebo_process = None
        self.env = None
        
    def start_gazebo(self, headless=True):
        """Start Gazebo simulation for training"""
        rospy.loginfo("Starting Gazebo simulation...")
        
        # Launch command
        if headless:
            launch_cmd = [
                'roslaunch', 'New_description', 'rl_training.launch',
                'headless:=true', 'gui:=false', 'paused:=false'
            ]
        else:
            launch_cmd = [
                'roslaunch', 'New_description', 'rl_training.launch', 
                'headless:=false', 'gui:=true', 'paused:=false'
            ]
            
        # Start Gazebo process
        self.gazebo_process = subprocess.Popen(launch_cmd)
        
        # Wait for Gazebo to start
        time.sleep(10)
        rospy.loginfo("Gazebo started!")
        
    def stop_gazebo(self):
        """Stop Gazebo simulation"""
        if self.gazebo_process:
            rospy.loginfo("Stopping Gazebo...")
            self.gazebo_process.terminate()
            self.gazebo_process.wait()
            
        # Kill any remaining Gazebo processes
        subprocess.run(['pkill', '-f', 'gzserver'], stderr=subprocess.DEVNULL)
        subprocess.run(['pkill', '-f', 'gzclient'], stderr=subprocess.DEVNULL)
        
    def create_environment(self):
        """Create the robot arm environment"""
        rospy.loginfo("Creating robot arm environment...")
        self.env = RobotArmEnv()
        rospy.loginfo("Environment created!")
        
    def train_with_stable_baselines3(self):
        """
        Train using Stable Baselines3 (Popular RL library)
        Install with: pip install stable-baselines3[extra]
        """
        try:
            from stable_baselines3 import SAC, PPO, TD3
            from stable_baselines3.common.env_util import make_vec_env
            from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
            from stable_baselines3.common.monitor import Monitor
            import gym
            from gym import spaces
            
        except ImportError:
            rospy.logerr("Stable Baselines3 not installed!")
            rospy.logerr("Install with: pip install stable-baselines3[extra]")
            return
            
        # Wrapper to make RobotArmEnv compatible with gym
        class GymWrapper(gym.Env):
            def __init__(self, ros_env):
                super(GymWrapper, self).__init__()
                self.ros_env = ros_env
                
                # Define action and observation spaces
                self.action_space = spaces.Box(
                    low=-np.pi, high=np.pi, 
                    shape=(4,), dtype=np.float32
                )
                
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(14,), dtype=np.float32  # 4 pos + 4 vel + 3 target + 3 ee_pos
                )
                
            def reset(self):
                return self.ros_env.reset()
                
            def step(self, action):
                return self.ros_env.step(action)
                
            def close(self):
                self.ros_env.close()
                
        # Create wrapped environment
        gym_env = GymWrapper(self.env)
        gym_env = Monitor(gym_env, './logs/')
        
        # Select algorithm
        if self.algorithm == 'SAC':
            model = SAC(
                'MlpPolicy', 
                gym_env,
                verbose=1,
                tensorboard_log="./tensorboard_logs/",
                learning_rate=3e-4,
                buffer_size=100000,
                batch_size=256,
                gamma=0.99,
                tau=0.005
            )
        elif self.algorithm == 'PPO':
            model = PPO(
                'MlpPolicy',
                gym_env, 
                verbose=1,
                tensorboard_log="./tensorboard_logs/",
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                gamma=0.99
            )
        elif self.algorithm == 'TD3':
            model = TD3(
                'MlpPolicy',
                gym_env,
                verbose=1, 
                tensorboard_log="./tensorboard_logs/",
                learning_rate=3e-4,
                buffer_size=100000,
                batch_size=256,
                gamma=0.99
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
            
        # Setup callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path='./models/',
            name_prefix=f'{self.algorithm}_robot_arm'
        )
        
        # Train the model
        rospy.loginfo(f"Starting training with {self.algorithm}...")
        model.learn(
            total_timesteps=self.total_timesteps,
            callback=checkpoint_callback,
            tb_log_name=f"{self.algorithm}_robot_arm"
        )
        
        # Save final model
        model.save(f"./models/{self.algorithm}_robot_arm_final")
        rospy.loginfo("Training completed!")
        
        return model
    
    def train_with_mbpo(self):
        """
        Train using your existing MBPO framework directly
        This uses the original MBPOTrainer from the parent directory
        """
        try:
            from ros_mbpo_integration import ROSMBPOIntegration
            
            rospy.loginfo("Starting MBPO training with existing framework...")
            
            # Environment configuration for ROS-Gazebo adapter
            env_config = {
                'headless': True,
                'real_time': False,
                'max_steps': 200,
                'success_distance': 0.02,
                'dense_reward': True
            }
            
            # Agent configuration (optimized for robot arm)
            agent_config = {
                'lr_actor': 0.0001,
                'lr_critic': 0.001,
                'gamma': 0.99,
                'tau': 0.005,
                'noise_std': 0.2,
                'noise_decay': 0.9995,
                'hidden_dims': [512, 256, 128]
            }
            
            # Calculate episodes from timesteps (assuming ~200 steps per episode)
            episodes = max(100, self.total_timesteps // 200)
            
            rospy.loginfo(f"Using your existing MBPO framework")
            rospy.loginfo(f"Running for {episodes} episodes (~{self.total_timesteps} timesteps)")
            
            # Create integration with your existing MBPO trainer
            integration = ROSMBPOIntegration(env_config, agent_config)
            
            # Run training using your existing framework
            integration.train(
                episodes=episodes,
                max_steps=200,
                rollout_every=10  # Model rollouts every 10 episodes
            )
            
            return integration
            
        except ImportError as e:
            rospy.logerr(f"MBPO integration failed: {e}")
            rospy.logerr("Make sure the parent directory MBPO components are accessible")
            import traceback
            traceback.print_exc()
            return None
        except Exception as e:
            rospy.logerr(f"MBPO training error: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def train_with_custom_algorithm(self):
        """
        Train using custom RL algorithm
        This is where you'd integrate with your parent RL project
        """
        rospy.loginfo("Training with custom algorithm...")
        
        # Example training loop
        for episode in range(1000):
            obs = self.env.reset()
            total_reward = 0
            done = False
            step = 0
            
            while not done and step < 1000:
                # Your RL algorithm would compute action here
                # For now, using random actions as example
                action = np.random.uniform(-0.1, 0.1, 4)
                
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                step += 1
                
                # Your learning update would go here
                # e.g., update_policy(obs, action, reward, next_obs, done)
                
            rospy.loginfo(f"Episode {episode}: Reward = {total_reward:.2f}, Steps = {step}")
            
            # Save model periodically
            if episode % 100 == 0:
                rospy.loginfo(f"Saving model at episode {episode}")
                # Save your model here
                
    def test_trained_model(self, model_path):
        """Test a trained model"""
        try:
            from stable_baselines3 import SAC, PPO, TD3
            
            # Load model based on algorithm
            if 'SAC' in model_path:
                model = SAC.load(model_path)
            elif 'PPO' in model_path:
                model = PPO.load(model_path)
            elif 'TD3' in model_path:
                model = TD3.load(model_path)
            else:
                rospy.logerr("Unknown model type")
                return
                
            rospy.loginfo("Testing trained model...")
            
            # Test episodes
            for episode in range(10):
                obs = self.env.reset()
                total_reward = 0
                done = False
                step = 0
                
                while not done and step < 1000:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.env.step(action)
                    total_reward += reward
                    step += 1
                    
                    if done:
                        rospy.loginfo(f"Target reached in {step} steps!")
                        
                rospy.loginfo(f"Test Episode {episode}: Reward = {total_reward:.2f}")
                
        except ImportError:
            rospy.logerr("Model testing requires stable-baselines3")
            
    def run(self, mode='train', model_path=None, headless=True):
        """
        Run training or testing
        
        Args:
            mode (str): 'train' or 'test'
            model_path (str): Path to model for testing
            headless (bool): Run Gazebo without GUI
        """
        try:
            # Setup signal handler for clean shutdown
            def signal_handler(sig, frame):
                rospy.loginfo("Shutting down...")
                if self.env:
                    self.env.close()
                self.stop_gazebo()
                sys.exit(0)
                
            signal.signal(signal.SIGINT, signal_handler)
            
            # Start Gazebo
            self.start_gazebo(headless=headless)
            
            # Create environment  
            self.create_environment()
            
            if mode == 'train':
                # Choose training method
                if self.algorithm in ['SAC', 'PPO', 'TD3']:
                    self.train_with_stable_baselines3()
                elif self.algorithm == 'MBPO':
                    self.train_with_mbpo()
                else:
                    self.train_with_custom_algorithm()
            elif mode == 'test':
                if model_path:
                    self.test_trained_model(model_path)
                else:
                    rospy.logerr("Model path required for testing")
            else:
                rospy.logerr(f"Unknown mode: {mode}")
                
        except Exception as e:
            rospy.logerr(f"Training error: {e}")
        finally:
            # Cleanup
            if self.env:
                self.env.close()
            self.stop_gazebo()


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Robot Arm RL Training')
    parser.add_argument('--algorithm', default='SAC', choices=['SAC', 'PPO', 'TD3', 'MBPO', 'custom'],
                      help='RL algorithm to use')
    parser.add_argument('--timesteps', type=int, default=100000,
                      help='Total training timesteps') 
    parser.add_argument('--mode', default='train', choices=['train', 'test'],
                      help='Training or testing mode')
    parser.add_argument('--model-path', type=str, default=None,
                      help='Path to trained model for testing')
    parser.add_argument('--gui', action='store_true',
                      help='Show Gazebo GUI (slower training)')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = RLTrainer(
        algorithm=args.algorithm,
        total_timesteps=args.timesteps
    )
    
    # Run training or testing
    trainer.run(
        mode=args.mode,
        model_path=args.model_path,
        headless=not args.gui
    )


if __name__ == '__main__':
    main()