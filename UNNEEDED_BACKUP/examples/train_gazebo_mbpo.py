#!/usr/bin/env python3
"""
Example training script for Gazebo MBPO integration
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from gazebo_mbpo_trainer import GazeboMBPOTrainer

# Import training configurations
TRAINING_CONFIGS = {
    "quick_test": {
        "num_episodes": 5,
        "max_episode_steps": 50,
        "model_ensemble_size": 3,
        "model_hidden_size": 64,
        "policy_hidden_size": 64,
        "value_hidden_size": 64,
        "batch_size": 32,
        "model_train_freq": 10,
        "policy_train_freq": 5,
        "save_freq": 5,
        "log_freq": 1,
        "target_return": -50,
    },
    "development": {
        "num_episodes": 100,
        "max_episode_steps": 200,
        "model_ensemble_size": 5,
        "model_hidden_size": 128,
        "policy_hidden_size": 128,
        "value_hidden_size": 128,
        "batch_size": 64,
        "model_train_freq": 25,
        "policy_train_freq": 10,
        "save_freq": 20,
        "log_freq": 5,
        "target_return": -100,
    },
    "production": {
        "num_episodes": 1000,
        "max_episode_steps": 500,
        "model_ensemble_size": 7,
        "model_hidden_size": 256,
        "policy_hidden_size": 256,
        "value_hidden_size": 256,
        "batch_size": 128,
        "model_train_freq": 50,
        "policy_train_freq": 25,
        "save_freq": 50,
        "log_freq": 10,
        "target_return": -50,
    }
}

def create_training_config(config_name="development"):
    """Create training configuration based on preset"""
    if config_name not in TRAINING_CONFIGS:
        print(f"Warning: Unknown config {config_name}, using development")
        config_name = "development"
    
    base_config = TRAINING_CONFIGS[config_name]
    
    return {
        # Training parameters from preset
        "num_episodes": base_config["num_episodes"],
        "max_episode_steps": base_config["max_episode_steps"],
        "batch_size": base_config["batch_size"],
        "save_freq": base_config["save_freq"],
        "log_freq": base_config["log_freq"],
        
        # Model-based parameters
        "model_ensemble_size": base_config["model_ensemble_size"],
        "model_hidden_size": base_config["model_hidden_size"],
        "rollout_length": 5,
        "num_model_rollouts": 400,
        "real_ratio": 0.25,
        "model_train_freq": base_config["model_train_freq"],
        
        # Policy parameters
        "policy_hidden_size": base_config["policy_hidden_size"],
        "value_hidden_size": base_config["value_hidden_size"],
        "policy_train_freq": base_config["policy_train_freq"],
        
        # Gazebo settings
        "auto_manage_gazebo": False,  # We manage manually
        "headless": True,
        
        # Environment configuration
        "env_config": {
            "max_steps": base_config["max_episode_steps"],
            "success_distance": 0.02,
            "workspace_radius": 0.15,
            "dense_reward": True,
            "success_reward": 100.0,
            "distance_reward_scale": -1.0,
            "action_penalty_scale": -0.01
        },
        
        # Agent configuration
        "agent_config": {
            "lr_actor": 0.0005,
            "lr_critic": 0.001,
            "gamma": 0.99,
            "tau": 0.005,
            "noise_std": 0.2,
            "noise_decay": 0.995,
            "hidden_dims": [256, 128, 64]
        },
        
        # Model configuration
        "model_config": {
            "hidden_dims": [256, 256, 128],
            "learning_rate": 0.001,
            "ensemble_size": 5
        },
        
        # Buffer configuration
        "buffer_config": {
            "capacity": 100000,
            "prioritized": False
        },
        
        # Early stopping
        "early_stopping": {
            "patience": 20,
            "min_delta": 0.001,
            "monitor": "distance"
        }
    }

def train_gazebo_mbpo(config_path=None, **kwargs):
    """Train MBPO with Gazebo simulation"""
    
    # Load or create configuration
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded config from {config_path}")
    else:
        config = create_training_config()
        print("Using default configuration")
    
    # Override with command line arguments
    for key, value in kwargs.items():
        if key in config and value is not None:
            config[key] = value
            print(f"Override: {key} = {value}")
    
    # Save config for reference
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_save_path = f"/home/ducanh/rl_model_based/logs/training/gazebo_mbpo_config_{timestamp}.json"
    os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
    
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {config_save_path}")
    
    # Create and run trainer
    print("🚀 Starting Gazebo MBPO training...")
    print(f"📊 Training config: {json.dumps(config, indent=2)}")
    
    trainer = GazeboMBPOTrainer(config)
    success = trainer.train()
    
    if success:
        print("✅ Training completed successfully!")
        
        # Save final checkpoint
        trainer.save_checkpoint(f"final_{timestamp}")
        
        # Print training summary
        stats = trainer.training_stats
        if stats['epoch_rewards']:
            print(f"📈 Final average reward: {stats['epoch_rewards'][-1]:.2f}")
            print(f"📈 Best reward: {max(stats['epoch_rewards']):.2f}")
        if stats['epoch_success_rates']:
            print(f"🎯 Final success rate: {stats['epoch_success_rates'][-1]:.1%}")
            print(f"🎯 Best success rate: {max(stats['epoch_success_rates']):.1%}")
    else:
        print("❌ Training failed!")
    
    return success

def test_gazebo_connection():
    """Test Gazebo connection without full training"""
    print("🧪 Testing Gazebo connection...")
    
    config = {
        "auto_manage_gazebo": True,
        "headless": True,
        "num_epochs": 1,
        "steps_per_epoch": 10,
        "env_config": {"max_steps": 10}
    }
    
    trainer = GazeboMBPOTrainer(config)
    
    try:
        # Test Gazebo startup
        if not trainer.setup_gazebo():
            print("❌ Failed to start Gazebo")
            return False
        
        # Test component initialization
        if not trainer.initialize_components():
            print("❌ Failed to initialize components")
            return False
        
        # Test data collection
        print("Testing data collection...")
        reward, success, distance = trainer.collect_real_data(10)
        print(f"✅ Test episode: reward={reward:.2f}, success={success}, distance={distance:.3f}")
        
        print("✅ Gazebo connection test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False
    finally:
        trainer.cleanup()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Gazebo MBPO Training Examples")
    
    # Training mode selection
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                       help='Mode: train or test connection')
    
    # Configuration
    parser.add_argument('--config', type=str, choices=list(TRAINING_CONFIGS.keys()) + ['custom'], 
                       default='development', help='Training configuration preset')
    parser.add_argument('--config-file', type=str, help='Path to custom configuration file')
    
    # Training parameters  
    parser.add_argument('--episodes', type=int, help='Number of training episodes')
    parser.add_argument('--headless', action='store_true', default=True, help='Run without GUI')
    parser.add_argument('--gui', action='store_false', dest='headless', help='Run with GUI')
    
    # Environment parameters
    parser.add_argument('--success-distance', type=float, help='Success distance threshold')
    parser.add_argument('--workspace-radius', type=float, help='Workspace radius')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        # Test connection
        success = test_gazebo_connection()
    else:
        # Full training
        print(f"🚀 Starting MBPO training with '{args.config}' configuration")
        
        if args.config_file:
            success = train_gazebo_mbpo(args.config_file)
        else:
            # Use preset configuration
            config = create_training_config(args.config)
            
            # Apply command line overrides
            if args.episodes is not None:
                config['num_episodes'] = args.episodes
            if args.headless is not None:
                config['headless'] = args.headless
                
            # Environment overrides
            if args.success_distance is not None:
                config['env_config']['success_distance'] = args.success_distance
            if args.workspace_radius is not None:
                config['env_config']['workspace_radius'] = args.workspace_radius
            
            success = train_gazebo_mbpo_with_config(config)
    
    return 0 if success else 1

def train_gazebo_mbpo_with_config(config):
    """Train MBPO with provided configuration"""
    try:
        print(f"📋 Training Configuration:")
        print(f"  Episodes: {config['num_episodes']}")
        print(f"  Max steps per episode: {config['max_episode_steps']}")
        print(f"  Model ensemble size: {config['model_ensemble_size']}")
        print(f"  Batch size: {config['batch_size']}")
        
        # Create trainer
        trainer = GazeboMBPOTrainer(
            workspace_path="/home/ducanh/rl_model_based/robot_ws",
            use_gui=not config.get('headless', True),
            real_time=False
        )
        
        # Train
        results = trainer.train(config)
        
        print("🎉 Training completed successfully!")
        print(f"📊 Final results: {results}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    exit(main())