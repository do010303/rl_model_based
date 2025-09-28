"""
Project Information and Metadata
"""

__version__ = "1.0.0"
__author__ = "Quan Van Nguyen"
__email__ = "vnquan.hust.200603@gmail.com"
__description__ = "4-DOF Robot Arm Reinforcement Learning Controller"
__license__ = "MIT"

# Project metadata
PROJECT_NAME = "Robotarm-RL-4DoF"
PROJECT_URL = "https://github.com/your-username/Robotarm-RL-4DoF"

# Default configuration
DEFAULT_CONFIG = {
    "environment": {
        "max_steps": 200,
        "success_distance": 0.05,
        "dense_reward": True,
        "workspace_radius": 0.8
    },
    "agent": {
        "lr_actor": 0.001,
        "lr_critic": 0.002,
        "gamma": 0.99,
        "tau": 0.005,
        "noise_std": 0.2,
        "hidden_dims": [256, 128]
    },
    "training": {
        "episodes": 300,
        "batch_size": 64,
        "buffer_size": 100000,
        "curriculum_stages": 4
    }
}
