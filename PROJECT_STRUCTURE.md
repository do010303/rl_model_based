# 🏗️ Project Structure

This document describes the organized structure of the RL Model-Based project.

## 📁 Directory Structure

```
rl_model_based/
├── 📚 docs/                           # Documentation
│   ├── README.md                      # Main project documentation
│   ├── CONTRIBUTING.md                # Contribution guidelines
│   ├── LICENSE                        # License information
│   ├── REPORT_09_10_REINFORCEMENT_LEARNING.md
│   └── TRAINING_IMPROVEMENTS.md       # Training methodology improvements
│
├── 🧠 agents/                         # RL Agents
│   ├── base_agent.py                  # Base agent interface
│   └── ddpg_agent.py                  # DDPG implementation
│
├── 🌍 environments/                   # Training Environments
│   └── robot_4dof_env.py              # 4-DOF robot arm environment
│
├── 📝 examples/                       # Training Scripts
│   ├── train_ddpg.py                  # Main training script (DDPG/MBPO)
│   ├── train_curriculum.py            # Curriculum learning
│   ├── test_model.py                  # Model testing utilities
│   └── visualize_robot.py             # Visualization tools
│
├── 🤖 models/                         # Neural Network Models
│   └── dynamics_model.py              # World dynamics model
│
├── 💾 replay_memory/                  # Experience Replay
│   └── replay_buffer.py               # Smart cleanup replay buffer
│
├── 🎯 training/                       # Training Utilities
│   ├── curriculum.py                  # Curriculum learning logic
│   └── __init__.py
│
├── 🛠️ utils/                          # Utilities
│   ├── her.py                         # Hindsight Experience Replay
│   └── early_stopping.py             # Early stopping utilities
│
├── 💾 checkpoints/                    # Model Checkpoints (Organized)
│   ├── ddpg/                          # DDPG model checkpoints
│   │   ├── ddpg_4dof_actor.h5
│   │   ├── ddpg_4dof_critic.h5
│   │   └── ddpg_4dof_config.json
│   ├── mbpo/                          # MBPO model checkpoints
│   │   ├── mbpo_4dof_actor.h5
│   │   ├── mbpo_4dof_critic.h5
│   │   └── mbpo_4dof_config.json
│   ├── curriculum/                    # Curriculum learning checkpoints
│   │   └── curriculum_*.h5
│   └── replay_buffers/                # Saved replay buffers
│       ├── replay_buffer.pkl          # DDPG replay buffer
│       └── mbpo_replay_buffer.pkl     # MBPO replay buffer
│
├── 📊 logs/                          # Training Logs & Results
│   ├── training/                      # Training logs
│   └── results/                       # Result plots and metrics
│       └── mbpo_training_results.png
│
├── 🔧 configs/                       # Configuration Files
│   ├── requirements.txt               # Python dependencies
│   └── setup.sh                      # Environment setup script
│
├── 📜 scripts/                       # Utility Scripts
│   ├── demo.py                       # Demo script
│   └── test_nan_prevention.py        # Testing utilities
│
├── 🤖 src/                           # ROS Integration (Future Gazebo)
│   └── New_robot_arm_urdf/           # Robot URDF files for Gazebo
│
└── 📄 Core Files
    ├── mbpo_trainer.py               # MBPO trainer implementation
    └── __init__.py                   # Package initialization
```

## 🚀 Quick Start

### Training with MBPO (Recommended)
```bash
python3 examples/train_ddpg.py --episodes 300 --method mbpo
```

### Training with DDPG
```bash
python3 examples/train_ddpg.py --episodes 300 --method ddpg
```

## 📋 Key Features

### ✨ Smart Buffer Management
- **Success-prioritized cleanup** preserves high-reward experiences
- **Competitive buffer cleanup** maintains training quality
- **Dynamic capacity management** prevents memory overflow

### 🧠 Advanced RL Algorithms
- **MBPO** (Model-Based Policy Optimization) with dynamics model
- **DDPG** (Deep Deterministic Policy Gradient) 
- **Curriculum Learning** support
- **HER** (Hindsight Experience Replay) integration

### 🎯 Performance Optimizations
- **NaN prevention** in dynamics models
- **Gradient clipping** for stability
- **Batch normalization** for faster convergence
- **Success rate tracking** and early convergence detection

## 📊 Results Location

All training results are saved in organized locations:
- **Model checkpoints**: `checkpoints/{algorithm}/`
- **Replay buffers**: `checkpoints/replay_buffers/`
- **Training plots**: `logs/results/`
- **Training logs**: `logs/training/`

## 🔧 Configuration

Project configuration files are located in:
- `configs/requirements.txt` - Python dependencies
- `configs/setup.sh` - Environment setup

## 📚 Documentation

All documentation is organized in the `docs/` folder:
- Main README with usage instructions
- Training methodology improvements
- Performance analysis reports