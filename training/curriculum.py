"""
Curriculum Learning Implementation for Robot Arm Training
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from environments.robot_4dof_env import Robot4DOFEnv
from agents.base_agent import BaseAgent

class CurriculumStage:
    """
    Represents a single curriculum learning stage.
    """
    
    def __init__(self, name: str, episodes: int, config: Dict[str, Any], 
                 success_threshold: float = 0.4, min_episodes: int = None):
        """
        Initialize curriculum stage.
        
        Args:
            name: Stage name
            episodes: Maximum episodes for this stage
            config: Environment configuration for this stage
            success_threshold: Success rate to advance to next stage
            min_episodes: Minimum episodes before checking advancement
        """
        self.name = name
        self.episodes = episodes
        self.config = config
        self.success_threshold = success_threshold
        self.min_episodes = min_episodes or episodes // 2
        
        # Statistics
        self.completed_episodes = 0
        self.successes = 0
        self.rewards = []
        self.distances = []
        
    def update_stats(self, reward: float, success: bool, distance: float):
        """Update stage statistics."""
        self.completed_episodes += 1
        self.rewards.append(reward)
        self.distances.append(distance)
        if success:
            self.successes += 1
    
    def get_success_rate(self) -> float:
        """Get current success rate."""
        return self.successes / max(1, self.completed_episodes)
    
    def should_advance(self) -> bool:
        """Check if should advance to next stage."""
        if self.completed_episodes < self.min_episodes:
            return False
        
        # Check success rate over recent episodes
        recent_window = min(50, self.completed_episodes)
        recent_successes = sum(1 for i in range(-recent_window, 0) 
                             if i < 0 and self.completed_episodes + i >= 0)  # Placeholder logic
        
        recent_success_rate = self.get_success_rate()
        return recent_success_rate >= self.success_threshold
    
    def is_complete(self) -> bool:
        """Check if stage is complete."""
        return self.completed_episodes >= self.episodes or self.should_advance()

class CurriculumTrainer:
    """
    Curriculum Learning trainer for progressive difficulty training.
    
    Implements automatic progression through difficulty stages based on
    performance metrics and success thresholds.
    """
    
    def __init__(self, base_env_config: Dict[str, Any], agent: BaseAgent, 
                 curriculum_config: Optional[Dict[str, Any]] = None):
        """
        Initialize curriculum trainer.
        
        Args:
            base_env_config: Base environment configuration
            agent: RL agent to train
            curriculum_config: Curriculum-specific configuration
        """
        self.base_env_config = base_env_config
        self.agent = agent
        self.curriculum_config = curriculum_config or {}
        
        # Create curriculum stages
        self.stages = self._create_curriculum_stages()
        self.current_stage_idx = 0
        self.current_stage = self.stages[0]
        
        # Training statistics
        self.total_episodes = 0
        self.total_successes = 0
        self.stage_history = []
        
        print(f"🎯 Initialized Curriculum Learning with {len(self.stages)} stages")
        for i, stage in enumerate(self.stages):
            print(f"   Stage {i+1}: {stage.name} ({stage.episodes} episodes)")
    
    def _create_curriculum_stages(self) -> List[CurriculumStage]:
        """Create curriculum learning stages."""
        stages = []
        
        # Stage 1: Easy positioning (large target, close distance)
        stage1_config = self.base_env_config.copy()
        stage1_config.update({
            'success_distance': 0.15,  # 15cm tolerance
            'workspace_radius': 0.5,   # Smaller workspace
            'success_reward': 50.0,
            'dense_reward': True
        })
        stages.append(CurriculumStage(
            name="Basic Positioning",
            episodes=50,
            config=stage1_config,
            success_threshold=0.6,
            min_episodes=20
        ))
        
        # Stage 2: Intermediate positioning
        stage2_config = self.base_env_config.copy()
        stage2_config.update({
            'success_distance': 0.10,  # 10cm tolerance
            'workspace_radius': 0.6,
            'success_reward': 75.0,
            'dense_reward': True
        })
        stages.append(CurriculumStage(
            name="Intermediate Precision",
            episodes=75,
            config=stage2_config,
            success_threshold=0.5,
            min_episodes=30
        ))
        
        # Stage 3: Advanced targeting
        stage3_config = self.base_env_config.copy()
        stage3_config.update({
            'success_distance': 0.07,  # 7cm tolerance
            'workspace_radius': 0.7,
            'success_reward': 100.0,
            'dense_reward': True
        })
        stages.append(CurriculumStage(
            name="Advanced Targeting",
            episodes=100,
            config=stage3_config,
            success_threshold=0.45,
            min_episodes=40
        ))
        
        # Stage 4: Expert precision
        stage4_config = self.base_env_config.copy()
        stage4_config.update({
            'success_distance': 0.05,  # 5cm tolerance (final target)
            'workspace_radius': 0.8,   # Full workspace
            'success_reward': 100.0,
            'dense_reward': True
        })
        stages.append(CurriculumStage(
            name="Expert Precision",
            episodes=75,
            config=stage4_config,
            success_threshold=0.4,
            min_episodes=25
        ))
        
        return stages
    
    def train(self, total_episodes: int = 300, render: bool = False, 
             save_checkpoints: bool = True) -> Dict[str, Any]:
        """
        Train agent with curriculum learning.
        
        Args:
            total_episodes: Maximum total episodes across all stages
            render: Whether to render environment
            save_checkpoints: Whether to save model checkpoints
            
        Returns:
            Training results dictionary
        """
        print(f"🚀 Starting Curriculum Learning Training")
        print(f"Total Episodes: {total_episodes}")
        print("=" * 60)
        
        episode_count = 0
        all_rewards = []
        all_distances = []
        all_success_rates = []
        
        while episode_count < total_episodes and self.current_stage_idx < len(self.stages):
            stage = self.current_stage
            
            print(f"\n📚 Stage {self.current_stage_idx + 1}: {stage.name}")
            print(f"   Target Success Distance: {stage.config['success_distance']*100:.1f}cm")
            print(f"   Workspace Radius: {stage.config['workspace_radius']:.1f}m")
            print(f"   Success Threshold: {stage.success_threshold:.1%}")
            
            # Create environment for current stage
            env = Robot4DOFEnv(config=stage.config)
            
            # Train on current stage
            stage_results = self._train_stage(
                env, stage, 
                max_episodes=min(stage.episodes, total_episodes - episode_count),
                render=render
            )
            
            # Update global statistics
            episode_count += stage_results['episodes']
            all_rewards.extend(stage_results['rewards'])
            all_distances.extend(stage_results['distances'])
            all_success_rates.extend(stage_results['success_rates'])
            
            # Save checkpoint
            if save_checkpoints:
                import os
                os.makedirs('checkpoints', exist_ok=True)
                checkpoint_path = f'checkpoints/curriculum_stage_{self.current_stage_idx + 1}'
                self.agent.save_model(checkpoint_path)
                print(f"💾 Checkpoint saved: {checkpoint_path}")
            
            # Check if stage is complete
            if stage.is_complete():
                success_rate = stage.get_success_rate()
                print(f"✅ Stage {self.current_stage_idx + 1} completed!")
                print(f"   Episodes: {stage.completed_episodes}")
                print(f"   Success Rate: {success_rate:.1%}")
                print(f"   Avg Reward: {np.mean(stage.rewards):.1f}")
                print(f"   Avg Distance: {np.mean(stage.distances):.3f}m")
                
                # Record stage completion
                self.stage_history.append({
                    'stage_name': stage.name,
                    'episodes': stage.completed_episodes,
                    'success_rate': success_rate,
                    'avg_reward': np.mean(stage.rewards),
                    'avg_distance': np.mean(stage.distances)
                })
                
                # Advance to next stage
                self.current_stage_idx += 1
                if self.current_stage_idx < len(self.stages):
                    self.current_stage = self.stages[self.current_stage_idx]
            else:
                print(f"⏰ Stage {self.current_stage_idx + 1} incomplete (reached episode limit)")
                break
            
            env.close()
        
        # Training complete
        total_success_rate = sum(all_success_rates) / max(1, len(all_success_rates))
        
        print(f"\n🎉 Curriculum Training Complete!")
        print(f"Total Episodes: {episode_count}")
        print(f"Overall Success Rate: {total_success_rate:.1%}")
        print(f"Final Average Distance: {np.mean(all_distances[-100:]):.3f}m")
        
        # Print stage summary
        print(f"\n📊 Stage Summary:")
        for i, stage_info in enumerate(self.stage_history):
            print(f"   Stage {i+1} ({stage_info['stage_name']}): "
                  f"{stage_info['success_rate']:.1%} success, "
                  f"{stage_info['episodes']} episodes")
        
        return {
            'total_episodes': episode_count,
            'all_rewards': all_rewards,
            'all_distances': all_distances,
            'all_success_rates': all_success_rates,
            'overall_success_rate': total_success_rate,
            'stage_history': self.stage_history,
            'completed_stages': len(self.stage_history)
        }
    
    def _train_stage(self, env: Robot4DOFEnv, stage: CurriculumStage, 
                    max_episodes: int, render: bool = False) -> Dict[str, Any]:
        """Train on a specific curriculum stage."""
        
        stage_rewards = []
        stage_distances = []
        stage_success_rates = []
        
        for episode in range(max_episodes):
            state, info = env.reset()
            episode_reward = 0.0
            episode_distances = []
            episode_success = False
            
            for step in range(stage.config.get('max_steps', 200)):
                # Select action
                action = self.agent.act(state, add_noise=True)
                
                # Take step
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Store experience (implement your replay buffer logic here)
                episode_reward += reward
                episode_distances.append(info.get('distance_to_target', float('inf')))
                
                if info.get('goal_reached', False):
                    episode_success = True
                
                # Render if requested
                if render and episode % 20 == 0:
                    env.render(mode='human')
                
                state = next_state
                
                if done:
                    break
            
            # Update stage statistics
            avg_distance = np.mean(episode_distances) if episode_distances else float('inf')
            stage.update_stats(episode_reward, episode_success, avg_distance)
            
            # Update global agent statistics  
            self.agent.update_training_stats(episode_reward, True)
            
            # Store results
            stage_rewards.append(episode_reward)
            stage_distances.append(avg_distance)
            stage_success_rates.append(float(episode_success))
            
            # Print progress
            if episode % 10 == 0:
                recent_success_rate = np.mean(stage_success_rates[-10:])
                status_icon = "✅" if episode_success else "🔄"
                print(f"   {status_icon} Episode {episode:3d} | "
                      f"Reward: {episode_reward:6.1f} | "
                      f"Success Rate: {recent_success_rate:.2%} | "
                      f"Distance: {avg_distance:.3f}m")
            
            # Check early completion
            if stage.should_advance() and episode >= stage.min_episodes:
                print(f"   🎯 Early advancement criteria met at episode {episode}")
                break
        
        return {
            'episodes': min(episode + 1, max_episodes),
            'rewards': stage_rewards,
            'distances': stage_distances,
            'success_rates': stage_success_rates
        }
