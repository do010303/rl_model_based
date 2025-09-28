"""
Model Testing and Evaluation Script for 4-DOF Robot Arm
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.robot_4dof_env import Robot4DOFEnv
from agents.ddpg_agent import DDPGAgent

def test_model(model_path: str, num_episodes: int = 100, render: bool = True,
              save_results: bool = True) -> Dict:
    """
    Test a trained model and evaluate its performance.
    
    Args:
        model_path: Path to the saved model
        num_episodes: Number of test episodes
        render: Whether to render the environment
        save_results: Whether to save detailed results
        
    Returns:
        Dictionary containing test results
    """
    
    print(f"🧪 Testing Model: {model_path}")
    print(f"Test Episodes: {num_episodes}")
    print("=" * 60)
    
    # Environment configuration (same as training)
    env_config = {
        'max_steps': 200,
        'success_distance': 0.05,
        'workspace_radius': 0.8,
        'dense_reward': True,
        'success_reward': 100.0
    }
    
    # Initialize environment
    env = Robot4DOFEnv(config=env_config)
    
    # Initialize agent and load model
    agent = DDPGAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0]
    )
    
    try:
        agent.load_model(model_path)
        print(f"✅ Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return {}
    
    # Test statistics
    episode_rewards = []
    episode_distances = []
    episode_successes = []
    episode_steps = []
    final_distances = []
    
    # Spatial analysis
    start_positions = []
    end_positions = []
    target_positions = []
    
    print(f"\n🚀 Starting {num_episodes} test episodes...")
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0.0
        step_count = 0
        episode_distances_list = []
        episode_success = False
        
        # Record initial positions
        initial_ee_pos = info['end_effector_position']
        target_pos = info['target_position']
        start_positions.append(initial_ee_pos)
        target_positions.append(target_pos)
        
        # Episode loop
        for step in range(env_config['max_steps']):
            # Select action (no exploration noise during testing)
            action = agent.act(state, add_noise=False)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update statistics
            episode_reward += reward
            distance = info.get('distance_to_target', float('inf'))
            episode_distances_list.append(distance)
            step_count += 1
            
            # Check success
            if info.get('goal_reached', False):
                episode_success = True
            
            # Render if requested
            if render and episode < 5:  # Only render first few episodes
                env.render(mode='human')
                time.sleep(0.05)  # Slow down for visualization
            
            state = next_state
            
            if done:
                break
        
        # Record episode results
        final_ee_pos = info['end_effector_position']
        final_distance = info.get('distance_to_target', float('inf'))
        
        episode_rewards.append(episode_reward)
        episode_distances.append(np.mean(episode_distances_list))
        episode_successes.append(episode_success)
        episode_steps.append(step_count)
        final_distances.append(final_distance)
        end_positions.append(final_ee_pos)
        
        # Print progress
        if episode % 20 == 0 or episode_success:
            success_icon = "✅" if episode_success else "❌"
            print(f"{success_icon} Episode {episode:3d}/{num_episodes} | "
                  f"Reward: {episode_reward:6.1f} | "
                  f"Steps: {step_count:3d} | "
                  f"Final Distance: {final_distance:.3f}m")
    
    env.close()
    
    # Calculate comprehensive statistics
    success_count = sum(episode_successes)
    success_rate = success_count / num_episodes
    avg_reward = np.mean(episode_rewards)
    avg_distance = np.mean(episode_distances)
    avg_final_distance = np.mean(final_distances)
    avg_steps = np.mean(episode_steps)
    
    # Distance statistics
    successful_distances = [d for d, s in zip(final_distances, episode_successes) if s]
    failed_distances = [d for d, s in zip(final_distances, episode_successes) if not s]
    
    results = {
        'model_path': model_path,
        'num_episodes': num_episodes,
        'success_count': success_count,
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'avg_distance': avg_distance,
        'avg_final_distance': avg_final_distance,
        'avg_steps': avg_steps,
        'episode_rewards': episode_rewards,
        'episode_distances': episode_distances,
        'episode_successes': episode_successes,
        'final_distances': final_distances,
        'successful_distances': successful_distances,
        'failed_distances': failed_distances,
        'start_positions': start_positions,
        'end_positions': end_positions,
        'target_positions': target_positions
    }
    
    # Print summary
    print(f"\n📊 TEST RESULTS SUMMARY")
    print("=" * 40)
    print(f"Success Rate: {success_rate:.1%} ({success_count}/{num_episodes})")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.1f}")
    print(f"Average Final Distance: {avg_final_distance:.3f}m")
    
    if successful_distances:
        print(f"Successful Episodes - Avg Distance: {np.mean(successful_distances):.3f}m")
    if failed_distances:
        print(f"Failed Episodes - Avg Distance: {np.mean(failed_distances):.3f}m")
    
    # Performance classification
    if success_rate >= 0.8:
        print("🏆 EXCELLENT: Model performs exceptionally well!")
    elif success_rate >= 0.6:
        print("✅ GOOD: Model shows strong performance!")
    elif success_rate >= 0.4:
        print("🔄 DECENT: Model shows reasonable performance!")
    elif success_rate >= 0.2:
        print("⚠️  POOR: Model needs improvement!")
    else:
        print("❌ VERY POOR: Model requires significant retraining!")
    
    # Save and plot results
    if save_results:
        plot_test_results(results)
        save_test_data(results)
    
    return results

def plot_test_results(results: Dict):
    """Create comprehensive test result visualizations."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Model Test Results - Success Rate: {results["success_rate"]:.1%}', 
                 fontsize=16, fontweight='bold')
    
    # 1. Success/Failure distribution
    ax = axes[0, 0]
    success_count = results['success_count']
    failure_count = results['num_episodes'] - success_count
    
    wedges, texts, autotexts = ax.pie(
        [success_count, failure_count],
        labels=['Success', 'Failure'],
        colors=['green', 'red'],
        autopct='%1.1f%%',
        startangle=90
    )
    ax.set_title('Success/Failure Distribution')
    
    # 2. Episode rewards
    ax = axes[0, 1]
    ax.plot(results['episode_rewards'], alpha=0.7)
    ax.axhline(y=np.mean(results['episode_rewards']), color='red', linestyle='--', 
               label=f'Average: {np.mean(results["episode_rewards"]):.1f}')
    ax.set_title('Episode Rewards')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 3. Final distances histogram
    ax = axes[0, 2]
    ax.hist(results['final_distances'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(x=0.05, color='red', linestyle='--', label='Success Threshold (5cm)')
    ax.set_title('Final Distance Distribution')
    ax.set_xlabel('Distance to Target (m)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Success vs failure distances
    ax = axes[1, 0]
    if results['successful_distances']:
        ax.hist(results['successful_distances'], bins=15, alpha=0.7, color='green', 
                label=f'Successful ({len(results["successful_distances"])})')
    if results['failed_distances']:
        ax.hist(results['failed_distances'], bins=15, alpha=0.7, color='red',
                label=f'Failed ({len(results["failed_distances"])})')
    ax.set_title('Distance Distribution by Outcome')
    ax.set_xlabel('Final Distance (m)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Steps per episode
    ax = axes[1, 1]
    colors = ['green' if success else 'red' for success in results['episode_successes']]
    ax.scatter(range(len(results['episode_steps'])), results['episode_steps'], 
               c=colors, alpha=0.6)
    ax.axhline(y=np.mean(results['episode_steps']), color='blue', linestyle='--',
               label=f'Average: {np.mean(results["episode_steps"]):.1f}')
    ax.set_title('Steps per Episode')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. 3D workspace visualization
    ax = axes[1, 2]
    ax.remove()
    ax = fig.add_subplot(2, 3, 6, projection='3d')
    
    # Plot start positions
    start_pos = np.array(results['start_positions'])
    ax.scatter(start_pos[:, 0], start_pos[:, 1], start_pos[:, 2], 
               c='blue', s=20, alpha=0.6, label='Start Positions')
    
    # Plot end positions (colored by success)
    end_pos = np.array(results['end_positions'])
    colors = ['green' if success else 'red' for success in results['episode_successes']]
    ax.scatter(end_pos[:, 0], end_pos[:, 1], end_pos[:, 2],
               c=colors, s=30, alpha=0.8, label='End Positions')
    
    # Plot target positions
    target_pos = np.array(results['target_positions'])
    ax.scatter(target_pos[:, 0], target_pos[:, 1], target_pos[:, 2],
               c='yellow', s=40, alpha=0.8, marker='*', label='Targets')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Workspace Analysis')
    ax.legend()
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/model_test_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Test result plots saved to results/model_test_results.png")

def save_test_data(results: Dict):
    """Save test data to file."""
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            json_results[key] = value.tolist()
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
            json_results[key] = [arr.tolist() for arr in value]
        else:
            json_results[key] = value
    
    # Save to JSON
    os.makedirs('results', exist_ok=True)
    with open('results/test_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print("💾 Test data saved to results/test_results.json")

def compare_models(model_paths: List[str], num_episodes: int = 50) -> Dict:
    """Compare performance of multiple models."""
    
    print(f"🔍 Comparing {len(model_paths)} models...")
    
    comparison_results = {}
    
    for model_path in model_paths:
        print(f"\n📊 Testing {model_path}...")
        results = test_model(model_path, num_episodes, render=False, save_results=False)
        
        if results:
            comparison_results[model_path] = {
                'success_rate': results['success_rate'],
                'avg_reward': results['avg_reward'],
                'avg_final_distance': results['avg_final_distance']
            }
    
    # Print comparison
    print(f"\n🏆 MODEL COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Model':<30} {'Success Rate':<15} {'Avg Reward':<12} {'Avg Distance':<12}")
    print("-" * 60)
    
    for model_path, results in comparison_results.items():
        model_name = os.path.basename(model_path)
        print(f"{model_name:<30} {results['success_rate']:<15.1%} "
              f"{results['avg_reward']:<12.1f} {results['avg_final_distance']:<12.3f}")
    
    return comparison_results

def main():
    """Main testing function."""
    
    print("🧪 4-DOF Robot Arm Model Testing")
    print("=" * 50)
    
    # Available models to test
    available_models = [
        'checkpoints/ddpg_4dof',
        'checkpoints/curriculum_final',
        'checkpoints/curriculum_stage_4'
    ]
    
    # Find available models
    existing_models = []
    for model_path in available_models:
        if os.path.exists(f"{model_path}_actor.h5"):
            existing_models.append(model_path)
    
    if not existing_models:
        print("❌ No trained models found!")
        print("Please train a model first using:")
        print("   python examples/train_ddpg.py")
        print("   or")
        print("   python examples/train_curriculum.py")
        return
    
    print(f"📁 Found {len(existing_models)} trained models:")
    for i, model in enumerate(existing_models):
        print(f"   {i+1}. {model}")
    
    # Test the most recent model
    model_to_test = existing_models[-1]  # Use the last (most recent) model
    print(f"\n🎯 Testing model: {model_to_test}")
    
    # Run comprehensive test
    results = test_model(
        model_path=model_to_test,
        num_episodes=100,
        render=True,  # Show first few episodes
        save_results=True
    )
    
    # Compare all available models if more than one
    if len(existing_models) > 1:
        print(f"\n🔍 Comparing all available models...")
        compare_models(existing_models, num_episodes=25)

if __name__ == "__main__":
    main()
