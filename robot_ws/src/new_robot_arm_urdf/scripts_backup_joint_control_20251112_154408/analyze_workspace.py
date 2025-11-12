#!/usr/bin/env python3
"""
Robot Workspace Analysis Tool
Determines the reachable workspace of the 4DOF robot arm

This script:
1. Samples many random joint configurations
2. Calculates end-effector positions using FK
3. Visualizes the workspace as a point cloud
4. Determines safe bounds for target placement
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Add path for FK utilities
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fk_ik_utils import fk

def sample_workspace(num_samples=10000):
    """
    Sample random joint configurations and compute reachable positions
    
    Args:
        num_samples: Number of random configurations to sample
        
    Returns:
        positions: Nx3 array of [x, y, z] end-effector positions
        joint_configs: Nx4 array of joint angles that produced each position
    """
    print(f"🔍 Sampling {num_samples} random joint configurations...")
    
    # Joint limits (with safety margin)
    SAFETY_MARGIN = 0.087  # 5 degrees
    joint_limits_low = np.array([
        -np.pi/2 + SAFETY_MARGIN,  # Joint1: -85°
        -np.pi/2 + SAFETY_MARGIN,  # Joint2: -85°
        -np.pi/2 + SAFETY_MARGIN,  # Joint3: -85°
        0.0 + SAFETY_MARGIN        # Joint4: 5°
    ])
    joint_limits_high = np.array([
        np.pi/2 - SAFETY_MARGIN,   # Joint1: +85°
        np.pi/2 - SAFETY_MARGIN,   # Joint2: +85°
        np.pi/2 - SAFETY_MARGIN,   # Joint3: +85°
        np.pi - SAFETY_MARGIN      # Joint4: 175°
    ])
    
    positions = []
    joint_configs = []
    
    for i in range(num_samples):
        # Random joint configuration
        joints = np.random.uniform(joint_limits_low, joint_limits_high)
        
        # Calculate forward kinematics
        try:
            x, y, z = fk(joints)
            
            # Store if valid (not NaN)
            if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                positions.append([x, y, z])
                joint_configs.append(joints)
        except:
            continue
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{num_samples} configurations...")
    
    positions = np.array(positions)
    joint_configs = np.array(joint_configs)
    
    print(f"✅ Found {len(positions)} reachable positions")
    
    return positions, joint_configs

def analyze_workspace(positions):
    """
    Analyze workspace statistics
    
    Args:
        positions: Nx3 array of [x, y, z] positions
    """
    print("\n" + "="*70)
    print("📊 WORKSPACE ANALYSIS")
    print("="*70)
    
    # Calculate bounds
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    z_min, z_max = positions[:, 2].min(), positions[:, 2].max()
    
    print(f"\n🔹 X-axis (forward/back):")
    print(f"   Range: {x_min:.3f}m to {x_max:.3f}m")
    print(f"   Span: {x_max - x_min:.3f}m ({(x_max - x_min)*100:.1f}cm)")
    
    print(f"\n🔹 Y-axis (left/right):")
    print(f"   Range: {y_min:.3f}m to {y_max:.3f}m")
    print(f"   Span: {y_max - y_min:.3f}m ({(y_max - y_min)*100:.1f}cm)")
    
    print(f"\n🔹 Z-axis (up/down):")
    print(f"   Range: {z_min:.3f}m to {z_max:.3f}m")
    print(f"   Span: {z_max - z_min:.3f}m ({(z_max - z_min)*100:.1f}cm)")
    
    # Check current target settings
    print("\n" + "="*70)
    print("🎯 CURRENT TARGET CONFIGURATION (CONSERVATIVE)")
    print("="*70)
    
    target_x = 0.15  # Drawing surface position
    target_y_min = -0.10  # Conservative workspace (was -0.14)
    target_y_max = 0.10   # Conservative workspace (was +0.14)
    target_z_min = 0.08   # Conservative workspace (was 0.05)
    target_z_max = 0.18   # Conservative workspace (was 0.22)
    
    print(f"\nCurrent target bounds:")
    print(f"  X: {target_x:.3f}m (fixed)")
    print(f"  Y: {target_y_min:.3f}m to {target_y_max:.3f}m")
    print(f"  Z: {target_z_min:.3f}m to {target_z_max:.3f}m")
    
    # Check if targets are reachable
    print("\n" + "="*70)
    print("✅ REACHABILITY CHECK")
    print("="*70)
    
    x_reachable = x_min <= target_x <= x_max
    y_reachable = y_min <= target_y_min and target_y_max <= y_max
    z_reachable = z_min <= target_z_min and target_z_max <= z_max
    
    print(f"\n  X: {'✅ REACHABLE' if x_reachable else '❌ OUT OF REACH'}")
    if not x_reachable:
        print(f"     Target: {target_x:.3f}m, Workspace: {x_min:.3f}m to {x_max:.3f}m")
    
    print(f"  Y: {'✅ REACHABLE' if y_reachable else '❌ OUT OF REACH'}")
    if not y_reachable:
        print(f"     Target: {target_y_min:.3f}m to {target_y_max:.3f}m")
        print(f"     Workspace: {y_min:.3f}m to {y_max:.3f}m")
    
    print(f"  Z: {'✅ REACHABLE' if z_reachable else '❌ OUT OF REACH'}")
    if not z_reachable:
        print(f"     Target: {target_z_min:.3f}m to {target_z_max:.3f}m")
        print(f"     Workspace: {z_min:.3f}m to {z_max:.3f}m")
    
    # Calculate coverage
    positions_in_target = positions[
        (positions[:, 0] >= target_x - 0.02) & (positions[:, 0] <= target_x + 0.02) &
        (positions[:, 1] >= target_y_min) & (positions[:, 1] <= target_y_max) &
        (positions[:, 2] >= target_z_min) & (positions[:, 2] <= target_z_max)
    ]
    
    coverage = len(positions_in_target) / len(positions) * 100
    
    print(f"\n📊 Target zone coverage: {coverage:.1f}%")
    print(f"   ({len(positions_in_target)}/{len(positions)} sampled positions reach target zone)")
    
    # Recommendations
    print("\n" + "="*70)
    print("💡 RECOMMENDED TARGET BOUNDS")
    print("="*70)
    
    # Add safety margin (90% of workspace)
    margin = 0.05  # 5cm safety margin
    
    rec_x_min = x_min + margin
    rec_x_max = x_max - margin
    rec_y_min = y_min + margin
    rec_y_max = y_max - margin
    rec_z_min = z_min + margin
    rec_z_max = z_max - margin
    
    print(f"\nFor maximum reachability, set targets within:")
    print(f"  X: {rec_x_min:.3f}m to {rec_x_max:.3f}m ({(rec_x_min)*100:.1f}cm to {(rec_x_max)*100:.1f}cm)")
    print(f"  Y: {rec_y_min:.3f}m to {rec_y_max:.3f}m ({(rec_y_min)*100:.1f}cm to {(rec_y_max)*100:.1f}cm)")
    print(f"  Z: {rec_z_min:.3f}m to {rec_z_max:.3f}m ({(rec_z_min)*100:.1f}cm to {(rec_z_max)*100:.1f}cm)")
    
    # Suggest optimal fixed X
    print(f"\n🎯 For fixed X-position targets:")
    print(f"   Best X: {(x_min + x_max)/2:.3f}m ({((x_min + x_max)/2)*100:.1f}cm) - center of workspace")
    print(f"   Current X: {target_x:.3f}m ({target_x*100:.1f}cm)")
    
    if target_x < x_min or target_x > x_max:
        print(f"   ⚠️  Current X is OUTSIDE workspace! Move to {(x_min + x_max)/2:.3f}m")
    elif target_x < rec_x_min or target_x > rec_x_max:
        print(f"   ⚠️  Current X is near edge of workspace. Consider {(x_min + x_max)/2:.3f}m")
    else:
        print(f"   ✅ Current X is within safe workspace")
    
    return {
        'x_range': (x_min, x_max),
        'y_range': (y_min, y_max),
        'z_range': (z_min, z_max),
        'recommended_x': (rec_x_min, rec_x_max),
        'recommended_y': (rec_y_min, rec_y_max),
        'recommended_z': (rec_z_min, rec_z_max),
        'coverage': coverage
    }

def visualize_workspace(positions, save_path=None):
    """
    Create 3D visualization of workspace
    
    Args:
        positions: Nx3 array of [x, y, z] positions
        save_path: Optional path to save figure
    """
    print("\n📊 Creating workspace visualization...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # 3D scatter plot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                c=positions[:, 2], cmap='viridis', alpha=0.1, s=1)
    
    # Draw CONSERVATIVE target zone
    target_x = 0.15
    target_y = [-0.10, 0.10]  # Conservative (was [-0.14, 0.14])
    target_z = [0.08, 0.18]   # Conservative (was [0.05, 0.22])
    
    # Draw target box
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    # Create vertices of target zone box
    vertices = [
        [target_x-0.01, target_y[0], target_z[0]],
        [target_x+0.01, target_y[0], target_z[0]],
        [target_x+0.01, target_y[1], target_z[0]],
        [target_x-0.01, target_y[1], target_z[0]],
        [target_x-0.01, target_y[0], target_z[1]],
        [target_x+0.01, target_y[0], target_z[1]],
        [target_x+0.01, target_y[1], target_z[1]],
        [target_x-0.01, target_y[1], target_z[1]],
    ]
    
    # Define the 6 faces of the box
    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[7], vertices[6], vertices[2], vertices[3]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]]
    ]
    
    # Add target zone box
    ax1.add_collection3d(Poly3DCollection(faces, alpha=0.3, facecolor='red', edgecolor='red', linewidths=2))
    
    ax1.set_xlabel('X (m) - Forward')
    ax1.set_ylabel('Y (m) - Left/Right')
    ax1.set_zlabel('Z (m) - Up/Down')
    ax1.set_title('3D Workspace (Red = Target Zone)')
    
    # XY projection
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(positions[:, 0], positions[:, 1], alpha=0.1, s=1)
    ax2.plot([target_x-0.01, target_x+0.01, target_x+0.01, target_x-0.01, target_x-0.01],
             [target_y[0], target_y[0], target_y[1], target_y[1], target_y[0]], 
             'r-', linewidth=2, label='Target Zone')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top View (XY Plane)')
    ax2.grid(True)
    ax2.legend()
    ax2.axis('equal')
    
    # XZ projection
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(positions[:, 0], positions[:, 2], alpha=0.1, s=1)
    ax3.plot([target_x-0.01, target_x+0.01, target_x+0.01, target_x-0.01, target_x-0.01],
             [target_z[0], target_z[0], target_z[1], target_z[1], target_z[0]], 
             'r-', linewidth=2, label='Target Zone')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Side View (XZ Plane)')
    ax3.grid(True)
    ax3.legend()
    ax3.axis('equal')
    
    # YZ projection
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(positions[:, 1], positions[:, 2], alpha=0.1, s=1)
    ax4.plot([target_y[0], target_y[1], target_y[1], target_y[0], target_y[0]],
             [target_z[0], target_z[0], target_z[1], target_z[1], target_z[0]], 
             'r-', linewidth=2, label='Target Zone')
    ax4.set_xlabel('Y (m)')
    ax4.set_ylabel('Z (m)')
    ax4.set_title('Front View (YZ Plane)')
    ax4.grid(True)
    ax4.legend()
    ax4.axis('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved to: {save_path}")
    
    plt.show()

def main():
    """Main workspace analysis"""
    print("="*70)
    print("🤖 ROBOT WORKSPACE ANALYZER")
    print("="*70)
    
    # Sample workspace
    positions, joint_configs = sample_workspace(num_samples=10000)
    
    # Analyze
    stats = analyze_workspace(positions)
    
    # Visualize
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, 'workspace_analysis.png')
    visualize_workspace(positions, save_path)
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\n📊 Check the visualization: {save_path}")
    print("\n💡 Next steps:")
    print("   1. Review the recommended target bounds above")
    print("   2. Update main_rl_environment_noetic.py with safe bounds")
    print("   3. Restart training with reachable targets")
    
    return stats

if __name__ == '__main__':
    main()
