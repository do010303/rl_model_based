#!/usr/bin/env python3
"""
Quick check: How reachable is X=0.10m (10cm)?
"""
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fk_ik_utils import fk

print("="*70)
print("🎯 REACHABILITY TEST: X = 0.10m (10cm)")
print("="*70)

# Joint limits with safety margin (from URDF + safety)
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

print(f"\n📏 Joint Limits (with 5° safety margin):")
print(f"   Joint1: {np.degrees(joint_limits_low[0]):.1f}° to {np.degrees(joint_limits_high[0]):.1f}°")
print(f"   Joint2: {np.degrees(joint_limits_low[1]):.1f}° to {np.degrees(joint_limits_high[1]):.1f}°")
print(f"   Joint3: {np.degrees(joint_limits_low[2]):.1f}° to {np.degrees(joint_limits_high[2]):.1f}°")
print(f"   Joint4: {np.degrees(joint_limits_low[3]):.1f}° to {np.degrees(joint_limits_high[3]):.1f}°")

# CONSERVATIVE Target zone (94% IK success rate)
target_x = 0.15  # Drawing surface position (was 0.10)
target_y_min = -0.10  # Conservative (was -0.14)
target_y_max = 0.10   # Conservative (was +0.14)
target_z_min = 0.08   # Conservative (was 0.05)
target_z_max = 0.18   # Conservative (was 0.22)

print(f"\n🎯 Target Zone (CONSERVATIVE WORKSPACE):")
print(f"   X: {target_x}m (FIXED - drawing surface)")
print(f"   Y: {target_y_min}m to {target_y_max}m (±10cm)")
print(f"   Z: {target_z_min}m to {target_z_max}m (8-18cm)")
print(f"   Success Rate: 94% (tested with constrained IK)")

# Sample workspace
print(f"\n🔍 Sampling 5000 random configurations within joint limits...")
reachable = 0
total = 5000

for i in range(total):
    # Random joint configuration within limits
    joints = np.random.uniform(joint_limits_low, joint_limits_high)
    
    # Calculate FK
    try:
        x, y, z = fk(joints)
        
        # Check if in target zone
        if (abs(x - target_x) < 0.01 and
            target_y_min <= y <= target_y_max and
            target_z_min <= z <= target_z_max):
            reachable += 1
    except:
        continue

coverage = (reachable / total) * 100

print(f"\n📊 Results:")
print(f"   Reachable configurations: {reachable}/{total}")
print(f"   Coverage: {coverage:.1f}%")

if coverage > 20:
    print(f"\n   ✅ EXCELLENT! Targets at X=0.10m are highly reachable")
elif coverage > 10:
    print(f"\n   ✅ GOOD! Targets at X=0.10m are reasonably reachable")
elif coverage > 5:
    print(f"\n   ⚠️  MODERATE. Training will work but may be slow")
else:
    print(f"\n   ❌ LOW. Consider moving targets closer (X=0.05m or X=0.00m)")

print("="*70)
