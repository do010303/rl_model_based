#!/usr/bin/env python3
"""
Test FK accuracy by comparing with known robot positions
"""
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fk_ik_utils import fk

print("="*70)
print("🧪 TESTING FORWARD KINEMATICS ACCURACY")
print("="*70)

# Test 1: Home position (all joints at 0)
print("\n📍 Test 1: Home Position [0, 0, 0, 0]")
joints_home = np.array([0.0, 0.0, 0.0, 0.0])
x, y, z = fk(joints_home)
print(f"   End-effector: x={x:.6f}m, y={y:.6f}m, z={z:.6f}m")
print(f"                 x={x*1000:.2f}mm, y={y*1000:.2f}mm, z={z*1000:.2f}mm")

# Calculate expected from URDF (sum of Z offsets when all joints = 0)
expected_z = 0.033399 + 0.052459 + 0.063131 + 0.052516 + 0.077946
print(f"   Expected Z (sum of URDF Z-offsets): {expected_z:.6f}m ({expected_z*1000:.2f}mm)")
print(f"   ✓ Match!" if abs(z - expected_z) < 0.001 else f"   ✗ Mismatch! Error: {abs(z - expected_z)*1000:.2f}mm")

# Test 2: Joint4 at 90° (perpendicular to surface)
print("\n📍 Test 2: Joint4 at 90° [0, 0, 0, π/2]")
joints_j4_90 = np.array([0.0, 0.0, 0.0, np.pi/2])
x, y, z = fk(joints_j4_90)
print(f"   End-effector: x={x:.6f}m, y={y:.6f}m, z={z:.6f}m")
print(f"                 x={x*1000:.2f}mm, y={y*1000:.2f}mm, z={z*1000:.2f}mm")

# Test 3: Joint1 at 90° (rotate base)
print("\n📍 Test 3: Joint1 at 90° [π/2, 0, 0, 0]")
joints_j1_90 = np.array([np.pi/2, 0.0, 0.0, 0.0])
x, y, z = fk(joints_j1_90)
print(f"   End-effector: x={x:.6f}m, y={y:.6f}m, z={z:.6f}m")
print(f"                 x={x*1000:.2f}mm, y={y*1000:.2f}mm, z={z*1000:.2f}mm")

# Test 4: Reach forward (typical drawing position)
print("\n📍 Test 4: Reach Forward [0, 0.5, 0.5, π/2]")
joints_forward = np.array([0.0, 0.5, 0.5, np.pi/2])
x, y, z = fk(joints_forward)
print(f"   End-effector: x={x:.6f}m, y={y:.6f}m, z={z:.6f}m")
print(f"                 x={x*1000:.2f}mm, y={y*1000:.2f}mm, z={z*1000:.2f}mm")

# Test 5: Check if 15cm target is reachable
print("\n📍 Test 5: Can we reach X=0.15m (15cm)?")
print("   Testing various joint configurations...")

reachable_configs = []
for j2 in np.linspace(-1.4, 1.4, 10):
    for j3 in np.linspace(-1.4, 1.4, 10):
        for j4 in np.linspace(0.1, 3.0, 10):
            joints = np.array([0.0, j2, j3, j4])
            x, y, z = fk(joints)
            if abs(x - 0.15) < 0.01 and abs(y) < 0.15 and 0.05 < z < 0.25:
                reachable_configs.append((joints, x, y, z))

if reachable_configs:
    print(f"   ✅ Found {len(reachable_configs)} configurations that reach X≈0.15m!")
    print(f"\n   Sample configurations:")
    for i, (joints, x, y, z) in enumerate(reachable_configs[:3]):
        print(f"   {i+1}. Joints: {np.degrees(joints).round(1)}°")
        print(f"      Position: x={x:.3f}m, y={y:.3f}m, z={z:.3f}m")
else:
    print(f"   ❌ NO configurations found that reach X=0.15m!")
    print(f"   Target may be unreachable!")

print("\n" + "="*70)
print("✅ FK TEST COMPLETE")
print("="*70)
