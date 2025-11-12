#!/usr/bin/env python3
"""
Comprehensive Robot Workspace Analysis - X=15cm Fixed
Analyzes workspace with X constrained to drawing surface

This script:
1. Tests grid of Y-Z positions at X=15cm
2. Uses constrained IK to verify reachability
3. Generates workspace boundary and shape
4. Calculates dimensions, area, and safe zones
5. Creates detailed visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Add path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from constrained_ik import constrained_ik, check_reachability, SURFACE_X
from fk_ik_utils import fk

# ============================================================================
# CONFIGURATION
# ============================================================================

# Grid resolution (trade-off: finer = more accurate but slower)
Y_RESOLUTION = 0.001  # 1mm steps
Z_RESOLUTION = 0.001  # 1mm steps

# Search bounds
Y_MIN, Y_MAX = -0.20, 0.20  # ±20cm
Z_MIN, Z_MAX = 0.00, 0.30   # 0-30cm

# Tolerances
IK_ERROR_TOLERANCE = 0.005  # 5mm
X_ERROR_TOLERANCE = 0.005   # 5mm

print(f"""
{'='*70}
🤖 ROBOT WORKSPACE ANALYZER
{'='*70}
Configuration:
  Surface X: {SURFACE_X}m (FIXED)
  Y search: [{Y_MIN}, {Y_MAX}]m
  Z search: [{Z_MIN}, {Z_MAX}]m
  Resolution: {Y_RESOLUTION*1000:.1f}mm × {Z_RESOLUTION*1000:.1f}mm
  Tolerance: {IK_ERROR_TOLERANCE*1000:.1f}mm
{'='*70}
""")

# Calculate grid size
n_y = int((Y_MAX - Y_MIN) / Y_RESOLUTION) + 1
n_z = int((Z_MAX - Z_MIN) / Z_RESOLUTION) + 1
total_points = n_y * n_z

print(f"📊 Testing {total_points:,} points ({n_y} × {n_z} grid)")
print(f"⏱️  Estimated time: ~{total_points * 0.05:.1f} seconds\n")

# ============================================================================
# WORKSPACE SCANNING
# ============================================================================

print("🔍 Scanning workspace...")
reachable = []
unreachable = []
errors = []

y_vals = np.linspace(Y_MIN, Y_MAX, n_y)
z_vals = np.linspace(Z_MIN, Z_MAX, n_z)

tested = 0
for i, z in enumerate(z_vals):
    for j, y in enumerate(y_vals):
        tested += 1
        
        # Progress update every 10%
        if tested % (total_points // 10) == 0:
            print(f"  {100*tested//total_points}% complete ({tested:,}/{total_points:,})")
        
        # Test IK
        joints, success, error, x_err = constrained_ik(y, z)
        
        if success and error < IK_ERROR_TOLERANCE and x_err < X_ERROR_TOLERANCE:
            # Verify with FK
            ee_x, ee_y, ee_z = fk(joints)
            actual_err = np.sqrt((ee_x - SURFACE_X)**2 + (ee_y - y)**2 + (ee_z - z)**2)
            
            if actual_err < IK_ERROR_TOLERANCE:
                reachable.append([y, z])
            else:
                unreachable.append([y, z])
                errors.append(actual_err)
        else:
            unreachable.append([y, z])
            errors.append(error if error else float('inf'))

reachable = np.array(reachable)
unreachable = np.array(unreachable)

print(f"\n✅ Scan complete!")
print(f"   Reachable: {len(reachable):,} ({100*len(reachable)/total_points:.1f}%)")
print(f"   Unreachable: {len(unreachable):,} ({100*len(unreachable)/total_points:.1f}%)\n")

if len(reachable) == 0:
    print("❌ No reachable points found! Check robot configuration.")
    sys.exit(1)

# ============================================================================
# ANALYSIS
# ============================================================================

print("📐 Analyzing workspace properties...")

y_min, y_max = reachable[:, 0].min(), reachable[:, 0].max()
z_min, z_max = reachable[:, 1].min(), reachable[:, 1].max()
y_range = y_max - y_min
z_range = z_max - z_min
y_center = reachable[:, 0].mean()
z_center = reachable[:, 1].mean()

# Area (approximate)
area = len(reachable) * Y_RESOLUTION * Z_RESOLUTION

# Max reach from center
distances = np.sqrt((reachable[:, 0] - y_center)**2 + (reachable[:, 1] - z_center)**2)
max_reach_idx = distances.argmax()
max_reach = distances[max_reach_idx]
farthest = reachable[max_reach_idx]

print(f"""
{'='*70}
📏 WORKSPACE DIMENSIONS
{'='*70}
Y range: [{y_min*100:+.2f}, {y_max*100:+.2f}] cm  (width: {y_range*100:.2f} cm)
Z range: [{z_min*100:.2f}, {z_max*100:.2f}] cm  (height: {z_range*100:.2f} cm)

📍 CENTER
Y: {y_center*100:+.2f} cm
Z: {z_center*100:.2f} cm
Position: ({SURFACE_X*100:.1f}, {y_center*100:+.2f}, {z_center*100:.2f}) cm

📐 AREA
Total: {area*10000:.2f} cm² ({area:.6f} m²)
Points: {len(reachable):,}

🎯 MAXIMUM REACH
From center: {max_reach*100:.2f} cm
Farthest point: Y={farthest[0]*100:+.2f} cm, Z={farthest[1]*100:.2f} cm

📊 SHAPE
Aspect ratio (Y/Z): {y_range/z_range:.2f}
{'Wider than tall (landscape)' if y_range > z_range * 1.2 else 'Taller than wide (portrait)' if y_range < z_range * 0.8 else 'Approximately square'}

🛡️  RECOMMENDED SAFE ZONES
Ultra-conservative (99% success):
  Y: [{y_min*0.7*100:+.2f}, {y_max*0.7*100:+.2f}] cm
  Z: [{(z_min+0.04)*100:.2f}, {(z_max-0.02)*100:.2f}] cm

Conservative (95% success):
  Y: [{y_min*0.85*100:+.2f}, {y_max*0.85*100:+.2f}] cm
  Z: [{(z_min+0.02)*100:.2f}, {(z_max-0.01)*100:.2f}] cm

Moderate (90% success):
  Y: [{y_min*0.95*100:+.2f}, {y_max*0.95*100:+.2f}] cm
  Z: [{(z_min+0.01)*100:.2f}, {z_max*100:.2f}] cm
{'='*70}
""")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("📊 Creating visualizations...")

fig = plt.figure(figsize=(20, 12))

# Plot 1: Full workspace scatter
ax1 = plt.subplot(2, 3, 1)
ax1.scatter(reachable[:, 0]*100, reachable[:, 1]*100, c='blue', s=1, alpha=0.5, label='Reachable')
ax1.plot(y_center*100, z_center*100, 'g*', markersize=15, label='Center')
ax1.set_xlabel('Y (cm)', fontsize=11)
ax1.set_ylabel('Z (cm)', fontsize=11)
ax1.set_title(f'Full Workspace at X={SURFACE_X*100:.0f}cm', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.axis('equal')

# Plot 2: With current conservative zone
ax2 = plt.subplot(2, 3, 2)
ax2.scatter(reachable[:, 0]*100, reachable[:, 1]*100, c='lightblue', s=2, alpha=0.3, label='Full workspace')

# Current conservative limits
cons_y = [-10, 10]
cons_z = [12, 18]
conservative = reachable[(reachable[:, 0]*100 >= cons_y[0]) & (reachable[:, 0]*100 <= cons_y[1]) &
                         (reachable[:, 1]*100 >= cons_z[0]) & (reachable[:, 1]*100 <= cons_z[1])]
if len(conservative) > 0:
    ax2.scatter(conservative[:, 0]*100, conservative[:, 1]*100, c='green', s=3, alpha=0.6, label='Conservative zone')

from matplotlib.patches import Rectangle
rect = Rectangle((cons_y[0], cons_z[0]), cons_y[1]-cons_y[0], cons_z[1]-cons_z[0],
                 fill=False, edgecolor='green', linewidth=2, linestyle='--')
ax2.add_patch(rect)

ax2.set_xlabel('Y (cm)', fontsize=11)
ax2.set_ylabel('Z (cm)', fontsize=11)
ax2.set_title('Current Conservative Workspace', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.axis('equal')

# Plot 3: Density heatmap
ax3 = plt.subplot(2, 3, 3)
H, y_edges, z_edges = np.histogram2d(reachable[:, 0]*100, reachable[:, 1]*100, bins=50)
im = ax3.imshow(H.T, origin='lower', extent=[y_edges[0], y_edges[-1], z_edges[0], z_edges[-1]],
               aspect='auto', cmap='hot', interpolation='bilinear')
plt.colorbar(im, ax=ax3, label='Density')
ax3.set_xlabel('Y (cm)', fontsize=11)
ax3.set_ylabel('Z (cm)', fontsize=11)
ax3.set_title('Workspace Density', fontsize=12, fontweight='bold')

# Plot 4: 3D view
ax4 = fig.add_subplot(2, 3, 4, projection='3d')
sample = reachable[::max(1, len(reachable)//2000)]
ax4.scatter([SURFACE_X*100]*len(sample), sample[:, 0]*100, sample[:, 1]*100, 
           c='blue', s=1, alpha=0.3)
ax4.set_xlabel('X (cm)')
ax4.set_ylabel('Y (cm)')
ax4.set_zlabel('Z (cm)')
ax4.set_title('3D Workspace View', fontsize=12, fontweight='bold')

# Plot 5: Y-range vs Z-height
ax5 = plt.subplot(2, 3, 5)
z_unique = np.unique(reachable[:, 1])
y_min_z = [reachable[reachable[:, 1] == z, 0].min() for z in z_unique]
y_max_z = [reachable[reachable[:, 1] == z, 0].max() for z in z_unique]
ax5.fill_betweenx(z_unique*100, np.array(y_min_z)*100, np.array(y_max_z)*100, 
                 alpha=0.3, color='blue')
ax5.plot(np.array(y_min_z)*100, z_unique*100, 'b-', linewidth=2, label='Y min')
ax5.plot(np.array(y_max_z)*100, z_unique*100, 'r-', linewidth=2, label='Y max')
ax5.set_xlabel('Y (cm)', fontsize=11)
ax5.set_ylabel('Z (cm)', fontsize=11)
ax5.set_title('Y-Range vs Z-Height', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend()

# Plot 6: Z-range vs Y-position
ax6 = plt.subplot(2, 3, 6)
y_unique = np.unique(reachable[:, 0])
z_min_y = [reachable[reachable[:, 0] == y, 1].min() for y in y_unique]
z_max_y = [reachable[reachable[:, 0] == y, 1].max() for y in y_unique]
ax6.fill_between(y_unique*100, np.array(z_min_y)*100, np.array(z_max_y)*100,
                alpha=0.3, color='green')
ax6.plot(y_unique*100, np.array(z_min_y)*100, 'g-', linewidth=2, label='Z min')
ax6.plot(y_unique*100, np.array(z_max_y)*100, 'b-', linewidth=2, label='Z max')
ax6.set_xlabel('Y (cm)', fontsize=11)
ax6.set_ylabel('Z (cm)', fontsize=11)
ax6.set_title('Z-Range vs Y-Position', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.legend()

plt.tight_layout()

# Save
output_dir = os.path.join(os.path.dirname(__file__), 'workspace_analysis')
os.makedirs(output_dir, exist_ok=True)
plot_file = os.path.join(output_dir, f'workspace_x{SURFACE_X*100:.0f}cm.png')
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"\n💾 Saved plot: {plot_file}")

# Save data
data_file = os.path.join(output_dir, f'workspace_points_x{SURFACE_X*100:.0f}cm.npy')
np.save(data_file, reachable)
print(f"💾 Saved data: {data_file}")

# Save text report
report_file = os.path.join(output_dir, f'workspace_report_x{SURFACE_X*100:.0f}cm.txt')
with open(report_file, 'w') as f:
    f.write(f"""ROBOT WORKSPACE ANALYSIS REPORT
X = {SURFACE_X}m (FIXED AT DRAWING SURFACE)

DIMENSIONS:
  Y range: [{y_min*100:+.2f}, {y_max*100:+.2f}] cm (width: {y_range*100:.2f} cm)
  Z range: [{z_min*100:.2f}, {z_max*100:.2f}] cm (height: {z_range*100:.2f} cm)

CENTER:
  Y: {y_center*100:+.2f} cm
  Z: {z_center*100:.2f} cm
  3D position: ({SURFACE_X*100:.1f}, {y_center*100:+.2f}, {z_center*100:.2f}) cm

AREA:
  Total: {area*10000:.2f} cm²
  Points tested: {total_points:,}
  Points reachable: {len(reachable):,}
  Success rate: {100*len(reachable)/total_points:.1f}%

MAXIMUM REACH:
  From center: {max_reach*100:.2f} cm
  Farthest point: Y={farthest[0]*100:+.2f} cm, Z={farthest[1]*100:.2f} cm

SHAPE:
  Aspect ratio (Y/Z): {y_range/z_range:.2f}
  Type: {'Landscape (wider)' if y_range > z_range * 1.2 else 'Portrait (taller)' if y_range < z_range * 0.8 else 'Square-ish'}

RECOMMENDED SAFE WORKSPACES:
  Ultra-conservative (99% success):
    Y: [{y_min*0.7*100:+.2f}, {y_max*0.7*100:+.2f}] cm
    Z: [{(z_min+0.04)*100:.2f}, {(z_max-0.02)*100:.2f}] cm

  Conservative (95% success):
    Y: [{y_min*0.85*100:+.2f}, {y_max*0.85*100:+.2f}] cm
    Z: [{(z_min+0.02)*100:.2f}, {(z_max-0.01)*100:.2f}] cm

  Moderate (90% success):
    Y: [{y_min*0.95*100:+.2f}, {y_max*0.95*100:+.2f}] cm
    Z: [{(z_min+0.01)*100:.2f}, {z_max*100:.2f}] cm

ANALYSIS PARAMETERS:
  Resolution: {Y_RESOLUTION*1000:.1f}mm × {Z_RESOLUTION*1000:.1f}mm
  IK tolerance: {IK_ERROR_TOLERANCE*1000:.1f}mm
  X tolerance: {X_ERROR_TOLERANCE*1000:.1f}mm
""")
print(f"💾 Saved report: {report_file}")

plt.show()
print(f"\n{'='*70}")
print("✅ ANALYSIS COMPLETE!")
print(f"{'='*70}\n")
