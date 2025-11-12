#!/usr/bin/env python3
"""
Constrained Inverse Kinematics for Drawing Surface Task

The robot must reach targets on a VERTICAL PLANE at X = 0.15m (15cm)
This is a 2D reaching problem constrained to the drawing surface.

Agent learns to control (Y, Z) positions
IK solver converts (X=0.15, Y, Z) → joint angles
"""

import numpy as np
from scipy.optimize import minimize
from fk_ik_utils import fk

# Try to import rospy for logging (optional)
try:
    import rospy
    HAS_ROSPY = True
except ImportError:
    HAS_ROSPY = False


# Drawing surface constraint
SURFACE_X = 0.15  # 15cm from robot base (FIXED)

# Joint limits (matching URDF)
JOINT_LIMITS = {
    'Joint1': (-np.pi/2, np.pi/2),      # ±90°
    'Joint2': (-np.pi/2, np.pi/2),      # ±90°
    'Joint3': (-np.pi/2, np.pi/2),      # ±90°
    'Joint4': (0.0, np.pi)              # 0-180°
}


def constrained_ik(target_y, target_z, initial_guess=None, tolerance=0.001, max_iterations=200):
    """
    Inverse Kinematics solver with X CONSTRAINED to surface position
    
    The end-effector MUST be at X = 0.15m (drawing surface)
    Agent only controls Y and Z positions
    
    Args:
        target_y: Desired Y position (meters)
        target_z: Desired Z position (meters)
        initial_guess: Initial joint angles (optional)
        tolerance: Position error tolerance (meters)
        max_iterations: Maximum optimization iterations
        
    Returns:
        joint_angles: [theta1, theta2, theta3, theta4] in RADIANS
        success: True if solution found within tolerance
        error: Final position error (meters)
        x_error: X position error (meters)
    """
    # SAFETY CHECK: Validate inputs for NaN/Inf
    if not np.isfinite(target_y) or not np.isfinite(target_z):
        if HAS_ROSPY:
            rospy.logerr(f"🛑 IK received invalid inputs! Y={target_y}, Z={target_z}")
        else:
            print(f"[ERROR] IK received invalid inputs! Y={target_y}, Z={target_z}")
        # Return safe home position
        joint_angles = np.array([0.0, 0.0, 0.0, np.pi/2])
        return joint_angles, False, float('inf'), float('inf')
    
    # Target position with FIXED X
    target_pos = np.array([SURFACE_X, target_y, target_z])
    
    # First, check if target is likely reachable
    reachable, reason = check_reachability(target_y, target_z, verbose=False)
    if not reachable:
        # Return a safe default position (home) if unreachable
        if HAS_ROSPY:
            rospy.logwarn(f"Target (Y={target_y:.3f}, Z={target_z:.3f}) likely unreachable: {reason}")
        else:
            print(f"[WARN] Target (Y={target_y:.3f}, Z={target_z:.3f}) likely unreachable: {reason}")
        joint_angles = np.array([0.0, 0.0, 0.0, np.pi/2])
        ee_x, ee_y, ee_z = fk(joint_angles)
        error = np.linalg.norm(np.array([ee_x, ee_y, ee_z]) - target_pos)
        return joint_angles, False, error, abs(ee_x - SURFACE_X)
    
    # Try multiple initial guesses to find better solutions
    initial_guesses = []
    
    if initial_guess is not None:
        initial_guesses.append(initial_guess)
    
    # Add strategic initial guesses based on target Y-Z
    # Guess 1: Simple approximation
    theta1_approx = np.arctan2(target_y, SURFACE_X)  # Rotate base toward Y
    initial_guesses.append(np.array([theta1_approx, 0.3, 0.3, 1.5]))
    
    # Guess 2: Home-like position
    initial_guesses.append(np.array([0.0, 0.2, 0.2, 1.57]))
    
    # Guess 3: Mid-range joints
    initial_guesses.append(np.array([0.0, 0.5, 0.5, 1.0]))
    
    # Joint limit bounds with safety margin
    margin = 0.087  # 5 degrees in radians
    bounds = [
        (JOINT_LIMITS['Joint1'][0] + margin, JOINT_LIMITS['Joint1'][1] - margin),
        (JOINT_LIMITS['Joint2'][0] + margin, JOINT_LIMITS['Joint2'][1] - margin),
        (JOINT_LIMITS['Joint3'][0] + margin, JOINT_LIMITS['Joint3'][1] - margin),
        (JOINT_LIMITS['Joint4'][0] + margin, JOINT_LIMITS['Joint4'][1] - margin),
    ]
    
    # Cost function: minimize distance to target
    def cost_function(joint_angles):
        """Calculate squared distance from end-effector to target"""
        ee_x, ee_y, ee_z = fk(joint_angles)
        ee_pos = np.array([ee_x, ee_y, ee_z])
        
        # Weighted error (emphasize X constraint EXTREMELY heavily!)
        error = ee_pos - target_pos
        
        # EXTREMELY HEAVY penalty for X deviation (must stay at surface!)
        x_weight = 1000.0  # Very strong penalty for X deviation
        y_weight = 1.0
        z_weight = 1.0
        
        weighted_error = np.array([
            x_weight * error[0]**2,
            y_weight * error[1]**2,
            z_weight * error[2]**2
        ])
        
        return np.sum(weighted_error)
    
    # Try all initial guesses and keep the best solution
    best_result = None
    best_error = float('inf')
    
    for guess in initial_guesses:
        # Optimize using L-BFGS-B (respects bounds)
        result = minimize(
            cost_function,
            guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iterations, 'ftol': tolerance**2}
        )
        
        # Calculate error for this solution
        ee_x, ee_y, ee_z = fk(result.x)
        ee_pos = np.array([ee_x, ee_y, ee_z])
        error = np.linalg.norm(ee_pos - target_pos)
        
        if error < best_error:
            best_error = error
            best_result = result
    
    # Extract best solution
    joint_angles = best_result.x
    
    # Calculate final error
    ee_x, ee_y, ee_z = fk(joint_angles)
    ee_pos = np.array([ee_x, ee_y, ee_z])
    error = np.linalg.norm(ee_pos - target_pos)
    
    # Check X constraint specifically
    x_error = abs(ee_x - SURFACE_X)
    
    # Success criteria: total error < tolerance AND X error < 10mm
    success = (error < tolerance) and (x_error < 0.010)
    
    return joint_angles, success, error, x_error


def constrained_ik_batch(target_yz_pairs, initial_guess=None):
    """
    Solve IK for multiple (Y, Z) targets in batch
    
    Args:
        target_yz_pairs: Array of shape (N, 2) with [Y, Z] pairs
        initial_guess: Initial joint angles for first target
        
    Returns:
        solutions: Array of shape (N, 4) with joint angles
        successes: Array of shape (N,) with success flags
        errors: Array of shape (N,) with position errors
    """
    N = len(target_yz_pairs)
    solutions = np.zeros((N, 4))
    successes = np.zeros(N, dtype=bool)
    errors = np.zeros(N)
    
    current_guess = initial_guess
    
    for i, (target_y, target_z) in enumerate(target_yz_pairs):
        joint_angles, success, error, _ = constrained_ik(
            target_y, target_z, 
            initial_guess=current_guess
        )
        
        solutions[i] = joint_angles
        successes[i] = success
        errors[i] = error
        
        # Use previous solution as next initial guess (warm start)
        current_guess = joint_angles
    
    return solutions, successes, errors


def validate_surface_constraint(joint_angles, tolerance=0.005):
    """
    Verify that joint angles result in X ≈ SURFACE_X
    
    Args:
        joint_angles: [theta1, theta2, theta3, theta4]
        tolerance: Maximum allowed X deviation (meters)
        
    Returns:
        valid: True if X constraint satisfied
        x_position: Actual X position
        x_error: Deviation from SURFACE_X
    """
    ee_x, ee_y, ee_z = fk(joint_angles)
    x_error = abs(ee_x - SURFACE_X)
    valid = (x_error < tolerance)
    
    return valid, ee_x, x_error


def check_reachability(target_y, target_z, verbose=False):
    """
    Quick check if (Y, Z) position is likely reachable at X=SURFACE_X
    
    Uses fast heuristic check before running full IK optimization
    
    Args:
        target_y: Desired Y position
        target_z: Desired Z position
        verbose: Print diagnostic info
        
    Returns:
        reachable: True if position likely reachable
        reason: String describing why unreachable (if applicable)
    """
    # Total distance from base to target
    distance_3d = np.sqrt(SURFACE_X**2 + target_y**2 + target_z**2)
    
    # Robot arm total length (approximate from URDF)
    # Base to J1: 0.033m
    # J1 to J2: 0.052m  
    # J2 to J3: 0.063m
    # J3 to J4: 0.052m
    # J4 to EE: 0.078m
    # Total: ~0.278m = 27.8cm
    max_reach = 0.28  # 28cm maximum reach
    min_reach = 0.03  # 3cm minimum reach (too close to base)
    
    if distance_3d > max_reach:
        if verbose:
            print(f"❌ Target too far! Distance={distance_3d*100:.1f}cm > max={max_reach*100:.1f}cm")
        return False, "too_far"
    
    if distance_3d < min_reach:
        if verbose:
            print(f"❌ Target too close! Distance={distance_3d*100:.1f}cm < min={min_reach*100:.1f}cm")
        return False, "too_close"
    
    # Check Z limits (physical constraint)
    if target_z < 0.03:  # Too close to ground
        if verbose:
            print(f"❌ Target too low! Z={target_z*100:.1f}cm < 3cm (ground collision)")
        return False, "ground_collision"
    
    if target_z > 0.25:  # Too high
        if verbose:
            print(f"❌ Target too high! Z={target_z*100:.1f}cm > 25cm")
        return False, "too_high"
    
    # Check Y limits (drawing surface bounds)
    if abs(target_y) > 0.15:
        if verbose:
            print(f"❌ Target outside Y range! |Y|={abs(target_y)*100:.1f}cm > 15cm")
        return False, "out_of_y_bounds"
    
    # Additional check: With X fixed at 0.15, check if Y-Z is reachable
    # Distance in Y-Z plane
    distance_yz = np.sqrt(target_y**2 + target_z**2)
    
    # Remaining reach after accounting for X=0.15
    remaining_reach = np.sqrt(max(0, max_reach**2 - SURFACE_X**2))
    
    if distance_yz > remaining_reach:
        if verbose:
            print(f"❌ Y-Z too far with X={SURFACE_X}! YZ_dist={distance_yz*100:.1f}cm > remaining={remaining_reach*100:.1f}cm")
        return False, "yz_unreachable"
    
    if verbose:
        print(f"✅ Target likely reachable: 3D_dist={distance_3d*100:.1f}cm, YZ_dist={distance_yz*100:.1f}cm")
    
    return True, "reachable"


def validate_surface_constraint(joint_angles, tolerance=0.005):
    """
    Verify that joint angles result in X ≈ SURFACE_X
    
    Args:
        joint_angles: [theta1, theta2, theta3, theta4]
        tolerance: Maximum allowed X deviation (meters)
        
    Returns:
        valid: True if X constraint satisfied
        x_position: Actual X position
        x_error: Deviation from SURFACE_X
    """
    ee_x, ee_y, ee_z = fk(joint_angles)
    x_error = abs(ee_x - SURFACE_X)
    valid = (x_error < tolerance)
    
    return valid, ee_x, x_error


# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

def test_constrained_ik():
    """Test constrained IK solver"""
    print("=" * 70)
    print("🧪 Testing Constrained IK Solver")
    print("=" * 70)
    print(f"Surface X position: {SURFACE_X}m (FIXED)\n")
    
    # Test cases: (Y, Z) pairs on the CONSERVATIVE drawing surface
    test_targets = [
        (0.0, 0.13),      # Center of surface (updated to conservative range)
        (0.05, 0.14),     # Right, mid-height
        (-0.05, 0.14),    # Left, mid-height
        (0.09, 0.10),     # Far right, low (within ±10cm)
        (-0.09, 0.17),    # Far left, high (within conservative range)
    ]
    
    print("Test Targets:")
    for i, (y, z) in enumerate(test_targets, 1):
        print(f"  {i}. Y={y:+.3f}m, Z={z:.3f}m")
    print()
    
    # Solve IK for each target
    for i, (target_y, target_z) in enumerate(test_targets, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}: Target (X={SURFACE_X}, Y={target_y:+.3f}, Z={target_z:.3f})")
        print(f"{'='*70}")
        
        # Solve IK
        joint_angles, success, error, x_error = constrained_ik(target_y, target_z)
        
        # Verify with FK
        ee_x, ee_y, ee_z = fk(joint_angles)
        
        # Results
        print(f"Joint angles (rad):  {np.round(joint_angles, 4)}")
        print(f"Joint angles (deg):  {np.round(np.degrees(joint_angles), 1)}")
        print(f"\nReached position:")
        print(f"  X: {ee_x:.6f}m (target: {SURFACE_X:.6f}m, error: {x_error*1000:.2f}mm)")
        print(f"  Y: {ee_y:.6f}m (target: {target_y:.6f}m, error: {abs(ee_y-target_y)*1000:.2f}mm)")
        print(f"  Z: {ee_z:.6f}m (target: {target_z:.6f}m, error: {abs(ee_z-target_z)*1000:.2f}mm)")
        print(f"\nTotal error: {error*1000:.2f}mm")
        print(f"X constraint: {'✅ SATISFIED' if x_error < 0.005 else '❌ VIOLATED'}")
        print(f"Success: {'✅ YES' if success else '❌ NO'}")
    
    print(f"\n{'='*70}")
    print("✅ Constrained IK test complete!")
    print(f"{'='*70}\n")


def test_workspace_coverage():
    """Test IK solver across the reachable workspace on surface"""
    print("=" * 70)
    print("🗺️  Testing Workspace Coverage on Drawing Surface")
    print("=" * 70)
    print(f"Surface: X = {SURFACE_X}m (FIXED)")
    print(f"Y range: -0.10m to +0.10m (CONSERVATIVE)")
    print(f"Z range: 0.08m to 0.18m (CONSERVATIVE)\n")
    
    # Grid of points on the surface - CONSERVATIVE WORKSPACE
    y_values = np.linspace(-0.10, 0.10, 10)
    z_values = np.linspace(0.08, 0.18, 10)
    
    successes = 0
    failures = 0
    max_error = 0.0
    
    print(f"Testing {len(y_values) * len(z_values)} points...\n")
    
    for y in y_values:
        for z in z_values:
            joint_angles, success, error, x_error = constrained_ik(y, z)
            
            if success and x_error < 0.005:
                successes += 1
            else:
                failures += 1
                print(f"❌ Failed: Y={y:+.3f}, Z={z:.3f}, error={error*1000:.1f}mm, x_err={x_error*1000:.1f}mm")
            
            max_error = max(max_error, error)
    
    total = successes + failures
    success_rate = (successes / total) * 100
    
    print(f"\n{'='*70}")
    print(f"Results:")
    print(f"  Total points tested: {total}")
    print(f"  Successes: {successes}")
    print(f"  Failures: {failures}")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Max error: {max_error*1000:.2f}mm")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'coverage':
        test_workspace_coverage()
    else:
        test_constrained_ik()
