"""
Forward Kinematics for 4DOF Robot Arm
Extracted DIRECTLY from robot_4dof_rl.urdf.xacro for accuracy

URDF Joint Origins (exact values):
- Joint1 (base to link1): xyz="0.000476 0.0 0.033399"
- Joint2 (link1 to link2): xyz="-0.009952 -0.001031 0.052459"  
- Joint3 (link2 to link3): xyz="0.002642 -0.0002 0.063131"
- Joint4 (link3 to link4): xyz="-0.000123 -7e-05 0.052516"
- Rigid5 (link4 to endefff): xyz="0.001137 0.01875 0.077946"

These values come from the actual CAD model exported to URDF.
"""
import numpy as np

def fk(joint_angles):
    """
    Forward Kinematics using EXACT URDF parameters
    
    Args:
        joint_angles: [theta1, theta2, theta3, theta4] in RADIANS
        
    Returns:
        (x, y, z) end-effector position in METERS
    """
    # URDF-based kinematic chain (EXACT values from robot_4dof_rl.urdf.xacro)
    # Joint1: base_link -> link_1_1_1
    j1_origin = np.array([0.000476, 0.0, 0.033399])
    
    # Joint2: link_1_1_1 -> link_2_1_1
    j2_origin = np.array([-0.009952, -0.001031, 0.052459])
    
    # Joint3: link_2_1_1 -> link_3_1_1
    j3_origin = np.array([0.002642, -0.0002, 0.063131])
    
    # Joint4: link_3_1_1 -> link_4_1_1
    j4_origin = np.array([-0.000123, -0.000070, 0.052516])
    
    # Rigid5: link_4_1_1 -> endefff_1 (fixed joint)
    ee_offset = np.array([0.001137, 0.01875, 0.077946])
    
    # Rotation matrices for each joint
    def rot_z(theta):
        """Rotation around Z-axis"""
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
    
    def rot_y(theta):
        """Rotation around Y-axis"""
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [ c, 0, s],
            [ 0, 1, 0],
            [-s, 0, c]
        ])
    
    # Build transformation chain
    # Joint1 rotates around Z-axis
    R1 = rot_z(joint_angles[0])
    p1 = R1 @ j2_origin  # Transform j2_origin to world frame after J1 rotation
    
    # Joint2 rotates around Y-axis  
    R2 = R1 @ rot_y(joint_angles[1])
    p2 = p1 + R2 @ j3_origin
    
    # Joint3 rotates around Y-axis
    R3 = R2 @ rot_y(joint_angles[2])
    p3 = p2 + R3 @ j4_origin
    
    # Joint4 rotates around Y-axis
    R4 = R3 @ rot_y(joint_angles[3])
    p4 = p3 + R4 @ ee_offset
    
    # Final end-effector position (add base offset)
    ee_pos = j1_origin + p4
    
    return (ee_pos[0], ee_pos[1], ee_pos[2])


