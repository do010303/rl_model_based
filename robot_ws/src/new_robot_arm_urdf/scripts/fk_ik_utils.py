"""
Forward Kinematics for 4DOF Robot Arm
Adapted from Control_Robot/p_fkdh.py for ROS Noetic

DH Parameters for your 4DOF robot:
- Base offset: 66mm (0.066m)
- Link lengths: 80mm, 80mm, 50mm (0.08m, 0.08m, 0.05m)
"""
import numpy as np

def fk(joint_angles):
    """
    Forward Kinematics using DH parameters
    
    Args:
        joint_angles: [theta1, theta2, theta3, theta4] in RADIANS
        
    Returns:
        (x, y, z) end-effector position in METERS
    """
    # Convert joint angles to degrees for DH calculation
    j1_deg = np.degrees(joint_angles[0])
    j2_deg = np.degrees(joint_angles[1])
    j3_deg = np.degrees(joint_angles[2])
    j4_deg = np.degrees(joint_angles[3])
    
    # DH parameters (from p_fkdh.py)
    # Convert mm to meters
    base_height = 0.066  # 66mm base offset
    link1 = 0.080  # 80mm
    link2 = 0.080  # 80mm  
    link3 = 0.050  # 50mm
    
    # DH table: [theta, d, a, alpha]
    # Joint 1: Revolute around Z
    theta1 = j1_deg
    d1 = base_height
    a1 = 0.0
    alpha1 = 90.0  # degrees
    
    # Joint 2: Revolute in XY plane
    theta2 = j2_deg
    d2 = 0.0
    a2 = link1
    alpha2 = 0.0
    
    # Joint 3: Revolute in XY plane
    theta3 = j3_deg
    d3 = 0.0
    a3 = link2
    alpha3 = 0.0
    
    # Joint 4: Revolute in XY plane
    theta4 = j4_deg
    d4 = 0.0
    a4 = link3
    alpha4 = 0.0
    
    # Calculate transformation matrices using DH convention
    def dh_matrix(theta, d, a, alpha):
        """DH transformation matrix"""
        theta_rad = np.radians(theta)
        alpha_rad = np.radians(alpha)
        
        return np.array([
            [np.cos(theta_rad), -np.sin(theta_rad)*np.cos(alpha_rad),  np.sin(theta_rad)*np.sin(alpha_rad), a*np.cos(theta_rad)],
            [np.sin(theta_rad),  np.cos(theta_rad)*np.cos(alpha_rad), -np.cos(theta_rad)*np.sin(alpha_rad), a*np.sin(theta_rad)],
            [0,                  np.sin(alpha_rad),                     np.cos(alpha_rad),                    d],
            [0,                  0,                                      0,                                     1]
        ])
    
    # Calculate each transformation
    T01 = dh_matrix(theta1, d1, a1, alpha1)
    T12 = dh_matrix(theta2, d2, a2, alpha2)
    T23 = dh_matrix(theta3, d3, a3, alpha3)
    T34 = dh_matrix(theta4, d4, a4, alpha4)
    
    # Forward kinematics: T_0_4 = T01 * T12 * T23 * T34
    T04 = T01 @ T12 @ T23 @ T34
    
    # Add the fixed offset from link_4 to endefff_1 (from URDF Rigid5 joint)
    # URDF: <origin xyz="0.001137 0.01875 0.077946" rpy="0 0 0"/>
    # This extends to the ACTUAL tip of the end-effector (endefff_1 link)
    endefff_offset = np.array([0.001137, 0.01875, 0.077946, 1.0])  # Homogeneous coordinates
    
    # Transform the offset to world coordinates
    ee_pos_homogeneous = T04 @ endefff_offset
    
    # Extract end-effector position
    x = ee_pos_homogeneous[0]
    y = ee_pos_homogeneous[1]
    z = ee_pos_homogeneous[2]
    
    return (x, y, z)


