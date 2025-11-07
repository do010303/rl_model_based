# Trajectory Visualization Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                  Robot End-Effector Movement                     │
│                  (X, Y, Z coordinates over time)                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ Position updates every 2mm movement
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
┌─────────────────────┐       ┌─────────────────────┐
│   RViz Trajectory   │       │  Gazebo Trajectory  │
│      Drawer         │       │      Drawer         │
└─────────────────────┘       └─────────────────────┘
          │                             │
          │ visualization_msgs/Marker   │ spawn_sdf_model service
          │ LINE_STRIP type             │ Cylinder models
          │                             │
          ▼                             ▼
┌─────────────────────┐       ┌─────────────────────┐
│   RViz Display      │       │  Gazebo Display     │
│                     │       │                     │
│  📊 Smooth green    │       │  🎨 Green cylinder  │
│     line (fast)     │       │     segments        │
│                     │       │     (visible!)      │
│  • Instant update   │       │  • ~0.1s per seg    │
│  • Lightweight      │       │  • Spawned models   │
│  • Analysis tool    │       │  • Demo/visual      │
└─────────────────────┘       └─────────────────────┘
```

## How Gazebo Drawing Works

### Step-by-Step Process

```
Episode Start
    │
    ├─> Robot moves to position A
    │   │
    │   ├─> Get end-effector position [x₁, y₁, z₁]
    │   └─> Store as first point
    │
    ├─> Robot moves to position B
    │   │
    │   ├─> Get end-effector position [x₂, y₂, z₂]
    │   ├─> Calculate distance: d = √((x₂-x₁)² + (y₂-y₁)² + (z₂-z₁)²)
    │   │
    │   └─> If d >= 2mm:
    │       │
    │       ├─> RViz: Add point to LINE_STRIP marker ⚡ FAST
    │       │
    │       └─> Gazebo: Spawn cylinder between points 🐢 SLOWER
    │           │
    │           ├─> Calculate midpoint: [(x₁+x₂)/2, (y₁+y₂)/2, (z₁+z₂)/2]
    │           ├─> Calculate length: d
    │           ├─> Calculate orientation quaternion
    │           ├─> Create SDF model (XML)
    │           └─> Spawn in Gazebo
    │
    ├─> Robot continues moving...
    │   └─> Repeat for each new position
    │
Episode End
    └─> Clear trajectory:
        ├─> RViz: Delete marker ⚡ INSTANT
        └─> Gazebo: Delete all cylinder models 🧹 ~0.01s each
```

## Cylinder Geometry Calculation

### From Two Points to Oriented Cylinder

```
Point A: [x₁, y₁, z₁]  ●─────────────● Point B: [x₂, y₂, z₂]
                        ╲           ╱
                         ╲ Cylinder╱
                          ╲       ╱
                           ╲     ╱
                            ╲   ╱
                             ╲ ╱
                              ●
                          Midpoint
                    [(x₁+x₂)/2, (y₁+y₂)/2, (z₁+z₂)/2]

Direction Vector: 
    d⃗ = [x₂-x₁, y₂-y₁, z₂-z₁]

Length:
    L = |d⃗| = √((x₂-x₁)² + (y₂-y₁)² + (z₂-z₁)²)

Orientation:
    Rotate cylinder from default Z-axis [0,0,1]
    to align with direction d⃗/L
    
    Using axis-angle to quaternion conversion:
    - Rotation axis: [0,0,1] × [dx,dy,dz] = [-dy, dx, 0]
    - Rotation angle: arccos(dz)
    - Convert to quaternion [qx, qy, qz, qw]
```

## Memory and Performance

### RViz Trajectory (visualization_msgs/Marker)

```
Memory per point: ~12 bytes (x, y, z as floats)
Update time: <1ms
Total for 100 points: ~1.2 KB, instant

Structure:
points: [Point, Point, Point, ...]
         ↓      ↓      ↓
        {x,y,z}{x,y,z}{x,y,z}...
```

### Gazebo Trajectory (Spawned Models)

```
Memory per segment: ~5 KB (full Gazebo model)
Spawn time: ~0.1s per segment
Total for 100 segments: ~500 KB, ~10 seconds

Structure:
segment_models: ["trajectory_seg_0", "trajectory_seg_1", ...]
                         ↓                    ↓
                 <Gazebo Model>        <Gazebo Model>
                 - Link                - Link
                 - Visual              - Visual
                 - Collision           - Collision
                 - Material            - Material
```

## Coordinate Frames

```
           Z (up)
           ↑
           │
           │
           └────→ X (forward from robot base)
          ╱
         ╱
        ↙ Y (left)

World Frame (origin: Gazebo world center)
    │
    ├─> Robot Base Frame
    │       │
    │       └─> Link 1 → Link 2 → Link 3 → Link 4
    │                                           │
    │                                           └─> End-Effector
    │                                                    │
    │                                                    └─> Trajectory Points
    │
    └─> All cylinders spawned in World Frame
```

## Data Flow During Training

```
┌──────────────────┐
│   RL Agent       │
│   (DDPG/SAC)     │
└────────┬─────────┘
         │ Action: [j₁, j₂, j₃, j₄]
         ▼
┌──────────────────┐
│   Environment    │
│ move_to_joint_   │
│   positions()    │
└────────┬─────────┘
         │ Send to Gazebo
         ▼
┌──────────────────┐       ┌──────────────────┐
│  Gazebo Physics  │──────▶│  get_state()     │
│  (joints move)   │       │  - Read link_4   │
└──────────────────┘       │    position      │
                           └────────┬─────────┘
                                    │ EE position [x,y,z]
                   ┌────────────────┴──────────────┐
                   │                               │
                   ▼                               ▼
         ┌──────────────────┐          ┌──────────────────┐
         │ trajectory_drawer│          │  gazebo_drawer   │
         │ .add_point_array │          │ .add_point_array │
         └────────┬─────────┘          └────────┬─────────┘
                  │                              │
                  ▼                              ▼
         ┌──────────────────┐          ┌──────────────────┐
         │ /visualization_  │          │ /gazebo/spawn_   │
         │   _marker        │          │   sdf_model      │
         │   (topic)        │          │   (service)      │
         └────────┬─────────┘          └────────┬─────────┘
                  │                              │
                  ▼                              ▼
         ┌──────────────────┐          ┌──────────────────┐
         │  RViz Display    │          │ Gazebo Display   │
         │  Green Line      │          │ Green Cylinders  │
         └──────────────────┘          └──────────────────┘
```

## Comparison Table

| Aspect | RViz Marker | Gazebo Cylinders |
|--------|-------------|------------------|
| **Rendering** | 2D line in 3D space | 3D cylinder models |
| **Visibility** | RViz window only | Gazebo window |
| **Speed** | Instant | ~0.1s per segment |
| **Memory** | ~12 bytes/point | ~5 KB/segment |
| **Clearing** | Instant | ~0.01s per model |
| **Smoothness** | Perfect curve | Segmented |
| **Physics** | No collision | Has collision box (disabled) |
| **Light effect** | Flat color | Ambient + diffuse + emissive |
| **Best for** | Analysis, debugging | Demo, visualization |

## Example Episode Timeline

```
Time    Event                          RViz         Gazebo
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
0.0s    Episode start                  Clear        Delete 45 models
        Reset to home position         -            -
        
0.2s    Move to position 1             Point added  -
        EE at [0.15, 0.02, 0.12]      

0.5s    Move to position 2             Point added  Cylinder spawned
        EE at [0.15, 0.025, 0.12]     Line grows   Model #1 created

0.8s    Move to position 3             Point added  Cylinder spawned
        EE at [0.15, 0.03, 0.125]     Line grows   Model #2 created

...     [Continue for 45 steps]        ...          ...

9.0s    Goal reached!                  45 points    44 cylinders
        Episode end                    Smooth line  Segmented line

9.1s    Clear trajectory               Instant ✓    Deleting...
        Prepare for next episode       

9.5s    All cleared                    Ready ✓      Ready ✓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Summary

**Two parallel systems, one trajectory:**

- **RViz**: Fast, lightweight, perfect for analysis
- **Gazebo**: Slower but visible in simulation, great for demos

**Together**: Best visualization experience! 🎨

---

**File**: TRAJECTORY_ARCHITECTURE.md  
**Date**: November 7, 2025  
**Purpose**: Technical explanation of dual trajectory system
