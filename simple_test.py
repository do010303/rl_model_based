#!/usr/bin/env python3
"""
Very simple test of individual components
"""

print("🧪 Testing individual components...")

# Test 1: Environment import
try:
    from environments.robot_4dof_env import Robot4DOFEnv
    print("✅ Environment import - OK")
except Exception as e:
    print(f"❌ Environment import failed: {e}")
    exit(1)

# Test 2: Create environment without reset
try:
    env = Robot4DOFEnv()
    print("✅ Environment creation - OK")
    print(f"   - Action space: {env.action_space}")
    print(f"   - Obs space: {env.observation_space}")
except Exception as e:
    print(f"❌ Environment creation failed: {e}")
    exit(1)

print("\n🎉 Basic components working!")
print("\n📋 Project structure created successfully!")
print("\n🚀 Ready for development!")
print("\n💡 Next steps:")
print("   • Add your custom training logic")
print("   • Customize environment parameters")  
print("   • Implement additional algorithms")
print("   • Test with real hardware")

print("\n✨ Example usage:")
print("   python3 examples/train_ddpg.py")
print("   python3 examples/test_model.py")
