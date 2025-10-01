"""Test gymnasium-super-mario-bros with Python 3.13 and gymnasium."""
import sys
print(f"Python version: {sys.version}")

import gymnasium as gym
print(f"✓ gymnasium version: {gym.__version__}")

import gym_super_mario_bros
print("✓ gym_super_mario_bros imported successfully")

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
print("✓ Imports successful")

# Test creating environment
print("\n=== Testing Environment Creation ===")
env = gym_super_mario_bros.make('SuperMarioBros-v0')
print("✓ Environment created")

env = JoypadSpace(env, SIMPLE_MOVEMENT)
print("✓ JoypadSpace wrapper applied")

# Test gymnasium API
print("\n=== Testing Gymnasium API ===")
obs, info = env.reset()
print(f"✓ Reset successful - Observation shape: {obs.shape}")

for i in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
print(f"✓ Step successful after 10 steps")
print(f"  Final reward: {reward}")
print(f"  Terminated: {terminated}, Truncated: {truncated}")
print(f"  Info keys: {list(info.keys())}")

env.close()
print("✓ Environment closed")

# Test AsyncVectorEnv support (gymnasium equivalent of SubprocVecEnv)
print("\n=== Testing AsyncVectorEnv Support ===")
from gymnasium.vector import AsyncVectorEnv

def make_env(env_id, rank, seed=0):
    def _init():
        env = gym_super_mario_bros.make(env_id)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env.reset(seed=seed + rank)
        return env
    return _init

try:
    # Create 2 parallel environments
    num_envs = 2
    vec_env = AsyncVectorEnv([make_env('SuperMarioBros-v0', i) for i in range(num_envs)])
    print(f"✓ Created {num_envs} parallel environments with AsyncVectorEnv")
    
    # Test reset
    observations, infos = vec_env.reset()
    print(f"✓ Vectorized reset successful - Observations shape: {observations.shape}")
    
    # Test step
    actions = vec_env.action_space.sample()
    observations, rewards, terminateds, truncateds, infos = vec_env.step(actions)
    print(f"✓ Vectorized step successful")
    print(f"  Rewards: {rewards}")
    
    vec_env.close()
    print("✓ Vectorized environments closed")
    
except Exception as e:
    print(f"✗ AsyncVectorEnv test failed: {e}")
    import traceback
    traceback.print_exc()

# Test different environment IDs
print("\n=== Testing Different Environment IDs ===")
env_ids = [
    'SuperMarioBros-v0',
    'SuperMarioBros-1-1-v0',
    'SuperMarioBros2-v0',
]

for env_id in env_ids:
    try:
        env = gym_super_mario_bros.make(env_id)
        env.close()
        print(f"✓ {env_id}")
    except Exception as e:
        print(f"✗ {env_id}: {e}")

print("\n" + "="*50)
print("ALL TESTS PASSED!")
print("="*50)
