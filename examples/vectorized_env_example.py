"""Example of using Super Mario Bros with vectorized environments (AsyncVectorEnv)."""
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.
    
    Args:
        env_id (str): the environment ID
        rank (int): index of the subprocess
        seed (int): the initial seed for RNG
    
    Returns:
        function: a function that will create the environment
    """
    def _init():
        env = gym_super_mario_bros.make(env_id)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    """Run parallel environments example."""
    # Configuration
    num_envs = 4
    env_id = 'SuperMarioBros-v0'
    
    print(f"Creating {num_envs} parallel environments...")
    
    # Create vectorized environment with 4 parallel processes
    env = AsyncVectorEnv([make_env(env_id, i) for i in range(num_envs)])
    
    print("Environments created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Reset all environments
    observations, infos = env.reset()
    print(f"\nReset complete. Observations shape: {observations.shape}")
    
    # Run for some steps
    num_steps = 100
    print(f"\nRunning {num_steps} steps across {num_envs} environments...")
    
    for step in range(num_steps):
        # Sample random actions for all environments
        actions = env.action_space.sample()
        
        # Step all environments
        observations, rewards, terminateds, truncateds, infos = env.step(actions)
        
        # Print info every 20 steps
        if step % 20 == 0:
            print(f"Step {step}: Rewards = {rewards}")
            
        # Check if any environment is done
        if any(terminateds) or any(truncateds):
            print(f"Step {step}: Some environments finished")
            # Environments are automatically reset by SubprocVecEnv
    
    print("\nClosing environments...")
    env.close()
    print("Done!")


if __name__ == '__main__':
    main()
