# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [8.0.0] - 2025-01-10

### Breaking Changes
- **Migrated from OpenAI Gym to Gymnasium**: The project now uses `gymnasium` instead of the deprecated `gym` library
  - All gym dependencies replaced with gymnasium equivalents
  - Import patterns remain the same but now use gymnasium under the hood
- **Updated nes-py dependency**: Now requires `nes-py>=9.0.0` which includes gymnasium support
- **Updated API signatures** (inherited from nes-py):
  - `reset()` now returns `(observation, info)` tuple instead of just `observation`
  - `step()` now returns `(observation, reward, terminated, truncated, info)` instead of `(observation, reward, done, info)`
- **Python version requirements**: Dropped support for Python 3.5-3.8, now requires Python 3.9+

### Added
- **Python 3.13 support**: Full compatibility with Python 3.13
- **SubprocVecEnv support**: Full support for parallel environment execution
  - Inherited from nes-py 9.0.0 pickling support
  - Can be used with `gymnasium.vector.SubprocVecEnv`
  - Example code provided in `examples/vectorized_env_example.py`
- Support for Python 3.10, 3.11, and 3.12
- Example script demonstrating parallel environment usage

### Changed
- **Updated dependencies to latest versions**:
  - `gym>=0.10.9` → `gymnasium>=1.0.0`
  - `nes-py>=4.0.0` → `nes-py>=9.0.0`
  - `numpy>=1.14.2` → `numpy>=2.2.1`
  - `matplotlib>=2.0.2` → `matplotlib>=3.10.0`
  - `opencv-python>=3.4.0.12` → `opencv-python>=4.11.0`
  - `pygame>=1.9.3` → `pygame>=2.6.1`
  - `pyglet>=1.3.2` → `pyglet>=2.1.0`
  - `setuptools>=39.0.1` → `setuptools>=75.8.0`
  - `tqdm>=4.19.5` → `tqdm>=4.67.1`
  - `twine>=1.11.0` → `twine>=6.0.1`
- Package description updated from "OpenAI Gym" to "Gymnasium"
- Keywords updated to reference Gymnasium instead of OpenAI-Gym
- **SuperMarioBrosRandomStagesEnv** migrated to Gymnasium API:
  - Updated imports from `gym` to `gymnasium`
  - Updated `reset()` method to return `(observation, info)` tuple
  - Removed deprecated `return_info` parameter

### Fixed
- **NumPy 2.x compatibility issues** in `smb_env.py`:
  - Fixed overflow errors in `_x_position` and `_left_x_position` properties
  - Properly converts numpy scalar types to Python integers

### Migration Guide

#### Updating Your Code

**Old code (v7.x with gym):**
```python
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

observation = env.reset()
for step in range(5000):
    observation, reward, done, info = env.step(env.action_space.sample())
    if done:
        observation = env.reset()

env.close()
```

**New code (v8.x with gymnasium):**
```python
import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

observation, info = env.reset()
for step in range(5000):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
    if done:
        observation, info = env.reset()

env.close()
```

#### Using with SubprocVecEnv for Parallel Training

```python
import gymnasium as gym
from gymnasium.vector import SubprocVecEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

def make_env(env_id, rank, seed=0):
    def _init():
        env = gym_super_mario_bros.make(env_id)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env.reset(seed=seed + rank)
        return env
    return _init

# Create 4 parallel environments
num_envs = 4
env = SubprocVecEnv([make_env('SuperMarioBros-v0', i) for i in range(num_envs)])

observations, infos = env.reset()
observations, rewards, terminateds, truncateds, infos = env.step(actions)

env.close()
```

See `examples/vectorized_env_example.py` for a complete working example.

### Removed
- Support for Python 3.5, 3.6, 3.7, and 3.8
- Compatibility with OpenAI Gym (use gymnasium instead)

### Technical Notes
- All code has been tested with Python 3.13 and the latest versions of dependencies
- The underlying NES emulator remains unchanged
- ROM files are properly accessible in multiprocessing scenarios
- All 512 registered environments work correctly with the new API
- Inherits full pickling support from nes-py 9.0.0 for multiprocessing
- Note: Gymnasium uses `AsyncVectorEnv` instead of `SubprocVecEnv` (example updated accordingly)

## [7.4.0] and earlier

See previous releases for older changes.
