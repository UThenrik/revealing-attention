"""
Environment wrappers for reinforcement learning.

This module contains custom Gymnasium wrappers for Atari environments,
including reward tracking and shaping functionality.
"""

import gymnasium as gym
import numpy as np


class FreewayRAMWrapper(gym.Wrapper):
    """Wrapper for Freeway that provides reward shaping based on chicken position."""
    
    def __init__(self, env):
        super().__init__(env)
        self.last_y = 0
    
    def step(self, action):
        """Step the environment with reward shaping."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        ram = self.env.unwrapped.ale.getRAM()
        current_y = ram[14]  # Freeway chicken Y-position
        
        # Reward shaping based on movement
        if current_y > self.last_y:
            reward += 0.01  # Reward for moving up
        elif current_y < self.last_y:
            reward += 0  # No penalty for moving down
        
        self.last_y = current_y
        return obs, reward, terminated, truncated, info


class RawRewardTracker(gym.Wrapper):
    """Wrapper that tracks both raw and clipped rewards for logging purposes."""
    
    def __init__(self, env):
        super().__init__(env)
        self._raw_episode_sum = 0
        self._clipped_episode_sum = 0

    def reset(self, **kwargs):
        """Reset the environment and reward tracking."""
        self._raw_episode_sum = 0
        self._clipped_episode_sum = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        """Step the environment and track rewards."""
        obs, rew, terminated, truncated, info = self.env.step(action)
        
        clipped_rew = np.sign(rew)
        self._raw_episode_sum += rew
        self._clipped_episode_sum += clipped_rew

        # Add reward information to info dict
        info["raw_reward"] = rew
        info["clipped_reward"] = clipped_rew

        if terminated or truncated:
            info["raw_episode_reward"] = self._raw_episode_sum
            info["clipped_episode_reward"] = self._clipped_episode_sum
            self._raw_episode_sum = 0
            self._clipped_episode_sum = 0

        return obs, rew, terminated, truncated, info
