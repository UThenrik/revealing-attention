"""
Replay buffers for reinforcement learning.

This module contains custom replay buffer implementations including
prioritized experience replay and step tracking.
"""

import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from gymnasium import spaces


class MyReplayBuffer(ReplayBuffer):
    """Replay buffer that tracks training steps for each experience."""
    
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        config: dict = {},
    ):
        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs,
            optimize_memory_usage, handle_timeout_termination
        )
        
        # Add storage for training steps
        self.training_steps = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
        training_step: int
    ) -> None:
        """Add a transition to the buffer with training step tracking."""
        super().add(obs, next_obs, action, reward, done, infos)
        self.training_steps[self.pos] = np.array([training_step] * self.n_envs)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> Tuple[ReplayBufferSamples, np.ndarray]:
        """Get samples with corresponding training steps."""
        samples = super()._get_samples(batch_inds, env)
        
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        
        # Get corresponding training steps
        steps = self.training_steps[batch_inds, env_indices]
        
        return samples, steps

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None, return_steps=False) -> Tuple[ReplayBufferSamples, np.ndarray]:
        """Sample from the buffer with optional step tracking."""
        if not self.optimize_memory_usage:
            batch_inds = np.random.randint(0, self.buffer_size if self.full else self.pos, size=batch_size)
        else:
            if self.full:
                batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
            else:
                batch_inds = np.random.randint(0, self.pos, size=batch_size)
        
        if return_steps:
            return self._get_samples(batch_inds, env)
        else:
            return self._get_samples(batch_inds, env)[0]


class PrioritizedReplayBuffer(MyReplayBuffer):
    """Prioritized Experience Replay buffer implementation."""
    
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment_per_sampling: float = 1e-5,
        epsilon: float = 1e-6,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        config: dict = {},
    ):
        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs,
            optimize_memory_usage, handle_timeout_termination, config
        )
        
        self.alpha = alpha
        self.beta = beta
        # Calculate beta increment based on total training steps
        self.beta_increment_per_sampling = (
            (1 - self.beta) / (config["total_timesteps"] - config["learning_starts"]) * config["train_freq"]
        )
        self.epsilon = epsilon

        self.priorities = np.zeros((self.buffer_size,), dtype=np.float32)
        self.max_priority = 1.0

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
        training_step: int
    ) -> None:
        """Add a transition with maximum priority."""
        current_pos = self.pos
        super().add(obs, next_obs, action, reward, done, infos, training_step)
        self.priorities[current_pos] = self.max_priority

    def sample(
        self, batch_size: int, env: Optional[VecNormalize] = None, return_steps: bool = False
    ) -> Union[ReplayBufferSamples, Tuple[ReplayBufferSamples, np.ndarray], Tuple[ReplayBufferSamples, np.ndarray, np.ndarray, np.ndarray]]:
        """Sample from the buffer using prioritized sampling."""
        if self.full:
            prios = self.priorities
            total = self.buffer_size
        else:
            prios = self.priorities[:self.pos]
            total = self.pos

        # Calculate sampling probabilities
        probs = prios ** self.alpha
        probs /= probs.sum()

        # Sample indices based on probabilities
        batch_inds = np.random.choice(total, batch_size, p=probs)
        samples, steps = self._get_samples(batch_inds, env)

        # Calculate importance sampling weights
        weights = (total * probs[batch_inds]) ** (-self.beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(-1)

        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        if return_steps:
            return samples, steps, weights, batch_inds
        else:
            return samples, weights, batch_inds

    def update_priorities(self, batch_inds: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities based on TD errors."""
        td_errors = np.abs(td_errors) + self.epsilon
        self.priorities[batch_inds] = td_errors
        self.max_priority = max(self.max_priority, td_errors.max())
