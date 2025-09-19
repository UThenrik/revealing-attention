"""
Custom policies for reinforcement learning.

This module contains custom policy implementations for DQN and other RL algorithms.
"""

import torch
from stable_baselines3.dqn.policies import DQNPolicy
from .rl_networks import MyDuelingQNetwork


class MyDQNPolicy(DQNPolicy):
    """Custom DQN policy with CTR strategy support."""
    
    def make_q_net(self):
        """Create the Q-network."""
        features_extractor = self.make_features_extractor()
        return MyDuelingQNetwork(features_extractor, self.action_space.n)

    def set_CTR_strategy(self, use_CTR):
        """Set whether to use CTR strategy in the Q network."""
        self.use_CTR_strategy = use_CTR

    def forward(self, obs, deterministic=False):
        """Forward pass for training."""
        q_values = self.q_net(obs)
        return q_values.argmax(dim=1), None

    def _predict(self, obs, deterministic=False):
        """Forward pass for prediction."""
        q_out = self.q_net(obs, predict=True).argmax(dim=1)
        return q_out
