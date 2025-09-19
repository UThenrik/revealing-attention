"""
Custom RL model implementations.

This module contains custom implementations of RL algorithms,
particularly DQN with attention mechanisms and custom training loops.
"""

import torch
import torch.nn.functional as F
import numpy as np
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.common.buffers import ReplayBuffer
from typing import Any, Dict, List, Optional, Tuple, Union
from .rl_networks import MyDuelingQNetwork
from .rl_buffers import MyReplayBuffer, PrioritizedReplayBuffer


class MyDQNModel(DQN):
    """Custom DQN implementation with attention mechanisms and step tracking."""
    
    def _on_step(self) -> None:
        """Update exploration rate and target network with step tracking."""
        self._n_calls += 1
        self.policy.q_net.features_extractor.step_counter = self._n_calls * self.n_envs
        self.policy.q_net_target.features_extractor.step_counter = self._n_calls * self.n_envs
        
        # Update target network
        if self._n_calls % max(self.target_update_interval // self.n_envs, 1) == 0:
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)

            # Handle batch norm running stats
            feat_stats = set(get_parameters_by_name(self.q_net.features_extractor, ["running_"]))
            all_running_params = get_parameters_by_name(self.q_net, ["running_"])
            if len(all_running_params) > len(feat_stats):
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        # Update exploration rate
        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

        # Log CTR lambda if using CTR strategy
        if self.policy.use_CTR_strategy:
            self.logger.record("rollout/current_lambda", 
                             self.policy.q_net.features_extractor.lam_scheduler(self._n_calls * self.n_envs))

        # Log PER beta if using prioritized replay
        if self.config["use_PER"]:
            self.logger.record("rollout/current_PER_beta", self.replay_buffer.beta)

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        """Store transition with training step tracking."""
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Handle terminal observations
        next_obs = new_obs_.copy()
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        replay_buffer.add(
            self._last_original_obs,
            next_obs,
            buffer_action,
            reward_,
            dones,
            infos,
            training_step=self._n_calls * self.n_envs,
        )

        self._last_obs = new_obs
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """Train the model with optional prioritized experience replay."""
        if self.config["use_PER"]:
            self.train_PER(gradient_steps, batch_size)
        else:
            self.train_normal(gradient_steps, batch_size)

    def train_normal(self, gradient_steps: int, batch_size: int = 100) -> None:
        """Standard DQN training with Double DQN."""
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data, steps = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env, return_steps=True)

            with torch.no_grad():
                # Double DQN: use online net to select actions, target net to evaluate
                next_q_online = self.q_net(replay_data.next_observations, steps=steps)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)
                next_q_target = self.q_net_target(replay_data.next_observations, steps=steps)
                next_q_values = torch.gather(next_q_target, dim=1, index=next_actions)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values
            current_q_values = self.q_net(replay_data.observations, steps=steps)
            current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute loss
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize
            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Update priorities for PER
        with torch.no_grad():
            td_errors = torch.abs(current_q_values - target_q_values)
        
        if hasattr(self.replay_buffer, "update_priorities"):
            priorities = td_errors.detach().squeeze().cpu().numpy()
            self.replay_buffer.update_priorities(replay_data.indices, priorities)

        self._log_q_values_simple(current_q_values)
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    def train_PER(self, gradient_steps: int, batch_size: int = 100) -> None:
        """Training with Prioritized Experience Replay."""
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample from PER buffer
            replay_data, steps, weights, indices = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env, return_steps=True
            )

            with torch.no_grad():
                # Double DQN
                next_q_online = self.q_net(replay_data.next_observations, steps=steps)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)
                next_q_target = self.q_net_target(replay_data.next_observations, steps=steps)
                next_q_values = torch.gather(next_q_target, dim=1, index=next_actions)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Q-value estimates
            current_q_values = self.q_net(replay_data.observations, steps=steps)
            current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # TD error
            td_errors = current_q_values - target_q_values

            # Weighted loss
            loss = (weights * F.smooth_l1_loss(current_q_values, target_q_values, reduction="none")).mean()
            losses.append(loss.item())

            # Optimize
            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            # Update priorities
            if hasattr(self.replay_buffer, "update_priorities"):
                new_priorities = td_errors.detach().squeeze().abs().cpu().numpy()
                self.replay_buffer.update_priorities(indices, new_priorities)

        self._log_q_values_simple(current_q_values)
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    def _log_q_values_simple(self, current_q_values, prefix="train"):
        """Log Q-value statistics."""
        q_values_np = current_q_values.detach().cpu().numpy()
        mean = np.mean(q_values_np)
        std = np.std(q_values_np)
        self.logger.record_mean(f"{prefix}/q_values/mean", mean)
        self.logger.record_mean(f"{prefix}/q_values/std", std)

    def _log_q_values_by_lambda(self, current_q_values, current_actions, lam, prefix="train", n_bins=8):
        """Log Q-values grouped by lambda values."""
        if lam is None:
            return
        
        lam_np = lam.detach().cpu().numpy().flatten()
        q_values_np = current_q_values.detach().cpu().numpy()
        actions_np = current_actions.detach().cpu().numpy().flatten()

        # Sort by lambda and split into bins
        sorted_indices = np.argsort(lam_np)
        bin_size = len(lam_np) // n_bins

        for i in range(n_bins):
            bin_indices = sorted_indices[i * bin_size:(i + 1) * bin_size]
            if len(bin_indices) == 0:
                continue

            q_bin = q_values_np[bin_indices, actions_np[bin_indices]]
            mean = np.mean(q_bin)
            std = np.std(q_bin)
            self.logger.record_mean(f"{prefix}/lambda_q_values/q{i+1}_mean", mean)
            self.logger.record_mean(f"{prefix}/lambda_q_values/q{i+1}_std", std)

        # Action agreement analysis
        action_bins = [
            actions_np[sorted_indices[i * bin_size:(i + 1) * bin_size]]
            for i in range(n_bins)
        ]
        reference_actions = action_bins[-1]
        for i, actions in enumerate(action_bins):
            if len(actions) > 0 and len(reference_actions) > 0:
                min_len = min(len(actions), len(reference_actions))
                agree = np.sum(actions[:min_len] == reference_actions[:min_len]) / min_len
                self.logger.record_mean(f"{prefix}/lambda_action_agreement/q{i+1}", agree)
