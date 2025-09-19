"""
Neural network architectures for reinforcement learning.

This module contains custom network architectures including feature extractors
and Q-networks for attention-guided RL training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class AutoencoderFeatureExtractor(BaseFeaturesExtractor):
    """Feature extractor that uses a pre-trained autoencoder with optional CTR attention."""
    
    def __init__(self, observation_space, autoencoder, lam_scheduler, device='cpu', CTR=None, config=None):
        self.use_CTR = config.get("use_CTR", False)
        self.beta_res_attention = config.get("beta_res_attention", 0.0)
        self.inverse_psi = config.get("inverse_psi", False)
        self.step_counter = 0

        # Pre-compute features dimension
        dummy_obs = torch.zeros(1, *observation_space.shape, device=device)
        dummy_obs = dummy_obs.float() / 255.0 * 2 - 1
        dummy_obs = dummy_obs.permute(0, 2, 3, 1)
        dummy_enc = autoencoder.to(device).encode(dummy_obs, flatten=False)
        features_dim = dummy_enc.flatten(start_dim=1).shape[1]
        
        super().__init__(observation_space, features_dim=features_dim)
        
        self.autoencoder = autoencoder.to(device)
        self.lam_scheduler = lam_scheduler
        self.device = device
        self.CTR = CTR.to(device) if CTR is not None else None

        # Freeze autoencoder parameters
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.autoencoder.eval()

        # Freeze CTR parameters if used
        if self.CTR is not None:
            for param in self.CTR.parameters():
                param.requires_grad = False
            self.CTR.eval()

        # Freeze BatchNorm running stats
        for name, buffer in self.autoencoder.named_buffers():
            if "running_mean" in name or "running_var" in name:
                buffer.requires_grad = False
                buffer.detach_()

        if self.CTR is not None:
            for name, buffer in self.CTR.named_buffers():
                if "running_mean" in name or "running_var" in name:
                    buffer.requires_grad = False
                    buffer.detach_()

    def forward(self, observations, steps=None, flatten=True, predict=False):
        """Extract features from observations."""
        x = observations.float() / 255.0 * 2 - 1
        x = x.permute(0, 2, 3, 1)

        if self.use_CTR:
            if predict:
                lam_current = self.lam_scheduler(self.step_counter)
                lam = lam_current * torch.ones((x.shape[0], 1), device=self.device, dtype=torch.float32)
            else:
                lam_current = self.lam_scheduler(steps)
                lam = torch.tensor(lam_current, device=self.device, dtype=torch.float32).unsqueeze(1)

            psi = self.CTR.psi(x, lam=lam)
            if self.inverse_psi:
                psi = 1 - psi
            enc = self.autoencoder.encode(x, flatten=False)
            enc_out = enc * (psi + (1 - psi) * self.beta_res_attention)
        else:
            enc_out = self.autoencoder.encode(x, flatten=False)

        if flatten:
            return enc_out.flatten(start_dim=1)
        else:
            return enc_out


class MyDuelingQNetwork(nn.Module):
    """Dueling DQN architecture with custom feature processing."""
    
    def __init__(self, features_extractor, n_actions):
        super().__init__()
        self.features_extractor = features_extractor
        self.features_extractor.eval()

        # Freeze feature extractor parameters
        for param in self.features_extractor.parameters():
            param.requires_grad = False

        self.act = nn.GELU()
        self.n_neurons = 128

        # CNN layers for feature processing
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=(1, 1), stride=1, padding=(0, 0)),
            self.act,
        )
        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=(1, 1), stride=1, padding=(0, 0)),
            self.act,
        )

        # Fully connected layers
        self.fc1 = nn.Linear(int(self.features_extractor.features_dim / 32 * 8), self.n_neurons)
        self.fc2 = nn.Linear(self.n_neurons, self.n_neurons)

        # Dueling architecture heads
        self.advantage = nn.Linear(self.n_neurons, n_actions)
        self.value = nn.Linear(self.n_neurons, 1)

    def forward(self, obs, steps=None, predict=False):
        """Forward pass through the network."""
        features = self.features_extractor(obs, steps=steps, flatten=False, predict=predict)
        
        x = self.cnn_layer1(features)
        x = self.cnn_layer2(x)
        x = x.flatten(start_dim=1)

        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))

        # Dueling architecture
        adv = self.advantage(x)
        val = self.value(x)
        q_out = val + adv - adv.mean(dim=1, keepdim=True)

        return q_out
    
    def train(self, mode: bool = True):
        """Set training mode while keeping feature extractor in eval mode."""
        super().train(mode)
        self.features_extractor.eval()
        return self

    def set_training_mode(self, mode: bool):
        """Set training mode for trainable components only."""
        self.fc1.train(mode)
        self.fc2.train(mode)
        self.advantage.train(mode)
        self.value.train(mode)

    def parameters(self, recurse: bool = True):
        """Return parameters excluding those from the feature extractor."""
        all_params = list(super().parameters(recurse=recurse))
        feat_params = set(self.features_extractor.parameters(recurse=recurse))
        return (p for p in all_params if p not in feat_params)
