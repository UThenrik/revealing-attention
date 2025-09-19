"""
Callbacks for reinforcement learning training.

This module contains custom callbacks for logging, model checkpointing,
and other training utilities.
"""

import os
import wandb
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class RawRewardLoggingCallback(BaseCallback):
    """Callback for logging raw and clipped rewards to WandB."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        """Log reward information at each step."""
        infos = self.locals.get("infos", [])
        for info in infos:
            log_data = {}

            # Per-step logs
            if "raw_reward" in info:
                log_data["charts/raw_reward"] = info["raw_reward"]
            if "clipped_reward" in info:
                log_data["charts/clipped_reward"] = info["clipped_reward"]

            # Episode-end logs
            if "raw_episode_reward" in info:
                log_data["charts/raw_episode_reward"] = info["raw_episode_reward"]
            if "clipped_episode_reward" in info:
                log_data["charts/clipped_episode_reward"] = info["clipped_episode_reward"]

            if log_data:
                wandb.log(log_data)

        return True


class WandbModelCheckpointCallback(BaseCallback):
    """Callback for saving model checkpoints during training."""
    
    def __init__(self, model_save_freq=100_000, base_save_dir="trained_models/DDQN", verbose=0):
        super().__init__(verbose)
        self.model_save_freq = model_save_freq
        self.base_save_dir = base_save_dir
        self.last_save_step = 0
        self.run_dir = None

    def _on_training_start(self) -> None:
        """Setup run-specific directory for model saving."""
        if wandb.run is not None:
            run_id = wandb.run.id
        else:
            run_id = "local_run"

        self.run_dir = os.path.join(self.base_save_dir, run_id)
        os.makedirs(self.run_dir, exist_ok=True)

    def _on_step(self) -> bool:
        """Save model checkpoint at specified intervals."""
        if (self.num_timesteps - self.last_save_step) >= self.model_save_freq:
            model_path = os.path.join(self.run_dir, f"model_{self.num_timesteps}.zip")
            self.model.save(model_path)

            if self.verbose > 0:
                print(f"Saved model checkpoint to {model_path}")

            self.last_save_step = self.num_timesteps

        return True
