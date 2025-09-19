"""
Training classes and utilities for neural network training.

This module contains training classes for various neural network architectures
including attention mechanisms and motor predictors.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .models import Motor_predictor_fwd, CTR_Attention_dil, CTR_Attention_SA


class Attention_training_class:
    """Training class for attention-based neural networks."""
    
    def __init__(self, autoencoder, gaze_predictor, action_dimAA, action_dimAH, lr, config):
        self.n_MP_samples = 6
        self.flag_load_CTR = False
        
        self.autoencoder = autoencoder
        self.gaze_predictor = gaze_predictor
        self.gaze_predictor.eval()
        self.autoencoder.eval()
        _, repr_shape = self.autoencoder.give_repr_size()

        # Motor predictors
        self.MP_AA = Motor_predictor_fwd(action_dimAA, repr_shape, game_name=config["game_name"]).to(autoencoder.device)
        self.MP_AH = Motor_predictor_fwd(action_dimAH, repr_shape, game_name=config["game_name"]).to(autoencoder.device)
        self.MP_AA_plain = Motor_predictor_fwd(action_dimAA, repr_shape, game_name=config["game_name"]).to(autoencoder.device)
        self.MP_AH_plain = Motor_predictor_fwd(action_dimAH, repr_shape, game_name=config["game_name"]).to(autoencoder.device)
        self.MP_AH_GP = Motor_predictor_fwd(action_dimAH, repr_shape, game_name=config["game_name"]).to(autoencoder.device)

        # Attention mechanisms
        if config["CTR_type"] == 'dil':
            self.CTR_AA = CTR_Attention_dil(repr_shape, config, autoencoder, autoencoder.device).to(autoencoder.device)
            self.CTR_AH = CTR_Attention_dil(repr_shape, config, autoencoder, autoencoder.device).to(autoencoder.device)
        elif config["CTR_type"] == 'SA':
            self.CTR_AA = CTR_Attention_SA(repr_shape, config, autoencoder, autoencoder.device).to(autoencoder.device)
            self.CTR_AH = CTR_Attention_SA(repr_shape, config, autoencoder, autoencoder.device).to(autoencoder.device)

        # Optimizers
        self.MP_AA_optimizer = optim.AdamW(self.MP_AA.parameters(), lr=lr, weight_decay=1e-2)
        self.MP_AH_optimizer = optim.AdamW(self.MP_AH.parameters(), lr=lr, weight_decay=1e-2)
        self.MP_AA_plain_optimizer = optim.AdamW(self.MP_AA_plain.parameters(), lr=lr, weight_decay=1e-2)
        self.MP_AH_plain_optimizer = optim.AdamW(self.MP_AH_plain.parameters(), lr=lr, weight_decay=1e-2)
        self.MP_AH_GP_optimizer = optim.AdamW(self.MP_AH_GP.parameters(), lr=lr, weight_decay=1e-2)
        self.CTR_AA_optimizer = optim.AdamW(self.CTR_AA.parameters(), lr=lr/2, weight_decay=1e-2)
        self.CTR_AH_optimizer = optim.AdamW(self.CTR_AH.parameters(), lr=lr/2, weight_decay=1e-2)

    def train_AE_att(self, j_epoch, replay_bufferAA, replay_bufferAH, batch_size, config, class_weightsAA, class_weightsAH, GP_comparison=False):
        """Train the attention-based autoencoder."""
        flag_sample_source = True  # flag to actually sample the source, slower but technically better
        results = [[[], []], [[], []]]

        for replay_buffer, CTR_att, motor_predictor, CTR_optimizer, MP_optimizer, MP_plain, MP_plain_optimizer, class_weights, id_res in zip(
            [replay_bufferAA, replay_bufferAH],
            [self.CTR_AA, self.CTR_AH],
            [self.MP_AA, self.MP_AH],
            [self.CTR_AA_optimizer, self.CTR_AH_optimizer],
            [self.MP_AA_optimizer, self.MP_AH_optimizer],
            [self.MP_AA_plain, self.MP_AH_plain],
            [self.MP_AA_plain_optimizer, self.MP_AH_plain_optimizer],
            [class_weightsAA, class_weightsAH],
            [0, 1]
        ):
            state_np, actions, _, _, _, done_np, train_flags = replay_buffer.sample_stacked_fwd(int(batch_size/config["train_val_split"]))
            state_all = torch.tensor(state_np, dtype=torch.float32, device=self.autoencoder.device, requires_grad=False)/255.0*2-1
            action_all = torch.tensor(actions, device=self.autoencoder.device, dtype=torch.int64, requires_grad=False)
            done_all = torch.tensor(done_np, device=self.autoencoder.device, dtype=torch.bool, requires_grad=False)
            non_final_mask_all = ~done_all

            if id_res == 0:
                if np.random.rand() > 0.5:
                    state_source_np, _, _, _, _, done_source_np, _ = replay_bufferAA.sample_stacked_fwd(int(batch_size/config["train_val_split"]))
                else:
                    state_source_np, _, _, _, _, done_source_np, _ = replay_bufferAH.sample_stacked_fwd(int(batch_size/config["train_val_split"]))

                if flag_sample_source:
                    state_source_all = torch.tensor(state_source_np, dtype=torch.float32, device=self.autoencoder.device, requires_grad=False)/255.0*2-1    
                    done_source_all = torch.tensor(done_source_np, device=self.autoencoder.device, dtype=torch.bool, requires_grad=False)
                    non_final_mask_source_all = ~done_source_all
                else:
                    state_source_all = state_all
                    non_final_mask_source_all = non_final_mask_all

            for sample_mode, state, action, non_final_mask, state_source, non_final_mask_source, id_train in zip(
                ['train', 'val'],
                [state_all[train_flags], state_all[~train_flags]],
                [action_all[train_flags], action_all[~train_flags]],
                [non_final_mask_all[train_flags], non_final_mask_all[~train_flags]],
                [state_source_all[train_flags], state_source_all[~train_flags]],
                [non_final_mask_source_all[train_flags], non_final_mask_source_all[~train_flags]],
                [0, 1]
            ):
                if id_res == 0 or id_res == 1:
                    if sample_mode == 'val':
                        motor_predictor.eval()
                        CTR_att.eval()
                        MP_plain.eval()
                        self.MP_AH_GP.eval()

                    actions_criterion = nn.CrossEntropyLoss()
                    
                    # Train motor predictor and CTR attention    
                    with torch.no_grad():
                        repr = self.autoencoder.encode(state, flatten=False)
                        repr_source = self.autoencoder.encode(state_source, flatten=False)

                    # Train MP and TD attention
                    nfma = non_final_mask  # non-final-mask-all

                    lam_basic = torch.rand(self.n_MP_samples, 1, device=self.autoencoder.device)
                    
                    batch_size_current = state.shape[0]
                    n_repeats = batch_size_current // self.n_MP_samples
                    n_remainder = batch_size_current % self.n_MP_samples
                    lam = lam_basic.repeat(n_repeats, 1)
                    if n_remainder > 0:
                        lam = torch.cat([lam, lam_basic[:batch_size_current % self.n_MP_samples]], dim=0)

                    lam_nfma = lam[nfma]
                    
                    if config["train_additional_MPs"]:
                        if nfma.sum() == 0:
                            loss_MP_plain = torch.tensor(float('nan'))
                            loss_MP_plain_full = torch.tensor(float('nan'))
                            loss_MP_GP = torch.tensor(float('nan'))
                        else:
                            repr_psi = CTR_att.forward(state[nfma], lam_nfma)
                            action_pred_plain = MP_plain(repr_psi, lam_nfma)
                            loss_MP_plain = actions_criterion(action_pred_plain, action[nfma].flatten())

                            action_pred_plain_full = MP_plain(repr.detach().clone(), lam=torch.ones(repr.shape[0], 1, device=self.autoencoder.device))
                            loss_MP_plain_full = actions_criterion(action_pred_plain_full[nfma], action[nfma].flatten())
                            
                            acc_plain = (action_pred_plain.argmax(dim=1) == action[nfma].flatten()).float().mean().detach().cpu().numpy()

                            if id_res == 1:
                                with torch.no_grad():
                                    gaze_pred = self.gaze_predictor(state[nfma])
                                    # Apply topk binarization if available
                                    if hasattr(self.gaze_predictor, 'topk_binarize'):
                                        gaze_pred = self.gaze_predictor.topk_binarize(gaze_pred, lam_nfma)
                                    repr_GP = (repr[nfma] * gaze_pred.unsqueeze(1)).flatten(start_dim=1)
                                action_pred_GP = self.MP_AH_GP(repr_GP, lam_nfma)
                                loss_MP_GP = actions_criterion(action_pred_GP, action[nfma].flatten())
                                acc_GP = (action_pred_GP.argmax(dim=1) == action[nfma].flatten()).float().mean().detach().cpu().numpy()

                                self.MP_AH_GP_optimizer.zero_grad()
                                loss_MP_GP.backward()
                                torch.nn.utils.clip_grad_norm_(self.MP_AH_GP.parameters(), 1.0)
                                self.MP_AH_GP_optimizer.step()
                            else:
                                loss_MP_GP = torch.zeros(1, device=self.autoencoder.device)
                                acc_GP = 0.0
                    else:
                        loss_MP_plain = torch.zeros(1)
                        loss_MP_plain_full = torch.zeros(1)
                        loss_MP_GP = torch.zeros(1)
                        acc_plain = 0.0
                        acc_GP = 0.0

                    idx_shuffle = np.random.permutation(np.arange(0, nfma.sum().item()))

                    repr_psio, psi, psi_source = CTR_att.psi_overlayed_repr_fwd(
                        state[nfma],
                        repr[nfma],
                        state_source[nfma][idx_shuffle],
                        repr_source[nfma][idx_shuffle],
                        lam_nfma
                    ) 
                    
                    action_pred_psio = motor_predictor(repr_psio.flatten(start_dim=1), lam_nfma)
                    loss_CE_all = actions_criterion(action_pred_psio, action[nfma].flatten())
                    acc_CTR = (action_pred_psio.argmax(dim=1) == action[nfma].flatten()).float().mean().detach().cpu().numpy()

                    action_pred_full = motor_predictor(repr.flatten(start_dim=1), lam=torch.ones(repr.shape[0], 1, device=self.autoencoder.device))
                    loss_CE_full = actions_criterion(action_pred_full[nfma], action[nfma].flatten())
                    acc_full = (action_pred_full[nfma].argmax(dim=1) == action[nfma].flatten()).float().mean().detach().cpu().numpy()
                    
                    mean_psi = torch.mean(psi)
                    means_psi = torch.mean(psi.flatten(1), dim=1)            

                    losses_lam_target = torch.zeros(self.n_MP_samples, device=self.autoencoder.device)
                    for i_lam in range(self.n_MP_samples):
                        mpsi = torch.mean(means_psi[i_lam::self.n_MP_samples])
                        lamb = lam_basic[i_lam].flatten()

                        both = torch.abs(mpsi - lamb) / (lamb + 1e-6)
                        only_negative = torch.abs(mpsi - lamb) / (lamb + 1e-6) if mpsi > lamb else 0.0
                        beta = np.max([0, 1-j_epoch/100])
                        
                        losses_lam_target[i_lam] = beta*both + (1-beta)*only_negative
                    loss_lam_target = losses_lam_target.mean()

                    loss_total = loss_CE_all + loss_lam_target + loss_CE_full
                    if config["train_additional_MPs"]:
                        loss_total += loss_MP_plain + loss_MP_plain_full

                    if sample_mode == 'train':
                        if j_epoch < config["n_epochs_CTR"]:
                            CTR_optimizer.zero_grad()
                            MP_optimizer.zero_grad()
                        if config["train_additional_MPs"]:
                            MP_plain_optimizer.zero_grad()

                        loss_total.backward()
                        
                        if j_epoch < config["n_epochs_CTR"]:
                            torch.nn.utils.clip_grad_norm_(CTR_att.parameters(), 1.0)
                            torch.nn.utils.clip_grad_norm_(motor_predictor.parameters(), 1.0)
                        if config["train_additional_MPs"]:
                            torch.nn.utils.clip_grad_norm_(MP_plain.parameters(), 1.0)
                        
                        if j_epoch < config["n_epochs_CTR"]:
                            CTR_optimizer.step()
                            MP_optimizer.step()
                        if config["train_additional_MPs"]:
                            MP_plain_optimizer.step()
                    
                    if sample_mode == 'val':
                        motor_predictor.train()
                        CTR_att.train()
                        MP_plain.train()
                        self.MP_AH_GP.train()

                    results[id_res][id_train] = (mean_psi.item(), loss_lam_target.item(), loss_CE_all.item(), acc_CTR, loss_CE_full.item(), acc_full, loss_MP_plain.item(), acc_plain, loss_MP_GP.item(), acc_GP)
                else:
                    results[id_res][id_train] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        return np.array(results)
