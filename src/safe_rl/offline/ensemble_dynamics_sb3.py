#!/usr/bin/env python3
# =============================================================================
# Calibrated Safe Policy Improvement for Continuous Offline Reinforcement Learning
# Federico Bianchi, Alessandro Farinelli, Alberto Castellini
# NeurIPS 2026
# =============================================================================
"""Deterministic K-member dynamics ensemble predicting s' - s.

Members are trained with MSE, i.e. a fixed-covariance Gaussian likelihood, rather than
a heteroscedastic NLL, so that disagreement between member means remains the
uncertainty signal used by the gate.

Provides next-state prediction (single member or ensemble mean) and the uncertainty
scores u_USAD (max pairwise L2 distance between member means) and u_VAR (mean
per-dimension variance). Only the "observation" entry is modelled; goal entries are
handled by the model environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def _to_torch(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _pairwise_max_l2(v: np.ndarray) -> float:
    """Return max_{i<j} ||v_i - v_j||_2 for member predictions v of shape [E, D]."""
    E = v.shape[0]
    if E <= 1:
        return 0.0
    maxd = 0.0
    for i in range(E):
        for j in range(i + 1, E):
            d = float(np.linalg.norm(v[i] - v[j], ord=2))
            if d > maxd:
                maxd = d
    return maxd


class DynMLP(nn.Module):
    """ReLU MLP mapping (s, a) to a deterministic delta prediction.

    hidden_dims=(512, 512) gives the MOReL architecture; (200, 200, 200, 200) the MOPO one.
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dims: Tuple[int, ...] = (512, 512)):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Alias kept for older import sites.
MLP512 = DynMLP


@dataclass
class EnsembleTrainConfig:
    """Training configuration for EnsembleDynamicsSB3."""
    ensemble_size:    int   = 4
    hidden_dims:      Tuple[int, ...] = (512, 512)  # per member
    lr:               float = 1e-3
    batch_size:       int   = 256
    epochs:           int   = 50
    bootstrap:        bool  = True   # independent bootstrap resample per member
    normalize_inputs: bool  = True   # standardize (s, a) with training-set mean/std
    seed:             int   = 0
    device:           str   = "cpu"


class EnsembleDynamicsSB3:
    """Deterministic MSE-trained dynamics ensemble over flat observation vectors.

    Members differ by random initialization and, when bootstrap=True, by an independent
    bootstrap resample of the training set. Inputs (s, a) are standardized when
    normalize_inputs=True. Member k is seeded with cfg.seed + 1000 * k.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        cfg: Optional[EnsembleTrainConfig] = None,
    ):
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.cfg     = cfg if cfg is not None else EnsembleTrainConfig()
        self.device  = torch.device(self.cfg.device)
        self.ensemble: List[nn.Module] = []
        self.optims:   List[optim.Optimizer] = []
        # Set by fit_from_transitions when normalize_inputs=True.
        self.input_mean: Optional[np.ndarray] = None
        self.input_std:  Optional[np.ndarray] = None
        self._build()

    def _build(self) -> None:
        _set_torch_seed(self.cfg.seed)
        self.ensemble, self.optims = [], []
        in_dim = self.obs_dim + self.act_dim
        out_dim = self.obs_dim
        for k in range(self.cfg.ensemble_size):
            _set_torch_seed(self.cfg.seed + 1000 * k)
            m = DynMLP(in_dim, out_dim, hidden_dims=tuple(self.cfg.hidden_dims)).to(self.device)
            self.ensemble.append(m)
            self.optims.append(optim.Adam(m.parameters(), lr=self.cfg.lr))

    def _pack_input(
        self,
        obs_dict: Dict[str, np.ndarray],
        action:   np.ndarray,
    ) -> np.ndarray:
        obs = np.asarray(obs_dict["observation"], dtype=np.float32)
        act = np.asarray(action, dtype=np.float32)
        if obs.ndim == 1: obs = obs[None, :]
        if act.ndim == 1: act = act[None, :]
        x = np.concatenate([obs, act], axis=-1)   # [1, in_dim]
        in_mean = getattr(self, "input_mean", None)
        in_std = getattr(self, "input_std", None)
        if in_mean is not None and in_std is not None:
            x = (x - in_mean) / in_std
        return x

    def fit_from_transitions(
        self,
        obs_dicts:      Dict[str, np.ndarray],
        actions:        np.ndarray,
        next_obs_dicts: Dict[str, np.ndarray],
        *,
        verbose: bool = True,
    ) -> None:
        """Train every member on delta = next_obs - obs with MSE loss.

        Args:
            obs_dicts: dict with "observation" of shape [N, obs_dim].
            actions: array of shape [N, act_dim].
            next_obs_dicts: dict with "observation" of shape [N, obs_dim].
            verbose: print normalization stats and periodic loss of member 0.
        """
        X = np.concatenate(
            [np.asarray(obs_dicts["observation"], np.float32),
             np.asarray(actions, np.float32)],
            axis=-1,
        )
        y = (np.asarray(next_obs_dicts["observation"], np.float32)
             - np.asarray(obs_dicts["observation"], np.float32))

        if self.cfg.normalize_inputs:
            self.input_mean = X.mean(axis=0, keepdims=True).astype(np.float32)
            self.input_std  = np.maximum(X.std(axis=0, keepdims=True), 1e-8).astype(np.float32)
            X = (X - self.input_mean) / self.input_std
            if verbose:
                print(f"[EnsembleDynamicsSB3] input normalization: "
                      f"mean_abs={np.abs(self.input_mean).mean():.4f} "
                      f"std_mean={self.input_std.mean():.4f}")

        N   = X.shape[0]
        bs  = int(self.cfg.batch_size)
        X_t = _to_torch(X, self.device)
        y_t = _to_torch(y, self.device)

        loss_fn = nn.MSELoss()
        rng_boot = np.random.default_rng(self.cfg.seed)

        for epoch in range(int(self.cfg.epochs)):
            for ki, (m, opt) in enumerate(zip(self.ensemble, self.optims)):
                if self.cfg.bootstrap:
                    boot_idx = rng_boot.integers(0, N, size=N)
                    Xk = X_t[boot_idx]
                    yk = y_t[boot_idx]
                    perm = torch.randperm(N, device=self.device)
                else:
                    # Shared mini-batch order across members.
                    if ki == 0:
                        perm = torch.randperm(N, device=self.device)
                    Xk, yk = X_t, y_t

                for start in range(0, N, bs):
                    idx = perm[start: start + bs]
                    xb = Xk[idx] if self.cfg.bootstrap else Xk[idx]
                    yb = yk[idx] if self.cfg.bootstrap else yk[idx]
                    opt.zero_grad(set_to_none=True)
                    loss_fn(m(xb), yb).backward()
                    opt.step()

            if verbose and (
                epoch % max(1, self.cfg.epochs // 10) == 0
                or epoch == self.cfg.epochs - 1
            ):
                with torch.no_grad():
                    sub = min(N, 1024)
                    l0 = loss_fn(self.ensemble[0](X_t[:sub]), y_t[:sub]).item()
                print(
                    f"[EnsembleDynamicsSB3] epoch={epoch:03d}/{self.cfg.epochs}"
                    f"  loss0={l0:.6f}"
                )

    @torch.no_grad()
    def _predict_all(
        self,
        obs_dict: Dict[str, np.ndarray],
        action:   np.ndarray,
    ) -> np.ndarray:
        """Return per-member delta predictions of shape [E, obs_dim]."""
        x  = self._pack_input(obs_dict, action)
        xt = _to_torch(x, self.device)
        preds = []
        for m in self.ensemble:
            p = m(xt).detach().cpu().numpy()[0]
            preds.append(p)
        return np.stack(preds, axis=0)   # [E, obs_dim]

    @torch.no_grad()
    def predict_next_obs_vec_mean(
        self,
        obs_dict: Dict[str, np.ndarray],
        action:   np.ndarray,
    ) -> np.ndarray:
        """Return obs + mean_k delta_k (ensemble-mean next observation)."""
        mus  = self._predict_all(obs_dict, action)
        obs  = np.asarray(obs_dict["observation"], np.float32)
        return obs + mus.mean(axis=0)

    @torch.no_grad()
    def sample_next_obs_vec(
        self,
        obs_dict:    Dict[str, np.ndarray],
        action:      np.ndarray,
        *,
        model_index: Optional[int]           = None,
        rng:         Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Predict the next observation with one ensemble member.

        model_index = k in 0..K-1 uses member k (TS-infinity, fixed per simulation);
        -1 uses the ensemble mean; None draws a member uniformly from rng.
        """
        mus = self._predict_all(obs_dict, action)   # [E, obs_dim]
        E   = mus.shape[0]

        if model_index == -1:
            delta = mus.mean(axis=0)
        else:
            if model_index is None:
                if rng is None:
                    rng = np.random.default_rng()
                model_index = int(rng.integers(0, E))
            model_index = int(np.clip(model_index, 0, E - 1))
            delta = mus[model_index]

        obs = np.asarray(obs_dict["observation"], np.float32)
        return obs + delta

    @torch.no_grad()
    def predict_next_obs_dict(
        self,
        obs_dict:    Dict[str, np.ndarray],
        action:      np.ndarray,
        *,
        model_index: Optional[int]                = None,
        rng:         Optional[np.random.Generator] = None,
    ) -> Dict[str, np.ndarray]:
        """Dict-level wrapper of sample_next_obs_vec; goal entries are copied unchanged."""
        obs_next = self.sample_next_obs_vec(
            obs_dict, action, model_index=model_index, rng=rng
        )
        return {
            "observation":   obs_next.astype(np.float32),
            "achieved_goal": np.asarray(obs_dict["achieved_goal"], np.float32).copy(),
            "desired_goal":  np.asarray(obs_dict["desired_goal"],  np.float32).copy(),
        }

    @torch.no_grad()
    def uncertainty_usad(
        self,
        obs_dict: Dict[str, np.ndarray],
        action:   np.ndarray,
    ) -> float:
        """Return u_USAD: max pairwise L2 distance between member predictions."""
        mus = self._predict_all(obs_dict, action)
        return _pairwise_max_l2(mus)

    @torch.no_grad()
    def uncertainty_epistemic(
        self,
        obs_dict: Dict[str, np.ndarray],
        action:   np.ndarray,
    ) -> float:
        """Return u_VAR: per-dimension variance across members, averaged over dimensions."""
        mus = self._predict_all(obs_dict, action)
        return float(mus.var(axis=0).mean())

    @torch.no_grad()
    def uncertainty_total(
        self,
        obs_dict: Dict[str, np.ndarray],
        action:   np.ndarray,
    ) -> float:
        """Same as uncertainty_epistemic; the ensemble has no aleatoric term."""
        return self.uncertainty_epistemic(obs_dict, action)
