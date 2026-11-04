# =============================================================================
# Calibrated Safe Policy Improvement for Continuous Offline Reinforcement Learning
# Federico Bianchi, Alessandro Farinelli, Alberto Castellini
# NeurIPS 2026
# =============================================================================
"""MSE dynamics ensemble with per-dimension target standardization (opt-in variant).

During training the regression target delta = s' - s is standardized per dimension, so
each state component contributes equally to the MSE loss regardless of its scale. After
fitting, the de-normalization is folded into the final linear layer,

    delta = sigma * (W h + b) + mu = (diag(sigma) W) h + (sigma * b + mu),

which is exact because the output layer is linear. The fitted model is therefore an
ordinary raw-delta predictor with the same weight layout as EnsembleDynamicsSB3.
Standardizing the target keeps the fixed-covariance Gaussian likelihood (diagonal,
data-independent covariance), so training remains MSE.

With scale_aware_uncertainty=True, u_USAD, u_VAR and the one-step calibration error are
computed on deltas divided by the per-dimension delta standard deviation. This changes
the units of the calibrated threshold eps; it is off by default.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from safe_rl.offline.ensemble_dynamics_sb3 import EnsembleDynamicsSB3, _to_torch


class EnsembleDynamicsV2(EnsembleDynamicsSB3):
    """MSE ensemble with per-dimension target standardization during training."""

    def __init__(self, *args, normalize_targets: bool = True,
                 scale_aware_uncertainty: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalize_targets = bool(normalize_targets)
        self.scale_aware_uncertainty = bool(scale_aware_uncertainty)
        self.target_mean: np.ndarray | None = None
        self.target_std: np.ndarray | None = None
        # Per-dimension std of delta, recorded at every fit; divisor of the
        # scale-aware uncertainty and calibration error.
        self.delta_scale: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # Scale-aware uncertainty and error
    # ------------------------------------------------------------------ #
    # Deltas are divided by delta_scale, making the measures dimensionless with
    # equal weight per component. eps values computed this way are not
    # comparable to those from the raw-delta metric.

    def _scaled(self, v: np.ndarray) -> np.ndarray:
        if not self.scale_aware_uncertainty or self.delta_scale is None:
            return v
        return v / self.delta_scale.reshape(1, -1)[:, : v.shape[-1]]

    def uncertainty_usad(self, obs_dict, action) -> float:
        """Return u_USAD, on scaled deltas when scale_aware_uncertainty is set."""
        mus = self._predict_all(obs_dict, action)
        if not self.scale_aware_uncertainty or self.delta_scale is None:
            return super().uncertainty_usad(obs_dict, action)
        m = mus / self.delta_scale.reshape(1, -1)
        d = m[:, None, :] - m[None, :, :]
        return float(np.sqrt((d ** 2).sum(-1)).max())

    def uncertainty_epistemic(self, obs_dict, action) -> float:
        """Return u_VAR, on scaled deltas when scale_aware_uncertainty is set."""
        if not self.scale_aware_uncertainty or self.delta_scale is None:
            return super().uncertainty_epistemic(obs_dict, action)
        mus = self._predict_all(obs_dict, action) / self.delta_scale.reshape(1, -1)
        return float(mus.var(axis=0).mean())

    def calibration_error(self, obs_flat: np.ndarray, action: np.ndarray,
                          next_obs_flat: np.ndarray) -> float:
        """Return ||mu_bar(s, a) - s'||_2, scaled like the uncertainty scores."""
        pred = self.predict_next_obs_vec_mean({"observation": obs_flat}, action)
        r = np.asarray(pred, np.float32) - np.asarray(next_obs_flat, np.float32)
        if self.scale_aware_uncertainty and self.delta_scale is not None:
            r = r / self.delta_scale.reshape(-1)
        return float(np.linalg.norm(r))

    def _fold_target_scaling(self, mean: np.ndarray, std: np.ndarray) -> None:
        """Fold delta = std * out + mean into the final linear layer (exact)."""
        mu = torch.as_tensor(mean.reshape(-1), dtype=torch.float32, device=self.device)
        sd = torch.as_tensor(std.reshape(-1), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            for m in self.ensemble:
                last = [layer for layer in m.net if isinstance(layer, nn.Linear)][-1]
                last.weight.mul_(sd.unsqueeze(1))
                last.bias.mul_(sd).add_(mu)

    def fit_from_transitions(
        self,
        obs_dicts: Dict[str, np.ndarray],
        actions: np.ndarray,
        next_obs_dicts: Dict[str, np.ndarray],
        *,
        verbose: bool = True,
    ) -> None:
        """Train the ensemble with MSE on (optionally standardized) deltas.

        Same inputs as EnsembleDynamicsSB3.fit_from_transitions. Records delta_scale and,
        when normalize_targets is set, folds the target scaling into the output layer.
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
            self.input_std = np.maximum(X.std(axis=0, keepdims=True), 1e-8).astype(np.float32)
            X = (X - self.input_mean) / self.input_std

        # Recorded whether or not the target is standardized.
        self.delta_scale = np.maximum(y.std(axis=0), 1e-8).astype(np.float32)

        if self.normalize_targets:
            self.target_mean = y.mean(axis=0, keepdims=True).astype(np.float32)
            self.target_std = np.maximum(y.std(axis=0, keepdims=True), 1e-8).astype(np.float32)
            y_fit = (y - self.target_mean) / self.target_std
            if verbose:
                r = float(self.target_std.max() / self.target_std.min())
                print(f"[EnsembleDynamicsV2] target standardization on "
                      f"{y.shape[1]} dims, scale spread {r:.0f}x", flush=True)
        else:
            y_fit = y

        N = X.shape[0]
        bs = int(self.cfg.batch_size)
        X_t = _to_torch(X, self.device)
        y_t = _to_torch(y_fit, self.device)

        loss_fn = nn.MSELoss()
        rng_boot = np.random.default_rng(self.cfg.seed)

        for epoch in range(int(self.cfg.epochs)):
            for ki, (m, opt) in enumerate(zip(self.ensemble, self.optims)):
                if self.cfg.bootstrap:
                    boot_idx = rng_boot.integers(0, N, size=N)
                    Xk, yk = X_t[boot_idx], y_t[boot_idx]
                    perm = torch.randperm(N, device=self.device)
                else:
                    if ki == 0:
                        perm = torch.randperm(N, device=self.device)
                    Xk, yk = X_t, y_t

                for start in range(0, N, bs):
                    idx = perm[start: start + bs]
                    opt.zero_grad(set_to_none=True)
                    loss_fn(m(Xk[idx]), yk[idx]).backward()
                    opt.step()

            if verbose and (epoch % max(1, self.cfg.epochs // 10) == 0
                            or epoch == self.cfg.epochs - 1):
                with torch.no_grad():
                    sub = min(N, 1024)
                    l0 = loss_fn(self.ensemble[0](X_t[:sub]), y_t[:sub]).item()
                print(f"[EnsembleDynamicsV2] epoch={epoch:03d}/{self.cfg.epochs}"
                      f"  loss0={l0:.6f}", flush=True)

        # Fold de-normalization into the output layer so the model predicts raw deltas.
        if self.normalize_targets:
            self._fold_target_scaling(self.target_mean, self.target_std)
            if verbose:
                print("[EnsembleDynamicsV2] target scaling folded into the output "
                      "layer; the model now predicts raw deltas", flush=True)
