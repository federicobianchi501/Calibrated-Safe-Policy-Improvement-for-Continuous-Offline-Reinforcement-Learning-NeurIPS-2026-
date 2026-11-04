# =============================================================================
# Calibrated Safe Policy Improvement for Continuous Offline Reinforcement Learning
# Federico Bianchi, Alessandro Farinelli, Alberto Castellini
# NeurIPS 2026
# =============================================================================
"""Calibrated Q-Learning (Nakamoto et al., NeurIPS 2023, arXiv:2303.05479).

CQL plus a one-sided penalty that keeps Q above a k-step discounted return
R_MC(s, a) computed from the dataset:
    Q_loss = CQL_loss + lambda_cal * E_D[max(0, R_MC(s, a) - Q(s, a))^2],
    R_MC[i] = sum_{j<k} gamma^j r[i+j], truncated at the first terminal (no bootstrap).

    agent = CalQLAgent(obs_dim, act_dim, cfg)
    agent.train(dataset, n_epochs=200)
    action = agent.act(obs)

CQL hyperparameters are inherited from CQLConfig; k_mc=5 (locomotion) or 10 (maze).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from safe_rl.algorithms.offline.cql import (
    CQLAgent,
    CQLConfig,
    DoubleQNetwork,
    GaussianActor,
    _mlp,
)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CalQLConfig(CQLConfig):
    """CQLConfig plus the calibration horizon k_mc and loss weight lambda_cal."""
    k_mc:       int   = 5      # 10 for maze domains
    lambda_cal: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# MC return computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_mc_returns(
    rewards:  np.ndarray,   # [N]
    dones:    np.ndarray,   # [N], 1.0 at terminal steps
    gamma:    float,
    k:        int,
) -> np.ndarray:
    """k-step discounted returns R_MC[i] = sum_{j<k} gamma^j r[i+j], float32 [N].

    Transitions are assumed to be stored in trajectory order. The sum stops
    after the first step with done = 1 or at the end of the array.
    """
    N = len(rewards)
    R_mc = np.zeros(N, dtype=np.float32)
    for i in range(N):
        disc   = 1.0
        cumret = 0.0
        for j in range(k):
            if i + j >= N:
                break
            cumret += disc * rewards[i + j]
            disc   *= gamma
            if dones[i + j] > 0.5:
                break
        R_mc[i] = cumret
    return R_mc


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class CalQLAgent(CQLAgent):
    """CQLAgent with a penalty on Q falling below R_MC at dataset (s, a)."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        cfg: Optional[CalQLConfig] = None,
    ):
        cal_cfg = cfg if cfg is not None else CalQLConfig()
        # A plain CQLConfig is upgraded to CalQLConfig with default calibration fields.
        if type(cal_cfg) is CQLConfig:
            cal_cfg = CalQLConfig(**{k: v for k, v in vars(cal_cfg).items()
                                    if k in CalQLConfig.__dataclass_fields__})
        super().__init__(obs_dim, act_dim, cfg=cal_cfg)
        self.cfg: CalQLConfig = cal_cfg
        self.loss_history["cal"] = []

        self._R_mc_tensor: Optional[torch.Tensor] = None

    # ── training ──────────────────────────────────────────────────────────────

    def train(self, dataset: Dict[str, Any], n_epochs: int = 200) -> None:
        """Train on an offline transition dict; each epoch is one gradient step.

        R_MC is computed once from the unshuffled dataset before training.
        """
        S, A, R, S2, D = self._dataset_to_tensors(dataset)
        N = S.shape[0]
        if N == 0:
            print("[Cal-QL] empty dataset — skipping.", flush=True)
            return
        bs = min(self.cfg.batch_size, N)

        state_dim = int(S.shape[1])
        if state_dim != self.obs_dim:
            self._build_networks(state_dim)
            S, A, R, S2, D = self._dataset_to_tensors(dataset)

        if self.cfg.normalize_states:
            self.state_mean = S.mean(0, keepdim=True)
            self.state_std  = S.std(0, keepdim=True).clamp(min=1e-3)
            S  = (S  - self.state_mean) / self.state_std
            S2 = (S2 - self.state_mean) / self.state_std

        r_np = R.cpu().numpy()
        d_np = D.cpu().numpy()
        R_mc_np = _compute_mc_returns(r_np, d_np, self.cfg.gamma, self.cfg.k_mc)
        R_mc = torch.as_tensor(R_mc_np, dtype=torch.float32, device=self.device)

        rng = np.random.default_rng(self.cfg.seed)
        α_s = self.cfg.alpha_sac

        for epoch in range(int(n_epochs)):
            idx = rng.choice(N, size=bs, replace=(N < bs))
            s, a, r, s2, d = S[idx], A[idx], R[idx], S2[idx], D[idx]
            r_mc = R_mc[idx]

            # Soft Bellman target.
            with torch.no_grad():
                a2, lp2 = self.actor.sample(s2)
                q2_min  = self.q_tgt.min(s2, a2)
                q_tgt   = r + self.cfg.gamma * (1.0 - d) * (q2_min - α_s * lp2)

            q1, q2 = self.q_net(s, a)
            q_std_loss = F.mse_loss(q1, q_tgt) + F.mse_loss(q2, q_tgt)

            # Conservative term, applied to each Q head.
            cql1 = self._cql_term(
                lambda ss, aa: self.q_net(ss, aa)[0], s, a)
            cql2 = self._cql_term(
                lambda ss, aa: self.q_net(ss, aa)[1], s, a)
            cql_loss = self.cfg.alpha_cql * (cql1 + cql2)

            # Calibration: penalize only Q < R_MC.
            cal1 = F.relu(r_mc - q1).pow(2).mean()
            cal2 = F.relu(r_mc - q2).pow(2).mean()
            cal_loss = self.cfg.lambda_cal * (cal1 + cal2)

            q_loss = q_std_loss + cql_loss + cal_loss
            self.q_opt.zero_grad()
            q_loss.backward()
            self.q_opt.step()

            # SAC actor update.
            a_new, lp_new = self.actor.sample(s)
            q_new_min     = self.q_net.min(s, a_new)
            a_loss        = (α_s * lp_new - q_new_min).mean()
            self.a_opt.zero_grad()
            a_loss.backward()
            self.a_opt.step()

            self._soft_update()

            self.loss_history["q"].append(float(q_loss))
            self.loss_history["actor"].append(float(a_loss))
            self.loss_history["cal"].append(float(cal_loss))
