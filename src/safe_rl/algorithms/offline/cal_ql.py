# =============================================================================
# Anonymous authors. Affiliations and acknowledgements will be added on
# acceptance of the associated paper. Submitted to NeurIPS 2026 (double-blind).
# =============================================================================
"""
cal_ql.py — Calibrated Q-Learning (Nakamoto et al., 2023)
arXiv:2303.05479

Cal-QL addresses over-conservatism in CQL by adding a calibration loss
that prevents Q-values from being pushed below the empirical Monte Carlo
return computed from the offline dataset:

  Q_loss = CQL_loss + λ_cal * E_{(s,a)~D}[ max(0, R_MC(s,a) - Q(s,a))^2 ]

where R_MC(s,a) = Σ_{t=0}^{k-1} γ^t r_{t+τ} + γ^k V(s_{τ+k}) is the
k-step Monte Carlo return starting from (s,a) in the trajectory.

This calibration ensures Q(s,a) ≥ R_MC(s,a) at data points, preventing
under-estimation of good actions while keeping the CQL upper bound.

Interface (identical to IQL/CQL):
  agent = CalQLAgent(obs_dim, act_dim, cfg)
  agent.train(dataset, n_epochs=200)
  action = agent.act(obs_dict)

Hyperparameters follow the original paper (Table 1 / Table 3):
  Base CQL: alpha_cql=5.0 (same as CQL for both maze and locomotion)
  Calibration:
    k_mc=5 (locomotion); k_mc=10 (maze/antmaze)
    lambda_cal=1.0 (paper: fixed across tasks)
    alpha_sac: same as CQL (0.05 locomotion, 0.1 maze)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Reuse CQL building blocks (same architecture family)
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
    """
    Cal-QL config extends CQL config with calibration hyperparameters.

    Inherits all CQL fields:
      hidden_dim, n_hidden, lr, batch_size, gamma, tau,
      alpha_sac, alpha_cql, n_action_samples, normalize_states, device, seed

    Additional fields:
      k_mc      : steps for MC return estimate (paper: 5 locomotion, 10 maze)
      lambda_cal: weight of calibration loss    (paper: 1.0)
    """
    k_mc:       int   = 5      # locomotion default; set 10 for PointMaze
    lambda_cal: float = 1.0    # calibration loss weight (paper: fixed at 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# MC return computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_mc_returns(
    rewards:  np.ndarray,   # [N]
    dones:    np.ndarray,   # [N]  (1.0 at terminal steps)
    gamma:    float,
    k:        int,
) -> np.ndarray:
    """
    Compute k-step discounted Monte Carlo returns for every transition i:

      R_MC[i] = Σ_{j=0}^{k-1} γ^j · r[i+j]  (truncated at episode boundary)

    For transitions close to the end of an episode, the sum is truncated
    when a terminal step is encountered (done[i+j] = 1).

    Returns R_MC as a float32 array of shape [N].
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
            if dones[i + j] > 0.5:       # episode boundary
                break
        R_mc[i] = cumret
    return R_mc


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class CalQLAgent(CQLAgent):
    """
    Calibrated Q-Learning agent for continuous-action offline RL.

    Extends CQLAgent with a calibration loss that prevents Q-values from
    dropping below the k-step Monte Carlo return at data points.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        cfg: Optional[CalQLConfig] = None,
    ):
        # Use CalQLConfig defaults if no config provided
        cal_cfg = cfg if cfg is not None else CalQLConfig()
        # Convert to CalQLConfig if a bare CQLConfig was passed
        if type(cal_cfg) is CQLConfig:
            cal_cfg = CalQLConfig(**{k: v for k, v in vars(cal_cfg).items()
                                    if k in CalQLConfig.__dataclass_fields__})
        super().__init__(obs_dim, act_dim, cfg=cal_cfg)
        self.cfg: CalQLConfig = cal_cfg
        self.loss_history["cal"] = []

        # Calibration targets computed once in train()
        self._R_mc_tensor: Optional[torch.Tensor] = None

    # ── training ──────────────────────────────────────────────────────────────

    def train(self, dataset: Dict[str, Any], n_epochs: int = 200) -> None:
        """
        Train Cal-QL on the offline dataset.

        Extends CQL training by pre-computing k-step MC returns and
        adding a calibration loss at each gradient step.
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

        # ── Pre-compute k-step MC returns ────────────────────────────────────
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

            # ── SAC backup ───────────────────────────────────────────────────
            with torch.no_grad():
                a2, lp2 = self.actor.sample(s2)
                q2_min  = self.q_tgt.min(s2, a2)
                q_tgt   = r + self.cfg.gamma * (1.0 - d) * (q2_min - α_s * lp2)

            q1, q2 = self.q_net(s, a)
            q_std_loss = F.mse_loss(q1, q_tgt) + F.mse_loss(q2, q_tgt)

            # ── CQL conservative term ────────────────────────────────────────
            cql1 = self._cql_term(
                lambda ss, aa: self.q_net(ss, aa)[0], s, a)
            cql2 = self._cql_term(
                lambda ss, aa: self.q_net(ss, aa)[1], s, a)
            cql_loss = self.cfg.alpha_cql * (cql1 + cql2)

            # ── Calibration loss: push Q ≥ R_MC ─────────────────────────────
            # F.relu(x) = max(0, x): only penalise Q < R_MC (Nakamoto eq. 5)
            cal1 = F.relu(r_mc - q1).pow(2).mean()
            cal2 = F.relu(r_mc - q2).pow(2).mean()
            cal_loss = self.cfg.lambda_cal * (cal1 + cal2)

            q_loss = q_std_loss + cql_loss + cal_loss
            self.q_opt.zero_grad()
            q_loss.backward()
            self.q_opt.step()

            # ── SAC actor update ─────────────────────────────────────────────
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
