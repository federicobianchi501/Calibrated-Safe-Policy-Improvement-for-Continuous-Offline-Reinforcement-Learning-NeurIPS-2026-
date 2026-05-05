# =============================================================================
# Anonymous authors. Affiliations and acknowledgements will be added on
# acceptance of the associated paper. Submitted to NeurIPS 2026 (double-blind).
# =============================================================================
"""
cql.py — Conservative Q-Learning (Kumar et al., NeurIPS 2020)
arXiv:2006.04779

CQL adds a conservative regularization to the standard SAC objective:
  Q_loss += α_cql * (logsumexp_{a~π,a~unif} Q(s,a) - Q(s, a_data))

This prevents Q-overestimation for OOD actions while using the dataset to
ground Q-values for in-distribution (s,a) pairs.

Interface (identical to IQL/TD3+BC):
  agent = CQLAgent(obs_dim, act_dim, cfg)
  agent.train(dataset, n_epochs=200)
  action = agent.act(obs_dict)

Hyperparameters follow the original paper:
  AntMaze / PointMaze (Table 3): alpha_cql=5.0, alpha_sac=0.1
  Locomotion — Hopper, Walker2d (Table 1): alpha_cql=5.0, alpha_sac=0.05
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

LOG_STD_MIN = -4.0
LOG_STD_MAX = 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CQLConfig:
    # Architecture (paper: 3 hidden layers, 256 units for all domains)
    hidden_dim: int = 256
    n_hidden:   int = 3

    # Training
    lr:         float = 3e-4
    batch_size: int   = 256
    gamma:      float = 0.99
    tau:        float = 0.005       # soft target-update coefficient

    # SAC temperature (fixed; auto-tuning not needed for offline RL)
    # AntMaze/PointMaze: 0.1 (Table 3); Locomotion: 0.05 (Table 1)
    alpha_sac:  float = 0.1

    # CQL conservative coefficient
    # Paper: α=5.0 for AntMaze and locomotion (Tables 1 & 3)
    alpha_cql:  float = 5.0

    # Number of action samples for logsumexp estimation (paper: 10)
    n_action_samples: int = 10

    # State normalization (paper: normalize_states=True for all domains)
    normalize_states: bool = True

    device: str = "cpu"
    seed:   int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Networks
# ─────────────────────────────────────────────────────────────────────────────

def _mlp(in_dim: int, out_dim: int, hidden_dim: int, n_hidden: int) -> nn.Sequential:
    layers: list = []
    prev = in_dim
    for _ in range(n_hidden):
        layers += [nn.Linear(prev, hidden_dim), nn.ReLU()]
        prev = hidden_dim
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class DoubleQNetwork(nn.Module):
    """Two Q-heads for double-Q learning."""
    def __init__(self, state_dim: int, act_dim: int, hidden_dim: int, n_hidden: int):
        super().__init__()
        self.q1 = _mlp(state_dim + act_dim, 1, hidden_dim, n_hidden)
        self.q2 = _mlp(state_dim + act_dim, 1, hidden_dim, n_hidden)

    def forward(self, s: torch.Tensor, a: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)

    def min(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(s, a)
        return torch.min(q1, q2)


class GaussianActor(nn.Module):
    """
    Stochastic Gaussian actor with tanh squashing (SAC-style).

    Outputs mean and log_std from shared trunk; log_prob accounts for the
    tanh change-of-variables (Appendix C of Haarnoja et al. 2018).
    """
    def __init__(self, state_dim: int, act_dim: int,
                 hidden_dim: int, n_hidden: int):
        super().__init__()
        self.trunk   = _mlp(state_dim, hidden_dim, hidden_dim, n_hidden - 1)
        self.mean_fc = nn.Linear(hidden_dim, act_dim)
        self.lstd_fc = nn.Linear(hidden_dim, act_dim)

    def _dist(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h      = self.trunk(s)
        mean   = self.mean_fc(h)
        log_std = self.lstd_fc(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std.exp()

    def sample(self, s: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (tanh-squashed action, log_prob)."""
        mean, std = self._dist(s)
        eps = torch.randn_like(mean)
        x   = mean + std * eps
        a   = torch.tanh(x)
        # log_prob with tanh Jacobian correction (sum over action dims)
        log_pi = (Normal(mean, std).log_prob(x)
                  - torch.log(1.0 - a.pow(2) + 1e-6)).sum(-1)
        return a, log_pi

    def sample_n(self, s: torch.Tensor, n: int
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample n actions for each state in s.
        Returns actions [B, n, act_dim] and log_probs [B, n].
        """
        B = s.shape[0]
        s_rep = s.unsqueeze(1).expand(-1, n, -1).reshape(B * n, -1)
        a, lp = self.sample(s_rep)
        return a.reshape(B, n, -1), lp.reshape(B, n)

    @torch.no_grad()
    def mean_action(self, s: torch.Tensor) -> torch.Tensor:
        """Deterministic mean action for evaluation."""
        mean, _ = self._dist(s)
        return torch.tanh(mean)


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class CQLAgent:
    """
    Conservative Q-Learning agent for continuous-action offline RL.

    Observation interface: dict with keys 'observation' and 'desired_goal'
    (PointMaze goal-conditioned format). Uses s = concat(obs, goal).
    For flat-obs domains (Hopper, Walker2d), obs_desired_goal has dim 0.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        cfg: Optional[CQLConfig] = None,
    ):
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.cfg     = cfg if cfg is not None else CQLConfig()
        self.device  = torch.device(self.cfg.device)

        torch.manual_seed(self.cfg.seed)
        self._build_networks(obs_dim)

        self.state_mean: Optional[torch.Tensor] = None
        self.state_std:  Optional[torch.Tensor] = None
        self.loss_history: Dict[str, list] = {"q": [], "actor": []}

    def _build_networks(self, state_dim: int) -> None:
        self.obs_dim = state_dim
        sd, ad = state_dim, self.act_dim
        hd, nh = self.cfg.hidden_dim, self.cfg.n_hidden

        self.q_net  = DoubleQNetwork(sd, ad, hd, nh).to(self.device)
        self.q_tgt  = DoubleQNetwork(sd, ad, hd, nh).to(self.device)
        self.actor  = GaussianActor(sd, ad, hd, nh).to(self.device)

        self.q_tgt.load_state_dict(self.q_net.state_dict())

        self.q_opt = optim.Adam(self.q_net.parameters(), lr=self.cfg.lr)
        self.a_opt = optim.Adam(self.actor.parameters(), lr=self.cfg.lr)

    # ── observation processing ────────────────────────────────────────────────

    def _obs_to_state(self, obs: Any) -> np.ndarray:
        """Dict or flat array → flat numpy state."""
        if isinstance(obs, dict):
            o = np.asarray(obs["observation"],   dtype=np.float32).reshape(-1)
            g = np.asarray(obs.get("desired_goal", np.zeros(0)),
                           dtype=np.float32).reshape(-1)
            return np.concatenate([o, g])
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    def _dataset_to_tensors(
        self, dataset: Dict[str, Any]
    ) -> Tuple[torch.Tensor, ...]:
        def _t(x): return torch.as_tensor(
            np.asarray(x, dtype=np.float32), device=self.device)

        obs = np.concatenate([
            dataset["obs_observation"],
            dataset.get("obs_desired_goal",
                        np.zeros((len(dataset["obs_observation"]), 0),
                                 dtype=np.float32))
        ], axis=-1)
        next_obs = np.concatenate([
            dataset["next_observation"],
            dataset.get("next_desired_goal",
                        np.zeros((len(dataset["next_observation"]), 0),
                                 dtype=np.float32))
        ], axis=-1)
        return (
            _t(obs),
            _t(dataset["act"]),
            _t(dataset["rew"]),
            _t(next_obs),
            _t(dataset["done"]),
        )

    # ── CQL conservative Q loss ───────────────────────────────────────────────

    def _cql_term(
        self,
        q_net_fn,          # callable: (s [B*n, sd], a [B*n, ad]) -> q [B*n]
        s: torch.Tensor,   # [B, sd]
        a_data: torch.Tensor,  # [B, ad]
    ) -> torch.Tensor:
        """
        CQL-H conservative term (Kumar 2020 eq. 4):
          logsumexp_{a~π ∪ a~unif}(Q(s,a)) - Q(s, a_data)

        Uses importance-sampling correction for the policy distribution and
        uniform distribution (no correction needed for uniform over [-1,1]^d).
        """
        B  = s.shape[0]
        n  = self.cfg.n_action_samples
        ad = self.act_dim

        # Policy samples
        with torch.no_grad():
            a_pi, log_pi = self.actor.sample_n(s, n)   # [B, n, ad], [B, n]
        s_rep = s.unsqueeze(1).expand(-1, n, -1).reshape(B * n, -1)
        q_pi  = q_net_fn(s_rep, a_pi.reshape(B * n, ad)).reshape(B, n)
        # IS correction: subtract log π(a|s) so that logsumexp gives
        # ∫ Q(s,a) da ≈ logsumexp(Q - log π) + const  (Kumar 2020 App. F)
        q_pi_is = q_pi - log_pi.detach()

        # Uniform random samples (log prob = -act_dim * log 2 = const → omit)
        a_rand = torch.FloatTensor(B, n, ad).uniform_(-1.0, 1.0).to(self.device)
        q_rand = q_net_fn(s_rep, a_rand.reshape(B * n, ad)).reshape(B, n)

        # logsumexp over both sets (concat along sample dim)
        all_q  = torch.cat([q_pi_is, q_rand], dim=1)            # [B, 2n]
        lse    = torch.logsumexp(all_q, dim=1) - np.log(2 * n)  # [B]

        # In-distribution Q value
        q_data = q_net_fn(s, a_data)                             # [B]

        return (lse - q_data).mean()

    # ── training ──────────────────────────────────────────────────────────────

    def _soft_update(self) -> None:
        for p, pt in zip(self.q_net.parameters(), self.q_tgt.parameters()):
            pt.data.copy_(self.cfg.tau * p.data + (1.0 - self.cfg.tau) * pt.data)

    def train(self, dataset: Dict[str, Any], n_epochs: int = 200) -> None:
        """Train CQL on the offline dataset."""
        S, A, R, S2, D = self._dataset_to_tensors(dataset)
        N = S.shape[0]
        if N == 0:
            print("[CQL] empty dataset — skipping.", flush=True)
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

        rng = np.random.default_rng(self.cfg.seed)

        for epoch in range(int(n_epochs)):
            idx = rng.choice(N, size=bs, replace=(N < bs))
            s, a, r, s2, d = S[idx], A[idx], R[idx], S2[idx], D[idx]

            α_s = self.cfg.alpha_sac

            # ── SAC backup ───────────────────────────────────────────────────
            with torch.no_grad():
                a2, lp2 = self.actor.sample(s2)
                q2_min  = self.q_tgt.min(s2, a2)
                q_tgt   = r + self.cfg.gamma * (1.0 - d) * (q2_min - α_s * lp2)

            q1, q2 = self.q_net(s, a)
            q_std_loss = F.mse_loss(q1, q_tgt) + F.mse_loss(q2, q_tgt)

            # ── CQL conservative term (applied to both Q heads) ─────────────
            cql1 = self._cql_term(
                lambda ss, aa: self.q_net(ss, aa)[0], s, a)
            cql2 = self._cql_term(
                lambda ss, aa: self.q_net(ss, aa)[1], s, a)
            cql_loss = self.cfg.alpha_cql * (cql1 + cql2)

            q_loss = q_std_loss + cql_loss
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

    # ── inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def act(self, obs: Any) -> np.ndarray:
        """Return deterministic mean action ∈ [-1,1]^act_dim."""
        s = torch.as_tensor(self._obs_to_state(obs),
                            dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.state_mean is not None:
            s = (s - self.state_mean) / self.state_std
        a = self.actor.mean_action(s).squeeze(0).cpu().numpy()
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def __call__(self, obs: Any) -> np.ndarray:
        return self.act(obs)
