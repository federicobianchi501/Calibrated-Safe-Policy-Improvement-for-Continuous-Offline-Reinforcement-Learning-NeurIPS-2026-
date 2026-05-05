# =============================================================================
# Anonymous authors. Affiliations and acknowledgements will be added on
# acceptance of the associated paper. Submitted to NeurIPS 2026 (double-blind).
# =============================================================================
"""
iql.py — Implicit Q-Learning (Kostrikov et al., 2021)
arXiv:2110.06169

Offline RL algorithm using:
  - Expectile regression for V-network
  - Advantage-weighted regression for actor
  - Standard TD backup for Q-network

Interface:
  agent = IQLAgent(obs_dim, act_dim, cfg)
  agent.train(dataset, n_epochs=100)
  action = agent.act(obs_dict)          # obs_dict: {"observation":..., "desired_goal":...}

Goal-conditioned: state = concat(observation, desired_goal).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IQLConfig:
    # Architecture (paper Table 5: 3 hidden layers for locomotion)
    hidden_dim: int   = 256
    n_hidden:   int   = 3

    # Training
    lr:         float = 3e-4
    batch_size: int   = 256
    gamma:      float = 0.99
    tau:        float = 0.005   # soft target update coefficient

    # IQL-specific
    beta:       float = 3.0     # inverse temperature for advantage weighting
    alpha:      float = 0.7     # expectile level for V (0.5 = MSE, 1 = max)
    clip_score: float = 100.0   # clip exp(beta*A) to avoid numerical issues

    # State normalization (paper Table 5: normalize_states=True for locomotion)
    normalize_states: bool  = True

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

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)

    def min(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(s, a)
        return torch.min(q1, q2)


class VNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, n_hidden: int):
        super().__init__()
        self.net = _mlp(state_dim, 1, hidden_dim, n_hidden)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s).squeeze(-1)


class DeterministicActor(nn.Module):
    """Deterministic actor with tanh squashing to [-1,1]."""
    def __init__(self, state_dim: int, act_dim: int, hidden_dim: int, n_hidden: int):
        super().__init__()
        self.net = _mlp(state_dim, act_dim, hidden_dim, n_hidden)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(s))


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class IQLAgent:
    """
    Implicit Q-Learning agent for continuous-action offline RL.

    Observation interface: dict with keys "observation" and "desired_goal"
    (PointMaze goal-conditioned format). Uses s = concat(obs, goal).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        cfg: Optional[IQLConfig] = None,
    ):
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.cfg = cfg if cfg is not None else IQLConfig()
        self.device = torch.device(self.cfg.device)

        torch.manual_seed(self.cfg.seed)
        state_dim = obs_dim  # after _obs_to_state (flat obs + goal concatenated)

        self.q_net  = DoubleQNetwork(state_dim, act_dim,
                                     self.cfg.hidden_dim, self.cfg.n_hidden).to(self.device)
        self.q_tgt  = DoubleQNetwork(state_dim, act_dim,
                                     self.cfg.hidden_dim, self.cfg.n_hidden).to(self.device)
        self.v_net  = VNetwork(state_dim, self.cfg.hidden_dim, self.cfg.n_hidden).to(self.device)
        self.actor  = DeterministicActor(state_dim, act_dim,
                                         self.cfg.hidden_dim, self.cfg.n_hidden).to(self.device)

        # hard copy Q -> Q_target
        self.q_tgt.load_state_dict(self.q_net.state_dict())

        self.q_opt  = optim.Adam(self.q_net.parameters(),  lr=self.cfg.lr)
        self.v_opt  = optim.Adam(self.v_net.parameters(),  lr=self.cfg.lr)
        self.a_opt  = optim.Adam(self.actor.parameters(),  lr=self.cfg.lr)

        # training history
        self.loss_history: Dict[str, list] = {"q": [], "v": [], "actor": []}

        # state normalization stats (set during train())
        self.state_mean: Optional[torch.Tensor] = None
        self.state_std:  Optional[torch.Tensor] = None

    # ── observation processing ────────────────────────────────────────────────
    def _obs_to_state(self, obs: Any) -> np.ndarray:
        """
        Convert observation (dict or flat array) to flat numpy state vector.
        For PointMaze: concat(observation, desired_goal).
        """
        if isinstance(obs, dict):
            o = np.asarray(obs["observation"], dtype=np.float32).reshape(-1)
            g = np.asarray(obs.get("desired_goal", np.zeros(2)), dtype=np.float32).reshape(-1)
            return np.concatenate([o, g])
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    def _dataset_to_tensors(
        self, dataset: Dict[str, Any]
    ) -> Tuple[torch.Tensor, ...]:
        """
        Parse dataset dict (as produced by transitions_*.npz) to torch tensors.

        Expected keys (PointMaze):
          obs_observation, obs_desired_goal, act, rew, done,
          next_observation, next_desired_goal
        """
        def _t(x): return torch.as_tensor(np.asarray(x, dtype=np.float32),
                                           device=self.device)

        obs  = np.concatenate([dataset["obs_observation"],
                               dataset.get("obs_desired_goal",
                                           np.zeros((len(dataset["obs_observation"]), 2),
                                                    dtype=np.float32))], axis=-1)
        next_obs = np.concatenate([dataset["next_observation"],
                                   dataset.get("next_desired_goal",
                                               np.zeros((len(dataset["next_observation"]), 2),
                                                        dtype=np.float32))], axis=-1)
        return (
            _t(obs),                          # [N, state_dim]
            _t(dataset["act"]),               # [N, act_dim]
            _t(dataset["rew"]),               # [N]
            _t(next_obs),                     # [N, state_dim]
            _t(dataset["done"]),              # [N]
        )

    # ── training ──────────────────────────────────────────────────────────────
    @staticmethod
    def _expectile_loss(diff: torch.Tensor, alpha: float) -> torch.Tensor:
        """Asymmetric L2 (expectile regression): weight alpha above 0, (1-alpha) below."""
        return torch.where(diff >= 0, alpha * diff.pow(2),
                           (1.0 - alpha) * diff.pow(2)).mean()

    def _soft_update(self) -> None:
        tau = self.cfg.tau
        for p, pt in zip(self.q_net.parameters(), self.q_tgt.parameters()):
            pt.data.copy_(tau * p.data + (1.0 - tau) * pt.data)

    def train(self, dataset: Dict[str, Any], n_epochs: int = 100) -> None:
        """
        Train IQL on the offline dataset.

        dataset: dict with keys as in transitions_*.npz
        """
        S, A, R, S2, D = self._dataset_to_tensors(dataset)
        N = S.shape[0]
        if N == 0:
            print("[IQL] D_train is empty — skipping training (policy stays at random init).", flush=True)
            return
        bs = min(self.cfg.batch_size, N)

        # Update obs_dim to actual state_dim (obs + goal)
        state_dim = int(S.shape[1])
        if state_dim != self.obs_dim:
            self._rebuild_networks(state_dim)
            S, A, R, S2, D = self._dataset_to_tensors(dataset)

        # State normalization (Kostrikov 2021, Table 5: normalize_states=True)
        if self.cfg.normalize_states:
            self.state_mean = S.mean(0, keepdim=True)
            self.state_std  = S.std(0, keepdim=True).clamp(min=1e-3)
            S  = (S  - self.state_mean) / self.state_std
            S2 = (S2 - self.state_mean) / self.state_std

        rng = np.random.default_rng(self.cfg.seed)

        for epoch in range(int(n_epochs)):
            idx = rng.choice(N, size=bs, replace=(N < bs))
            s  = S[idx]
            a  = A[idx]
            r  = R[idx]
            s2 = S2[idx]
            d  = D[idx]

            with torch.no_grad():
                v2 = self.v_net(s2)
                q_target = r + self.cfg.gamma * (1.0 - d) * v2

            # ── Q update ──
            q1, q2 = self.q_net(s, a)
            q_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
            self.q_opt.zero_grad()
            q_loss.backward()
            self.q_opt.step()

            # ── V update (expectile regression) ──
            with torch.no_grad():
                q_min = self.q_tgt.min(s, a)
            v     = self.v_net(s)
            diff  = q_min - v
            v_loss = self._expectile_loss(diff, self.cfg.alpha)
            self.v_opt.zero_grad()
            v_loss.backward()
            self.v_opt.step()

            # ── Actor update (advantage-weighted regression) ──
            # w = exp(β·A).clip(max=C), no mean-normalization (Kostrikov 2021 Algorithm 1)
            with torch.no_grad():
                adv  = self.q_tgt.min(s, a) - self.v_net(s)
                w    = torch.exp(self.cfg.beta * adv).clamp(max=self.cfg.clip_score)
            a_pred  = self.actor(s)
            a_loss  = (w * F.mse_loss(a_pred, a, reduction="none").mean(-1)).mean()
            self.a_opt.zero_grad()
            a_loss.backward()
            self.a_opt.step()

            self._soft_update()

            self.loss_history["q"].append(float(q_loss))
            self.loss_history["v"].append(float(v_loss))
            self.loss_history["actor"].append(float(a_loss))

    def _rebuild_networks(self, state_dim: int) -> None:
        """Re-initialise networks with the correct state_dim."""
        self.obs_dim = state_dim
        self.q_net  = DoubleQNetwork(state_dim, self.act_dim,
                                     self.cfg.hidden_dim, self.cfg.n_hidden).to(self.device)
        self.q_tgt  = DoubleQNetwork(state_dim, self.act_dim,
                                     self.cfg.hidden_dim, self.cfg.n_hidden).to(self.device)
        self.q_tgt.load_state_dict(self.q_net.state_dict())
        self.v_net  = VNetwork(state_dim, self.cfg.hidden_dim, self.cfg.n_hidden).to(self.device)
        self.actor  = DeterministicActor(state_dim, self.act_dim,
                                         self.cfg.hidden_dim, self.cfg.n_hidden).to(self.device)
        self.q_opt  = optim.Adam(self.q_net.parameters(),  lr=self.cfg.lr)
        self.v_opt  = optim.Adam(self.v_net.parameters(),  lr=self.cfg.lr)
        self.a_opt  = optim.Adam(self.actor.parameters(),  lr=self.cfg.lr)

    # ── inference ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def act(self, obs: Any) -> np.ndarray:
        """Return action ∈ [-1,1]^act_dim from observation dict or array."""
        s = torch.as_tensor(self._obs_to_state(obs), dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        if self.state_mean is not None:
            s = (s - self.state_mean) / self.state_std
        a = self.actor(s).squeeze(0).cpu().numpy()
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def __call__(self, obs: Any) -> np.ndarray:
        return self.act(obs)
