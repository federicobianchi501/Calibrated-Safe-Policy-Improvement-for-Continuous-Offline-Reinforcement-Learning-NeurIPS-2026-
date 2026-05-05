# =============================================================================
# Anonymous authors. Affiliations and acknowledgements will be added on
# acceptance of the associated paper. Submitted to NeurIPS 2026 (double-blind).
# =============================================================================
"""
td3bc.py — TD3+BC (Fujimoto & Gu, 2021)
arXiv:2106.06860

Offline RL algorithm: TD3 with a behavioral cloning (BC) regularization
term on the actor loss.

Actor loss: L_actor = -Q(s, pi(s)) + lambda_bc * ||pi(s) - a_data||^2
  where lambda_bc normalises the BC term by |Q|.

Interface:
  agent = TD3BCAgent(obs_dim, act_dim, cfg)
  agent.train(dataset, n_epochs=100)
  action = agent.act(obs_dict)
"""

from __future__ import annotations

from dataclasses import dataclass
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
class TD3BCConfig:
    hidden_dim:  int   = 256
    n_hidden:    int   = 2

    lr:          float = 3e-4
    batch_size:  int   = 256
    gamma:       float = 0.99
    tau:         float = 0.005   # soft target update

    # TD3 specific
    policy_noise:   float = 0.2   # noise added to target actions
    noise_clip:     float = 0.5   # clip target policy noise
    policy_freq:    int   = 2     # actor update every N critic updates

    # BC regularization
    alpha_bc:    float = 2.5      # BC weight; per the paper default

    # State normalization (Fujimoto & Gu 2021: explicitly required for locomotion)
    normalize_states: bool  = True

    device: str = "cpu"
    seed:   int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Networks (re-use the _mlp helper from iql, defined inline here for independence)
# ─────────────────────────────────────────────────────────────────────────────

def _mlp(in_dim: int, out_dim: int, hidden_dim: int, n_hidden: int) -> nn.Sequential:
    layers: list = []
    prev = in_dim
    for _ in range(n_hidden):
        layers += [nn.Linear(prev, hidden_dim), nn.ReLU()]
        prev = hidden_dim
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class TD3Actor(nn.Module):
    def __init__(self, state_dim: int, act_dim: int, hidden_dim: int, n_hidden: int):
        super().__init__()
        self.net = _mlp(state_dim, act_dim, hidden_dim, n_hidden)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(s))


class TD3Critic(nn.Module):
    """Double-Q critic for TD3."""
    def __init__(self, state_dim: int, act_dim: int, hidden_dim: int, n_hidden: int):
        super().__init__()
        self.q1 = _mlp(state_dim + act_dim, 1, hidden_dim, n_hidden)
        self.q2 = _mlp(state_dim + act_dim, 1, hidden_dim, n_hidden)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)

    def q1_value(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class TD3BCAgent:
    """
    TD3+BC offline RL agent.
    Works with PointMaze dict obs (goal-conditioned): s = concat(obs, goal).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        cfg: Optional[TD3BCConfig] = None,
    ):
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.cfg = cfg if cfg is not None else TD3BCConfig()
        self.device = torch.device(self.cfg.device)

        torch.manual_seed(self.cfg.seed)

        self.actor      = TD3Actor(obs_dim, act_dim,
                                   self.cfg.hidden_dim, self.cfg.n_hidden).to(self.device)
        self.actor_tgt  = TD3Actor(obs_dim, act_dim,
                                   self.cfg.hidden_dim, self.cfg.n_hidden).to(self.device)
        self.critic     = TD3Critic(obs_dim, act_dim,
                                    self.cfg.hidden_dim, self.cfg.n_hidden).to(self.device)
        self.critic_tgt = TD3Critic(obs_dim, act_dim,
                                    self.cfg.hidden_dim, self.cfg.n_hidden).to(self.device)

        self.actor_tgt.load_state_dict(self.actor.state_dict())
        self.critic_tgt.load_state_dict(self.critic.state_dict())

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=self.cfg.lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.cfg.lr)

        self._train_step = 0
        self.loss_history: Dict[str, list] = {"critic": [], "actor": []}

        # state normalization stats (set during train())
        self.state_mean: Optional[torch.Tensor] = None
        self.state_std:  Optional[torch.Tensor] = None

    # ── helpers ───────────────────────────────────────────────────────────────
    def _obs_to_state(self, obs: Any) -> np.ndarray:
        if isinstance(obs, dict):
            o = np.asarray(obs["observation"], dtype=np.float32).reshape(-1)
            g = np.asarray(obs.get("desired_goal", np.zeros(2)), dtype=np.float32).reshape(-1)
            return np.concatenate([o, g])
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    def _dataset_to_tensors(self, dataset: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
        def _t(x): return torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)
        obs = np.concatenate([
            dataset["obs_observation"],
            dataset.get("obs_desired_goal",
                        np.zeros((len(dataset["obs_observation"]), 2), dtype=np.float32)),
        ], axis=-1)
        nxt = np.concatenate([
            dataset["next_observation"],
            dataset.get("next_desired_goal",
                        np.zeros((len(dataset["next_observation"]), 2), dtype=np.float32)),
        ], axis=-1)
        return _t(obs), _t(dataset["act"]), _t(dataset["rew"]), _t(nxt), _t(dataset["done"])

    def _soft_update(self, src: nn.Module, tgt: nn.Module) -> None:
        tau = self.cfg.tau
        for p, pt in zip(src.parameters(), tgt.parameters()):
            pt.data.copy_(tau * p.data + (1.0 - tau) * pt.data)

    def _rebuild_networks(self, state_dim: int) -> None:
        self.obs_dim = state_dim
        self.actor      = TD3Actor(state_dim, self.act_dim,
                                   self.cfg.hidden_dim, self.cfg.n_hidden).to(self.device)
        self.actor_tgt  = TD3Actor(state_dim, self.act_dim,
                                   self.cfg.hidden_dim, self.cfg.n_hidden).to(self.device)
        self.critic     = TD3Critic(state_dim, self.act_dim,
                                    self.cfg.hidden_dim, self.cfg.n_hidden).to(self.device)
        self.critic_tgt = TD3Critic(state_dim, self.act_dim,
                                    self.cfg.hidden_dim, self.cfg.n_hidden).to(self.device)
        self.actor_tgt.load_state_dict(self.actor.state_dict())
        self.critic_tgt.load_state_dict(self.critic.state_dict())
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=self.cfg.lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.cfg.lr)

    # ── training ──────────────────────────────────────────────────────────────
    def train(self, dataset: Dict[str, Any], n_epochs: int = 100) -> None:
        """Train TD3+BC on offline dataset."""
        S, A, R, S2, D = self._dataset_to_tensors(dataset)
        N = S.shape[0]
        if N == 0:
            print("[TD3+BC] D_train is empty — skipping training (policy stays at random init).", flush=True)
            return
        state_dim = int(S.shape[1])
        if state_dim != self.obs_dim:
            self._rebuild_networks(state_dim)
            S, A, R, S2, D = self._dataset_to_tensors(dataset)

        bs = min(self.cfg.batch_size, N)

        # State normalization (Fujimoto & Gu 2021 Table 1: normalize_states=True)
        if self.cfg.normalize_states:
            self.state_mean = S.mean(0, keepdim=True)
            self.state_std  = S.std(0, keepdim=True).clamp(min=1e-3)
            S  = (S  - self.state_mean) / self.state_std
            S2 = (S2 - self.state_mean) / self.state_std

        rng = np.random.default_rng(self.cfg.seed)

        for epoch in range(int(n_epochs)):
            idx = rng.choice(N, size=bs, replace=(N < bs))
            s, a, r, s2, d = S[idx], A[idx], R[idx], S2[idx], D[idx]

            # ── Critic update ──
            with torch.no_grad():
                noise = (torch.randn_like(a) * self.cfg.policy_noise
                         ).clamp(-self.cfg.noise_clip, self.cfg.noise_clip)
                a2 = (self.actor_tgt(s2) + noise).clamp(-1.0, 1.0)
                q1_t, q2_t = self.critic_tgt(s2, a2)
                q_target = r + self.cfg.gamma * (1.0 - d) * torch.min(q1_t, q2_t)

            q1, q2 = self.critic(s, a)
            critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()
            self.loss_history["critic"].append(float(critic_loss))

            # ── Actor update (delayed) ──
            self._train_step += 1
            if self._train_step % self.cfg.policy_freq == 0:
                pi = self.actor(s)
                q1_pi = self.critic.q1_value(s, pi)

                # normalised BC penalty
                lam = self.cfg.alpha_bc / (q1_pi.abs().mean().detach() + 1e-8)
                bc_loss = F.mse_loss(pi, a)
                actor_loss = -lam * q1_pi.mean() + bc_loss

                self.actor_opt.zero_grad()
                actor_loss.backward()
                self.actor_opt.step()
                self.loss_history["actor"].append(float(actor_loss))

                self._soft_update(self.actor, self.actor_tgt)
                self._soft_update(self.critic, self.critic_tgt)

    # ── inference ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def act(self, obs: Any) -> np.ndarray:
        """Return action ∈ [-1,1]^act_dim."""
        s = torch.as_tensor(self._obs_to_state(obs), dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        if self.state_mean is not None:
            s = (s - self.state_mean) / self.state_std
        a = self.actor(s).squeeze(0).cpu().numpy()
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def __call__(self, obs: Any) -> np.ndarray:
        return self.act(obs)
