# =============================================================================
# Calibrated Safe Policy Improvement for Continuous Offline Reinforcement Learning
# Federico Bianchi, Alessandro Farinelli, Alberto Castellini
# NeurIPS 2026
# =============================================================================
"""Conservative Q-Learning (Kumar et al., NeurIPS 2020, arXiv:2006.04779).

SAC with the CQL(H) regularizer on both Q heads:
    Q_loss += alpha_cql * (logsumexp_{a ~ pi, a ~ U[-1,1]} Q(s, a) - Q(s, a_data)).

    agent = CQLAgent(obs_dim, act_dim, cfg)
    agent.train(dataset, n_epochs=200)
    action = agent.act(obs)

Defaults: alpha_cql=5.0, alpha_sac=0.1 (maze domains); the original paper uses
alpha_sac=0.05 on locomotion.
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
    """CQL hyperparameters."""

    hidden_dim: int = 256
    n_hidden:   int = 3

    # Training
    lr:         float = 3e-4
    batch_size: int   = 256
    gamma:      float = 0.99
    tau:        float = 0.005       # Polyak coefficient for the target Q network

    # Fixed SAC temperature (no automatic tuning).
    alpha_sac:  float = 0.1

    # Weight of the conservative term.
    alpha_cql:  float = 5.0

    # Samples per state, for each of the policy and uniform proposals.
    n_action_samples: int = 10

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
        """Elementwise min of the two Q heads."""
        q1, q2 = self.forward(s, a)
        return torch.min(q1, q2)


class GaussianActor(nn.Module):
    """Tanh-squashed Gaussian actor (SAC).

    Mean and log_std heads share a trunk; log_prob includes the tanh
    change-of-variables correction (Haarnoja et al., 2018).
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
        # Tanh Jacobian correction, summed over action dims.
        log_pi = (Normal(mean, std).log_prob(x)
                  - torch.log(1.0 - a.pow(2) + 1e-6)).sum(-1)
        return a, log_pi

    def sample_n(self, s: torch.Tensor, n: int
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample n actions per state: actions [B, n, act_dim], log_probs [B, n]."""
        B = s.shape[0]
        s_rep = s.unsqueeze(1).expand(-1, n, -1).reshape(B * n, -1)
        a, lp = self.sample(s_rep)
        return a.reshape(B, n, -1), lp.reshape(B, n)

    @torch.no_grad()
    def mean_action(self, s: torch.Tensor) -> torch.Tensor:
        """tanh(mean), used at evaluation."""
        mean, _ = self._dist(s)
        return torch.tanh(mean)


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class CQLAgent:
    """CQL agent for continuous-action offline RL.

    Dict observations map to concat(observation, desired_goal). When the goal
    keys are absent (flat-observation domains) the goal part has width 0.
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
        """Create Q, target Q, actor and optimizers for state_dim."""
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
        """Flatten an observation; dicts become concat(observation, desired_goal)."""
        if isinstance(obs, dict):
            o = np.asarray(obs["observation"],   dtype=np.float32).reshape(-1)
            g = np.asarray(obs.get("desired_goal", np.zeros(0)),
                           dtype=np.float32).reshape(-1)
            return np.concatenate([o, g])
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    def _dataset_to_tensors(
        self, dataset: Dict[str, Any]
    ) -> Tuple[torch.Tensor, ...]:
        """Convert a transition dict to tensors (S, A, R, S2, D)."""
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

    # ── conservative term ─────────────────────────────────────────────────────

    def _cql_term(
        self,
        q_net_fn,          # callable: (s [B*n, sd], a [B*n, ad]) -> q [B*n]
        s: torch.Tensor,   # [B, sd]
        a_data: torch.Tensor,  # [B, ad]
    ) -> torch.Tensor:
        """CQL(H) term: mean of logsumexp_{a ~ pi or U}(Q(s, a)) - Q(s, a_data).

        Policy samples are importance-corrected by -log pi(a|s); the constant
        uniform log-density is not subtracted from the uniform samples.
        """
        B  = s.shape[0]
        n  = self.cfg.n_action_samples
        ad = self.act_dim

        with torch.no_grad():
            a_pi, log_pi = self.actor.sample_n(s, n)   # [B, n, ad], [B, n]
        s_rep = s.unsqueeze(1).expand(-1, n, -1).reshape(B * n, -1)
        q_pi  = q_net_fn(s_rep, a_pi.reshape(B * n, ad)).reshape(B, n)
        # log E_pi[exp(Q) / pi] estimates log of the integral of exp(Q) over a.
        q_pi_is = q_pi - log_pi.detach()

        # Uniform samples on [-1, 1]^ad; log-density term omitted.
        a_rand = torch.FloatTensor(B, n, ad).uniform_(-1.0, 1.0).to(self.device)
        q_rand = q_net_fn(s_rep, a_rand.reshape(B * n, ad)).reshape(B, n)

        all_q  = torch.cat([q_pi_is, q_rand], dim=1)            # [B, 2n]
        lse    = torch.logsumexp(all_q, dim=1) - np.log(2 * n)  # [B]

        q_data = q_net_fn(s, a_data)                             # [B]

        return (lse - q_data).mean()

    # ── training ──────────────────────────────────────────────────────────────

    def _soft_update(self) -> None:
        """Polyak update of the target Q network."""
        for p, pt in zip(self.q_net.parameters(), self.q_tgt.parameters()):
            pt.data.copy_(self.cfg.tau * p.data + (1.0 - self.cfg.tau) * pt.data)

    def train(self, dataset: Dict[str, Any], n_epochs: int = 200) -> None:
        """Train on an offline transition dict; each epoch is one gradient step."""
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

            q_loss = q_std_loss + cql_loss
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

    # ── inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def act(self, obs: Any) -> np.ndarray:
        """Return the deterministic mean action in [-1, 1]^act_dim."""
        s = torch.as_tensor(self._obs_to_state(obs),
                            dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.state_mean is not None:
            s = (s - self.state_mean) / self.state_std
        a = self.actor.mean_action(s).squeeze(0).cpu().numpy()
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def __call__(self, obs: Any) -> np.ndarray:
        """Alias for act()."""
        return self.act(obs)
