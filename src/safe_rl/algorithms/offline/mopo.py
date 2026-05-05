# =============================================================================
# Anonymous authors. Affiliations and acknowledgements will be added on
# acceptance of the associated paper. Submitted to NeurIPS 2026 (double-blind).
# =============================================================================
"""
mopo.py — MOPO (Yu et al., NeurIPS 2020)
arXiv:2005.05951

Model-based Offline Policy Optimization with soft reward penalization.

Uses a heteroscedastic NLL ensemble (correct for MOPO: soft penalty, NOT hard gate).
Augments rewards: r_pen(s,a) = r(s,a) - lambda * sigma(s,a)
where sigma is the ensemble's predictive standard deviation.

Trains SAC on synthetic rollouts from the penalized model.

Key differences vs Cal-SPI:
  - NLL ensemble (heteroscedastic), NOT MSE   ← intentional for MOPO
  - Soft reward penalty (not hard gate)
  - SAC optimizer over synthetic data

Interface:
  agent = MOPOAgent(obs_dim, act_dim, cfg)
  agent.train(dataset, n_epochs=200)
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
class MOPOConfig:
    # Ensemble (NLL, heteroscedastic)
    ensemble_size: int   = 4
    ensemble_hidden: int = 512
    ensemble_epochs: int = 50
    ensemble_lr:   float = 1e-3
    ensemble_batch:  int = 256

    # P-MDP
    lam:           float = 1.0    # penalty weight λ for sigma
    rollout_length:  int = 5
    rollout_batch:   int = 1000

    # SAC optimizer
    hidden_dim:  int   = 256
    n_hidden:    int   = 2
    lr:          float = 3e-4
    batch_size:  int   = 256
    gamma:       float = 0.99
    tau:         float = 0.005
    sac_alpha:   float = 0.2

    device: str = "cpu"
    seed:   int = 0


# ─────────────────────────────────────────────────────────────────────────────
# NLL Ensemble (MOPO-style, heteroscedastic)
# ─────────────────────────────────────────────────────────────────────────────

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


class NLLMember(nn.Module):
    """Single ensemble member: outputs (mu, log_sigma^2) for NLL loss."""
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.mu_head     = nn.Linear(hidden, out_dim)
        self.logvar_head = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mu     = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(2 * LOG_STD_MIN, 2 * LOG_STD_MAX)
        return mu, logvar


class NLLEnsemble:
    """
    Heteroscedastic NLL ensemble for MOPO.
    NOTE: uses NLL (sigma learned), appropriate for soft penalty.
    """

    def __init__(self, obs_dim: int, act_dim: int, cfg: MOPOConfig):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        torch.manual_seed(cfg.seed)
        in_dim  = obs_dim + act_dim
        out_dim = obs_dim
        self.members: list = []
        self.optims: list = []
        for k in range(cfg.ensemble_size):
            torch.manual_seed(cfg.seed + 1000 * k)
            m = NLLMember(in_dim, out_dim, hidden=cfg.ensemble_hidden).to(self.device)
            self.members.append(m)
            self.optims.append(optim.Adam(m.parameters(), lr=cfg.ensemble_lr))

    def fit(self, obs: np.ndarray, act: np.ndarray, next_obs: np.ndarray) -> None:
        """Train ensemble on transitions (obs, act) -> delta = next_obs - obs."""
        X = np.concatenate([obs, act], axis=-1).astype(np.float32)
        y = (next_obs - obs).astype(np.float32)
        N = X.shape[0]
        Xt = torch.as_tensor(X, device=self.device)
        yt = torch.as_tensor(y, device=self.device)
        bs = self.cfg.ensemble_batch

        for _ in range(self.cfg.ensemble_epochs):
            perm = torch.randperm(N, device=self.device)
            for start in range(0, N, bs):
                idx = perm[start:start + bs]
                xb, yb = Xt[idx], yt[idx]
                for m, opt in zip(self.members, self.optims):
                    opt.zero_grad()
                    mu, logvar = m(xb)
                    # Gaussian NLL: 0.5*(logvar + (y-mu)^2/exp(logvar))
                    loss = (0.5 * (logvar + ((yb - mu).pow(2) / logvar.exp()))).mean()
                    loss.backward()
                    opt.step()

    @torch.no_grad()
    def predict(self, obs: np.ndarray, act: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (mu_bar, sigma_agg) where:
          mu_bar: [B, obs_dim] — ensemble mean of means
          sigma_agg: [B] — mean predictive std across members and dims
        """
        X = np.concatenate([obs, act], axis=-1).astype(np.float32)
        Xt = torch.as_tensor(X, device=self.device)

        mus, vars_ = [], []
        for m in self.members:
            mu, logvar = m(Xt)
            mus.append(mu.cpu().numpy())
            vars_.append(logvar.exp().cpu().numpy())

        mus  = np.stack(mus,  axis=0)  # [E, B, d]
        vars_= np.stack(vars_, axis=0)  # [E, B, d]

        mu_bar = mus.mean(axis=0)           # [B, d]
        # MOPO penalty signal: max_k ||std_k(s,a)||_2  (Yu et al. 2020, Sec. 4, eq. 5)
        # Uses the learned NLL predictive std of each member (aleatoric + epistemic).
        # Each member predicts a diagonal Gaussian; ||std_k||_F = ||sqrt(exp(logvar_k))||_2.
        stds = np.sqrt(vars_)                               # [E, B, d]
        sigma_agg = np.linalg.norm(stds, axis=-1).max(axis=0)  # [B]
        return mu_bar, sigma_agg

    @torch.no_grad()
    def sample_step(
        self, obs: np.ndarray, act: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Sample next obs using a random member.
        obs: [B, obs_dim], act: [B, act_dim]
        Returns: [B, obs_dim]
        """
        B = obs.shape[0]
        X = np.concatenate([obs, act], axis=-1).astype(np.float32)
        Xt = torch.as_tensor(X, device=self.device)
        k_idx = rng.integers(0, len(self.members), size=B)
        next_obs = np.empty_like(obs)
        for k, m in enumerate(self.members):
            mask = k_idx == k
            if not mask.any():
                continue
            mu_k, logvar_k = m(Xt[mask])
            std_k = logvar_k.exp().sqrt()
            # reparameterisation sample
            eps = torch.randn_like(mu_k)
            delta = (mu_k + std_k * eps).cpu().numpy()
            next_obs[mask] = obs[mask] + delta
        return next_obs


# ─────────────────────────────────────────────────────────────────────────────
# SAC helper networks (same as MORelFull)
# ─────────────────────────────────────────────────────────────────────────────

def _mlp(in_dim: int, out_dim: int, hidden_dim: int, n_hidden: int) -> nn.Sequential:
    layers: list = []
    prev = in_dim
    for _ in range(n_hidden):
        layers += [nn.Linear(prev, hidden_dim), nn.ReLU()]
        prev = hidden_dim
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, state_dim: int, act_dim: int, hidden_dim: int, n_hidden: int):
        super().__init__()
        self.net = _mlp(state_dim, hidden_dim, hidden_dim, n_hidden - 1)
        self.mu_h      = nn.Linear(hidden_dim, act_dim)
        self.logstd_h  = nn.Linear(hidden_dim, act_dim)

    def forward(self, s: torch.Tensor, deterministic: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        h   = F.relu(self.net(s))
        mu  = self.mu_h(h)
        lsd = self.logstd_h(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = lsd.exp()
        if deterministic:
            return torch.tanh(mu), torch.zeros(s.shape[0], device=s.device)
        eps = torch.randn_like(mu)
        u   = mu + std * eps
        a   = torch.tanh(u)
        lp  = (-0.5 * ((u - mu) / (std + 1e-8)).pow(2) - lsd
               - 0.5 * np.log(2 * np.pi)
               - torch.log(1.0 - a.pow(2) + 1e-6)).sum(-1)
        return a, lp


class SACCritic(nn.Module):
    def __init__(self, state_dim: int, act_dim: int, hidden_dim: int, n_hidden: int):
        super().__init__()
        self.q1 = _mlp(state_dim + act_dim, 1, hidden_dim, n_hidden)
        self.q2 = _mlp(state_dim + act_dim, 1, hidden_dim, n_hidden)

    def forward(self, s: torch.Tensor, a: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class MOPOAgent:
    """
    MOPO offline RL agent.
    Fits a NLL ensemble on the dataset, generates penalized synthetic rollouts,
    then trains SAC on those rollouts.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        cfg: Optional[MOPOConfig] = None,
    ):
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.cfg     = cfg if cfg is not None else MOPOConfig()
        self.device  = torch.device(self.cfg.device)
        self.rng     = np.random.default_rng(self.cfg.seed)
        self._obs_dim_raw = int(obs_dim)   # raw obs (no goal), for NLL ensemble

        self.ensemble = NLLEnsemble(obs_dim, act_dim, self.cfg)

        torch.manual_seed(self.cfg.seed)
        self.actor     = GaussianActor(obs_dim, act_dim,
                                       self.cfg.hidden_dim, self.cfg.n_hidden).to(self.device)
        self.critic    = SACCritic(obs_dim, act_dim,
                                   self.cfg.hidden_dim, self.cfg.n_hidden).to(self.device)
        self.critic_tgt = SACCritic(obs_dim, act_dim,
                                    self.cfg.hidden_dim, self.cfg.n_hidden).to(self.device)
        self.critic_tgt.load_state_dict(self.critic.state_dict())

        self.a_opt = optim.Adam(self.actor.parameters(),  lr=self.cfg.lr)
        self.c_opt = optim.Adam(self.critic.parameters(), lr=self.cfg.lr)

        self.loss_history: Dict[str, list] = {"critic": [], "actor": []}

    # ── helpers ───────────────────────────────────────────────────────────────
    def _obs_to_state(self, obs: Any) -> np.ndarray:
        if isinstance(obs, dict):
            o = np.asarray(obs["observation"], dtype=np.float32).reshape(-1)
            g = np.asarray(obs.get("desired_goal", np.zeros(2)), dtype=np.float32).reshape(-1)
            return np.concatenate([o, g])
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    def _soft_update(self) -> None:
        tau = self.cfg.tau
        for p, pt in zip(self.critic.parameters(), self.critic_tgt.parameters()):
            pt.data.copy_(tau * p.data + (1.0 - tau) * pt.data)

    def _rebuild_for_state_dim(self, state_dim: int, obs_dim_raw: int) -> None:
        """Rebuild SAC networks for state_dim; rebuild NLL ensemble for obs_dim_raw."""
        if state_dim == self.obs_dim and obs_dim_raw == self._obs_dim_raw:
            return
        self.obs_dim = state_dim
        self._obs_dim_raw = obs_dim_raw
        # NLL ensemble operates on raw obs (no goal)
        self.ensemble = NLLEnsemble(obs_dim_raw, self.act_dim, self.cfg)
        # SAC networks operate on state = obs + goal
        self.actor      = GaussianActor(state_dim, self.act_dim,
                                        self.cfg.hidden_dim, self.cfg.n_hidden).to(self.device)
        self.critic     = SACCritic(state_dim, self.act_dim,
                                    self.cfg.hidden_dim, self.cfg.n_hidden).to(self.device)
        self.critic_tgt = SACCritic(state_dim, self.act_dim,
                                    self.cfg.hidden_dim, self.cfg.n_hidden).to(self.device)
        self.critic_tgt.load_state_dict(self.critic.state_dict())
        self.a_opt = optim.Adam(self.actor.parameters(),  lr=self.cfg.lr)
        self.c_opt = optim.Adam(self.critic.parameters(), lr=self.cfg.lr)

    def _build_state_arrays(self, dataset: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build (obs_concat, next_obs_concat, act) with goal concatenated."""
        obs_raw  = np.asarray(dataset["obs_observation"],  dtype=np.float32)
        nxt_raw  = np.asarray(dataset["next_observation"], dtype=np.float32)
        goal     = np.asarray(dataset.get("obs_desired_goal",
                              np.zeros((len(obs_raw), 2), dtype=np.float32)), dtype=np.float32)
        nxt_goal = np.asarray(dataset.get("next_desired_goal", goal), dtype=np.float32)
        act      = np.asarray(dataset["act"], dtype=np.float32)
        obs = np.concatenate([obs_raw, goal], axis=-1)
        nxt = np.concatenate([nxt_raw, nxt_goal], axis=-1)
        return obs, nxt, act

    def _model_rollout(
        self, init_obs: np.ndarray, obs_dim_raw: int, goal_dim: int
    ) -> Dict[str, np.ndarray]:
        """Generate H-step rollouts with penalized reward."""
        B = init_obs.shape[0]
        H = self.cfg.rollout_length
        all_obs, all_act, all_rew, all_nxt, all_done = [], [], [], [], []

        obs = init_obs.copy()
        for _ in range(H):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                a_t, _ = self.actor(obs_t, deterministic=False)
            act = a_t.cpu().numpy().clip(-1.0, 1.0)

            # Ensemble prediction + sigma
            obs_raw = obs[:, :obs_dim_raw]
            mu_bar, sigma = self.ensemble.predict(obs_raw, act)

            # Penalized reward: distance + -lambda*sigma
            goal = obs[:, obs_dim_raw:]
            dist = np.linalg.norm(mu_bar[:, :min(2, obs_dim_raw)] - goal[:, :2], axis=-1)
            rew = -dist.astype(np.float32) - float(self.cfg.lam) * sigma.astype(np.float32)

            # Next state
            next_raw = self.ensemble.sample_step(obs_raw, act, self.rng)
            next_obs = np.concatenate([next_raw, goal], axis=-1)
            done = (dist < 0.45).astype(np.float32)

            all_obs.append(obs)
            all_act.append(act)
            all_rew.append(rew)
            all_nxt.append(next_obs)
            all_done.append(done)

            obs = np.where(done[:, None], init_obs, next_obs)

        return {
            "obs":  np.concatenate(all_obs,  axis=0),
            "act":  np.concatenate(all_act,  axis=0),
            "rew":  np.concatenate(all_rew,  axis=0),
            "nxt":  np.concatenate(all_nxt,  axis=0),
            "done": np.concatenate(all_done, axis=0),
        }

    # ── training ──────────────────────────────────────────────────────────────
    def train(self, dataset: Dict[str, Any], n_epochs: int = 200) -> None:
        """Train MOPO on offline dataset."""
        obs, nxt, act = self._build_state_arrays(dataset)
        state_dim   = obs.shape[1]
        obs_dim_raw = int(dataset["obs_observation"].shape[1])
        goal_dim    = state_dim - obs_dim_raw

        self._rebuild_for_state_dim(state_dim, obs_dim_raw)

        # Fit ensemble on raw obs (observation space only, not goal-concatenated)
        obs_raw = np.asarray(dataset["obs_observation"], dtype=np.float32)
        nxt_raw = np.asarray(dataset["next_observation"], dtype=np.float32)
        print("[MOPO] Fitting NLL ensemble...", flush=True)
        self.ensemble.fit(obs_raw, act, nxt_raw)

        N  = obs.shape[0]
        bs = min(self.cfg.batch_size, N)
        rng = np.random.default_rng(self.cfg.seed)

        def _t(x): return torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)

        for epoch in range(int(n_epochs)):
            # Generate synthetic rollouts
            idx0 = rng.choice(N, size=min(self.cfg.rollout_batch, N), replace=False)
            synth = self._model_rollout(obs[idx0], obs_dim_raw, goal_dim)

            M = len(synth["act"])
            idx = rng.choice(M, size=bs, replace=(M < bs))

            s  = _t(synth["obs"][idx])
            a  = _t(synth["act"][idx])
            r  = _t(synth["rew"][idx])
            s2 = _t(synth["nxt"][idx])
            d  = _t(synth["done"][idx])

            # Critic update
            with torch.no_grad():
                a2, lp2 = self.actor(s2)
                q1t, q2t = self.critic_tgt(s2, a2)
                q_tgt = r + self.cfg.gamma * (1.0 - d) * (
                    torch.min(q1t, q2t) - self.cfg.sac_alpha * lp2
                )
            q1, q2 = self.critic(s, a)
            c_loss = F.mse_loss(q1, q_tgt) + F.mse_loss(q2, q_tgt)
            self.c_opt.zero_grad(); c_loss.backward(); self.c_opt.step()
            self.loss_history["critic"].append(float(c_loss))

            # Actor update
            a_t, lp = self.actor(s)
            q1pi, q2pi = self.critic(s, a_t)
            a_loss = (self.cfg.sac_alpha * lp - torch.min(q1pi, q2pi)).mean()
            self.a_opt.zero_grad(); a_loss.backward(); self.a_opt.step()
            self.loss_history["actor"].append(float(a_loss))

            self._soft_update()

    # ── inference ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def act(self, obs: Any) -> np.ndarray:
        s = torch.as_tensor(self._obs_to_state(obs), dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        a, _ = self.actor(s, deterministic=True)
        return np.clip(a.squeeze(0).cpu().numpy(), -1.0, 1.0).astype(np.float32)

    def __call__(self, obs: Any) -> np.ndarray:
        return self.act(obs)


# ─────────────────────────────────────────────────────────────────────────────
# MOPOMSEAgent — MOPO with pre-trained MSE ensemble (no NLL re-training)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MOPOMSEConfig:
    """Config for MOPO-MSE: reuses a pre-trained MSE ensemble, no NLL training."""
    lam:            float = 1.0       # penalty weight λ for epistemic sigma
    rollout_length: int   = 5
    rollout_batch:  int   = 1000

    hidden_dim:  int   = 256
    n_hidden:    int   = 2
    lr:          float = 3e-4
    batch_size:  int   = 256
    gamma:       float = 0.99
    tau_soft:    float = 0.005
    sac_alpha:   float = 0.2

    device: str = "cpu"
    seed:   int = 0


class MOPOMSEAgent:
    """
    MOPO using the pre-trained MSE ensemble (same as Cal-SPI) for fair comparison.

    sigma(s,a) = sqrt( mean_d Var_k[ mu_k(s,a,d) ] )  — epistemic std from MSE ensemble.
    Reward penalization: r_pen = r(s,a) - lam * sigma(s,a)

    Trains SAC on synthetic rollouts with penalized rewards.
    """

    def __init__(
        self,
        ensemble,                        # EnsembleDynamicsSB3 (pre-trained MSE)
        obs_dim: int,                    # full state dim = obs_dim_raw + goal_dim
        act_dim: int,
        cfg: Optional[MOPOMSEConfig] = None,
        reward_fn=None,
    ):
        """
        reward_fn: optional callable(obs_flat, act, next_obs_flat) -> (rew, done)
            both float32 [B] arrays.  If None, uses PointMaze distance heuristic.
            Pass a domain-specific function for flat (non-goal-conditioned) envs.
        """
        self.ensemble  = ensemble
        self.obs_dim   = int(obs_dim)
        self.act_dim   = int(act_dim)
        self.cfg       = cfg if cfg is not None else MOPOMSEConfig()
        self.device    = torch.device(self.cfg.device)
        self.reward_fn = reward_fn

        torch.manual_seed(self.cfg.seed)
        self.rng = np.random.default_rng(self.cfg.seed)

        self.actor      = GaussianActor(obs_dim, act_dim,
                                        self.cfg.hidden_dim, self.cfg.n_hidden).to(self.device)
        self.critic     = SACCritic(obs_dim, act_dim,
                                    self.cfg.hidden_dim, self.cfg.n_hidden).to(self.device)
        self.critic_tgt = SACCritic(obs_dim, act_dim,
                                    self.cfg.hidden_dim, self.cfg.n_hidden).to(self.device)
        self.critic_tgt.load_state_dict(self.critic.state_dict())

        self.a_opt = optim.Adam(self.actor.parameters(),  lr=self.cfg.lr)
        self.c_opt = optim.Adam(self.critic.parameters(), lr=self.cfg.lr)

        self.loss_history: Dict[str, list] = {"critic": [], "actor": []}

    def _obs_to_state(self, obs: Any) -> np.ndarray:
        if isinstance(obs, dict):
            o = np.asarray(obs["observation"],  dtype=np.float32).reshape(-1)
            g = np.asarray(obs.get("desired_goal", np.zeros(2)), dtype=np.float32).reshape(-1)
            return np.concatenate([o, g])
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    def _sigma_batch(self, obs_flat: np.ndarray, act: np.ndarray) -> np.ndarray:
        """
        Epistemic std from MSE ensemble members.
        obs_flat: [B, obs_dim_raw]  (observation only, no goal)
        act:      [B, act_dim]
        Returns:  [B] sigma values = sqrt(mean_d Var_k[mu_k]).
        """
        X  = np.concatenate([obs_flat, act], axis=-1).astype(np.float32)
        Xt = torch.as_tensor(X, dtype=torch.float32)
        deltas = []
        with torch.no_grad():
            for m in self.ensemble.ensemble:
                deltas.append(m(Xt).cpu().numpy())
        deltas  = np.stack(deltas, axis=0)                      # [E, B, obs_dim_raw]
        epi_var = np.var(deltas, axis=0).mean(axis=-1)          # [B]
        return np.sqrt(np.maximum(epi_var, 0.0)).astype(np.float32)

    def _model_rollout(
        self,
        init_obs_flat: np.ndarray,   # [B, obs_dim_raw]
        desired_goal:  np.ndarray,   # [B, goal_dim]
    ) -> Dict[str, np.ndarray]:
        B   = init_obs_flat.shape[0]
        H   = self.cfg.rollout_length
        dev = self.device

        all_obs, all_act, all_rew, all_nxt, all_done = [], [], [], [], []
        obs = init_obs_flat.copy()

        for _ in range(H):
            state   = np.concatenate([obs, desired_goal], axis=-1)
            state_t = torch.as_tensor(state, dtype=torch.float32, device=dev)
            with torch.no_grad():
                a_t, _ = self.actor(state_t, deterministic=False)
            act = a_t.cpu().numpy().clip(-1.0, 1.0)

            # Epistemic sigma penalty
            sigma = self._sigma_batch(obs, act)

            # Ensemble step (random member per sample)
            k_idx = self.rng.integers(0, len(self.ensemble.ensemble), size=B)
            X  = np.concatenate([obs, act], axis=-1).astype(np.float32)
            Xt = torch.as_tensor(X, dtype=torch.float32)
            next_obs_flat = np.empty_like(obs)
            with torch.no_grad():
                for k in range(len(self.ensemble.ensemble)):
                    mk = k_idx == k
                    if not mk.any():
                        continue
                    delta = self.ensemble.ensemble[k](Xt[mk]).cpu().numpy()
                    next_obs_flat[mk] = obs[mk] + delta

            # r_pen = r_domain - lam * sigma (domain fn) or -dist - lam*sigma (PointMaze)
            if self.reward_fn is not None:
                rew, done = self.reward_fn(obs, act, next_obs_flat)
                rew  = rew.astype(np.float32) - float(self.cfg.lam) * sigma
                done = done.astype(np.float32)
            else:
                dist = np.linalg.norm(
                    next_obs_flat[:, :2] - desired_goal[:, :2], axis=-1
                )
                rew  = -dist.astype(np.float32) - float(self.cfg.lam) * sigma
                done = (dist < 0.45).astype(np.float32)

            all_obs.append(np.concatenate([obs,          desired_goal], axis=-1))
            all_act.append(act)
            all_rew.append(rew)
            all_nxt.append(np.concatenate([next_obs_flat, desired_goal], axis=-1))
            all_done.append(done)

            obs = np.where(done[:, None], init_obs_flat, next_obs_flat)

        return {
            "obs_observation":  np.concatenate(all_obs,  axis=0),
            "act":              np.concatenate(all_act,  axis=0),
            "rew":              np.concatenate(all_rew,  axis=0),
            "next_observation": np.concatenate(all_nxt,  axis=0),
            "done":             np.concatenate(all_done, axis=0),
        }

    def _soft_update(self) -> None:
        tau = self.cfg.tau_soft
        for p, pt in zip(self.critic.parameters(), self.critic_tgt.parameters()):
            pt.data.copy_(tau * p.data + (1.0 - tau) * pt.data)

    def train(self, dataset: Dict[str, Any], n_epochs: int = 200) -> None:
        """Train MOPO-MSE on offline dataset (ensemble already pre-trained)."""
        def _t(x): return torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)

        obs_raw = np.asarray(dataset["obs_observation"], dtype=np.float32)
        goal    = np.asarray(dataset.get("obs_desired_goal",
                             np.zeros((len(obs_raw), 2), dtype=np.float32)), dtype=np.float32)
        N  = len(obs_raw)
        if N == 0:
            print("[MOPO] D_train is empty — skipping training (policy stays at random init).", flush=True)
            return
        bs = min(self.cfg.batch_size, N)
        rng = np.random.default_rng(self.cfg.seed)

        for _ in range(int(n_epochs)):
            idx0  = rng.choice(N, size=min(self.cfg.rollout_batch, N), replace=False)
            synth = self._model_rollout(obs_raw[idx0], goal[idx0])

            M   = len(synth["act"])
            idx = rng.choice(M, size=bs, replace=(M < bs))

            s  = _t(synth["obs_observation"][idx])
            a  = _t(synth["act"][idx])
            r  = _t(synth["rew"][idx])
            s2 = _t(synth["next_observation"][idx])
            d  = _t(synth["done"][idx])

            with torch.no_grad():
                a2, lp2 = self.actor(s2)
                q1t, q2t = self.critic_tgt(s2, a2)
                q_tgt = r + self.cfg.gamma * (1.0 - d) * (
                    torch.min(q1t, q2t) - self.cfg.sac_alpha * lp2
                )
            q1, q2 = self.critic(s, a)
            c_loss = F.mse_loss(q1, q_tgt) + F.mse_loss(q2, q_tgt)
            self.c_opt.zero_grad(); c_loss.backward(); self.c_opt.step()
            self.loss_history["critic"].append(float(c_loss))

            a_t, lp = self.actor(s)
            q1pi, q2pi = self.critic(s, a_t)
            a_loss = (self.cfg.sac_alpha * lp - torch.min(q1pi, q2pi)).mean()
            self.a_opt.zero_grad(); a_loss.backward(); self.a_opt.step()
            self.loss_history["actor"].append(float(a_loss))

            self._soft_update()

    @torch.no_grad()
    def act(self, obs: Any) -> np.ndarray:
        s = torch.as_tensor(self._obs_to_state(obs), dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        a, _ = self.actor(s, deterministic=True)
        return np.clip(a.squeeze(0).cpu().numpy(), -1.0, 1.0).astype(np.float32)

    def __call__(self, obs: Any) -> np.ndarray:
        return self.act(obs)
