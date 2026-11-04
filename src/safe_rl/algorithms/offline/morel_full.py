# =============================================================================
# Calibrated Safe Policy Improvement for Continuous Offline Reinforcement Learning
# Federico Bianchi, Alessandro Farinelli, Alberto Castellini
# NeurIPS 2026
# =============================================================================
"""MOReL (Kidambi et al., NeurIPS 2020, arXiv:2005.05951) with a SAC policy learner.

Pessimistic MDP built on the same MSE dynamics ensemble used by Cal-SPI:
  - USAD(s, a) <= tau: ensemble transition and domain reward;
  - USAD(s, a) >  tau: HALT, i.e. reward -kappa and done = 1 (absorbing terminal).
tau is either fixed or a percentile of USAD on the training data. The policy is
trained by SAC on short model rollouts started from dataset states.

    agent = MORelFullAgent(ensemble, obs_dim, act_dim, cfg)
    agent.train(dataset, n_epochs=200)
    action = agent.act(obs)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from safe_rl.offline.ensemble_dynamics_sb3 import EnsembleDynamicsSB3


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MORelFullConfig:
    """MOReL pessimistic-MDP and SAC hyperparameters."""

    # Pessimistic MDP
    tau:            float = 0.2       # fixed USAD threshold, used when tau_percentile <= 0
    tau_percentile: float = 0.0       # > 0: tau = this percentile of USAD on the training data
                                      # (95.0 on MuJoCo domains, 0.0 on PointMaze)
    kappa:          float = 50.0      # HALT penalty: r = -kappa, done = 1
    rollout_length: int   = 5         # model rollout horizon H
    rollout_batch:  int   = 1000      # rollouts started per epoch

    # SAC
    hidden_dim:  int   = 256
    n_hidden:    int   = 2
    lr:          float = 3e-4
    batch_size:  int   = 256
    gamma:       float = 0.99
    tau_soft:    float = 0.005        # Polyak coefficient for the target critic
    sac_alpha:   float = 0.2          # entropy coefficient

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


class GaussianActor(nn.Module):
    """Tanh-squashed Gaussian actor (SAC); forward returns (action, log_prob)."""
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, state_dim: int, act_dim: int, hidden_dim: int, n_hidden: int):
        super().__init__()
        self.net = _mlp(state_dim, hidden_dim, hidden_dim, n_hidden - 1)
        self.mu_head      = nn.Linear(hidden_dim, act_dim)
        self.log_std_head = nn.Linear(hidden_dim, act_dim)

    def forward(
        self, s: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.net(s))
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()

        if deterministic:
            a = torch.tanh(mu)
            log_prob = torch.zeros(s.shape[0], device=s.device)
        else:
            eps = torch.randn_like(mu)
            u   = mu + std * eps
            a   = torch.tanh(u)
            # Gaussian log-density with tanh Jacobian correction.
            log_prob = (
                -0.5 * ((u - mu) / (std + 1e-8)).pow(2)
                - log_std
                - 0.5 * np.log(2 * np.pi)
                - torch.log(1.0 - a.pow(2) + 1e-6)
            ).sum(-1)
        return a, log_prob


class SACCritic(nn.Module):
    """Double-Q critic."""
    def __init__(self, state_dim: int, act_dim: int, hidden_dim: int, n_hidden: int):
        super().__init__()
        self.q1 = _mlp(state_dim + act_dim, 1, hidden_dim, n_hidden)
        self.q2 = _mlp(state_dim + act_dim, 1, hidden_dim, n_hidden)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class MORelFullAgent:
    """MOReL agent: pessimistic MDP over a trained dynamics ensemble, SAC policy."""

    def __init__(
        self,
        ensemble: EnsembleDynamicsSB3,
        obs_dim: int,
        act_dim: int,
        cfg: Optional[MORelFullConfig] = None,
        reward_fn=None,
    ):
        """
        Args:
            ensemble: trained dynamics ensemble; members map [obs, act] to obs deltas.
            reward_fn: optional callable(obs_flat, act, next_obs_flat) -> (rew, done),
                float32 arrays of shape [B]. If None, the PointMaze proxy is used:
                rew = -||xy' - goal||, done = (distance < 0.45).
        """
        self.ensemble  = ensemble
        self.obs_dim   = int(obs_dim)
        self.act_dim   = int(act_dim)
        self.cfg       = cfg if cfg is not None else MORelFullConfig()
        self.device    = torch.device(self.cfg.device)
        self.reward_fn = reward_fn

        torch.manual_seed(self.cfg.seed)
        self.rng = np.random.default_rng(self.cfg.seed)

        self.actor     = GaussianActor(obs_dim, act_dim,
                                       self.cfg.hidden_dim, self.cfg.n_hidden).to(self.device)
        self.critic    = SACCritic(obs_dim, act_dim,
                                   self.cfg.hidden_dim, self.cfg.n_hidden).to(self.device)
        self.critic_tgt = SACCritic(obs_dim, act_dim,
                                    self.cfg.hidden_dim, self.cfg.n_hidden).to(self.device)
        self.critic_tgt.load_state_dict(self.critic.state_dict())

        self.a_opt = optim.Adam(self.actor.parameters(),   lr=self.cfg.lr)
        self.c_opt = optim.Adam(self.critic.parameters(),  lr=self.cfg.lr)

        self.loss_history: Dict[str, list] = {"critic": [], "actor": []}

    # ── helpers ───────────────────────────────────────────────────────────────
    def _obs_to_state(self, obs: Any) -> np.ndarray:
        """Flatten an observation; dicts become concat(observation, desired_goal)."""
        if isinstance(obs, dict):
            o = np.asarray(obs["observation"], dtype=np.float32).reshape(-1)
            g = np.asarray(obs.get("desired_goal", np.zeros(2)), dtype=np.float32).reshape(-1)
            return np.concatenate([o, g])
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    def _usad(self, obs_flat: np.ndarray, act: np.ndarray) -> np.ndarray:
        """Max pairwise L2 distance between member predictions, shape [B].

        obs_flat is [B, obs_dim_raw] (observation without goal), act is [B, act_dim].
        """
        B = obs_flat.shape[0]
        X = np.concatenate([obs_flat, act], axis=-1).astype(np.float32)
        Xt = torch.as_tensor(X, dtype=torch.float32)
        E = len(self.ensemble.ensemble)
        deltas = []
        with torch.no_grad():
            for m in self.ensemble.ensemble:
                deltas.append(m(Xt).cpu().numpy())
        deltas = np.stack(deltas, axis=0)  # [E, B, obs_dim]
        usad = np.zeros(B, dtype=np.float64)
        for i in range(E):
            for j in range(i + 1, E):
                d = np.linalg.norm(deltas[i] - deltas[j], axis=-1)
                usad = np.maximum(usad, d)
        return usad.astype(np.float32)

    def _model_rollout(
        self,
        init_obs_flat: np.ndarray,    # [B, obs_dim_raw]
        init_goal: np.ndarray,         # [B, goal_dim]
        desired_goal: np.ndarray,      # [B, goal_dim], fixed during the rollout
    ) -> Dict[str, np.ndarray]:
        """Roll out the pessimistic MDP for rollout_length steps under the current actor.

        Rollouts that terminate restart from their initial state. Returns a
        transition dict whose observations include the goal.
        """
        B = init_obs_flat.shape[0]
        H = self.cfg.rollout_length
        device = self.device

        all_obs, all_act, all_rew, all_nxt, all_done = [], [], [], [], []

        obs = init_obs_flat.copy()

        for _ in range(H):
            state = np.concatenate([obs, desired_goal], axis=-1)
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
            with torch.no_grad():
                a_t, _ = self.actor(state_t, deterministic=False)
            act = a_t.cpu().numpy().clip(-1.0, 1.0)

            usad = self._usad(obs, act)
            ood_mask = usad > self.cfg.tau

            # One ensemble member drawn independently per sample and step.
            k_idx = self.rng.integers(0, len(self.ensemble.ensemble), size=B)
            X = np.concatenate([obs, act], axis=-1).astype(np.float32)
            Xt = torch.as_tensor(X, dtype=torch.float32)
            next_obs_flat = np.empty_like(obs)
            with torch.no_grad():
                for k in range(len(self.ensemble.ensemble)):
                    mask_k = k_idx == k
                    if not mask_k.any():
                        continue
                    delta = self.ensemble.ensemble[k](Xt[mask_k]).cpu().numpy()
                    next_obs_flat[mask_k] = obs[mask_k] + delta

            if self.reward_fn is not None:
                rew, done = self.reward_fn(obs, act, next_obs_flat)
                rew = rew.astype(np.float32)
                done = done.astype(np.float32)
            else:
                dist = np.linalg.norm(
                    next_obs_flat[:, :2] - desired_goal[:, :2], axis=-1
                )
                rew = -dist.astype(np.float32)
                done = (dist < 0.45).astype(np.float32)

            # HALT: the penalty replaces the domain reward and the transition is terminal.
            rew[ood_mask] = -float(self.cfg.kappa)
            done[ood_mask] = 1.0

            all_obs.append(np.concatenate([obs, desired_goal], axis=-1))
            all_act.append(act)
            all_rew.append(rew)
            all_nxt.append(np.concatenate([next_obs_flat, desired_goal], axis=-1))
            all_done.append(done)

            obs = np.where(done[:, None], init_obs_flat, next_obs_flat)

        return {
            "obs_observation": np.concatenate(all_obs, axis=0),
            "act":             np.concatenate(all_act, axis=0),
            "rew":             np.concatenate(all_rew, axis=0),
            "next_observation": np.concatenate(all_nxt, axis=0),
            "done":            np.concatenate(all_done, axis=0),
        }

    def _soft_update(self) -> None:
        """Polyak update of the target critic."""
        tau = self.cfg.tau_soft
        for p, pt in zip(self.critic.parameters(), self.critic_tgt.parameters()):
            pt.data.copy_(tau * p.data + (1.0 - tau) * pt.data)

    # ── training ──────────────────────────────────────────────────────────────
    def train(self, dataset: Dict[str, Any], n_epochs: int = 200) -> None:
        """Train SAC on fresh model rollouts; each epoch is one rollout batch and one step.

        If cfg.tau_percentile > 0, cfg.tau is overwritten from the dataset USAD.
        """
        def _t(x): return torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)

        obs_raw = np.asarray(dataset["obs_observation"], dtype=np.float32)
        goal    = np.asarray(dataset.get("obs_desired_goal",
                             np.zeros((len(obs_raw), 2), dtype=np.float32)), dtype=np.float32)
        N = len(obs_raw)
        if N == 0:
            print("[MOReL] D_train is empty — skipping training (policy stays at random init).", flush=True)
            return
        bs = min(self.cfg.batch_size, N)
        rng = np.random.default_rng(self.cfg.seed)

        if self.cfg.tau_percentile > 0:
            act_raw = np.asarray(dataset["act"], dtype=np.float32)
            usad_train = self._usad(obs_raw, act_raw)
            self.cfg.tau = float(np.percentile(usad_train, self.cfg.tau_percentile))
            print(f"[MOReL] tau set to {self.cfg.tau_percentile:.0f}th percentile of USAD: {self.cfg.tau:.4f}")

        for epoch in range(int(n_epochs)):
            start_idx = rng.choice(N, size=min(self.cfg.rollout_batch, N), replace=False)
            synthetic = self._model_rollout(
                init_obs_flat=obs_raw[start_idx],
                init_goal=goal[start_idx],
                desired_goal=goal[start_idx],
            )

            M = len(synthetic["act"])
            idx = rng.choice(M, size=bs, replace=(M < bs))

            s  = _t(synthetic["obs_observation"][idx])
            a  = _t(synthetic["act"][idx])
            r  = _t(synthetic["rew"][idx])
            s2 = _t(synthetic["next_observation"][idx])
            d  = _t(synthetic["done"][idx])

            # Critic: soft Bellman target.
            with torch.no_grad():
                a2_t, log_pi2 = self.actor(s2)
                q1_t, q2_t = self.critic_tgt(s2, a2_t)
                q_target = r + self.cfg.gamma * (1.0 - d) * (
                    torch.min(q1_t, q2_t) - self.cfg.sac_alpha * log_pi2
                )
            q1, q2 = self.critic(s, a)
            c_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
            self.c_opt.zero_grad()
            c_loss.backward()
            self.c_opt.step()
            self.loss_history["critic"].append(float(c_loss))

            # Actor.
            a_t, log_pi = self.actor(s)
            q1_pi, q2_pi = self.critic(s, a_t)
            a_loss = (self.cfg.sac_alpha * log_pi - torch.min(q1_pi, q2_pi)).mean()
            self.a_opt.zero_grad()
            a_loss.backward()
            self.a_opt.step()
            self.loss_history["actor"].append(float(a_loss))

            self._soft_update()

    # ── inference ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def act(self, obs: Any) -> np.ndarray:
        """Return the deterministic action tanh(mu) in [-1, 1]^act_dim."""
        s = torch.as_tensor(self._obs_to_state(obs), dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        a, _ = self.actor(s, deterministic=True)
        return np.clip(a.squeeze(0).cpu().numpy(), -1.0, 1.0).astype(np.float32)

    def __call__(self, obs: Any) -> np.ndarray:
        """Alias for act()."""
        return self.act(obs)
