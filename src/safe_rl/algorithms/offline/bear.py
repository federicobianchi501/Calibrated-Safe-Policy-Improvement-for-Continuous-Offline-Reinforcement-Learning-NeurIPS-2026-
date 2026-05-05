# =============================================================================
# Anonymous authors. Affiliations and acknowledgements will be added on
# acceptance of the associated paper. Submitted to NeurIPS 2026 (double-blind).
# =============================================================================
"""
bear.py — Bootstrapping Error Accumulation Reduction (Kumar et al., NeurIPS 2019)
arXiv:1906.00949

BEAR constrains the learned policy to stay within the support of the behavior
policy using a Maximum Mean Discrepancy (MMD) constraint enforced via a
Lagrangian dual variable:

  Actor: max_π  E[Q(s,π(s))]  s.t.  MMD(π(·|s), π_β(·|s)) ≤ ε_mmd
  Dual:  min_λ≥0  λ·(MMD - ε_mmd)  [λ updated by gradient ascent]

The behavior policy π_β is modeled by a conditional VAE (encoder + decoder).

Interface (identical to IQL/CQL):
  agent = BEARAgent(obs_dim, act_dim, cfg)
  agent.train(dataset, n_epochs=200)
  action = agent.act(obs_dict)  # best of n_vae_samples by Q-value

Hyperparameters follow the original paper:
  Locomotion (Hopper, Walker2d) — Table 1:
    delta_mmd=0.05, kernel='laplacian', kernel_bw=20,
    n_mmd_samples=4, alpha_sac=0.05
  AntMaze / PointMaze (navigation):
    delta_mmd=0.1, same kernel, alpha_sac=0.1
"""
from __future__ import annotations

from dataclasses import dataclass
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
class BEARConfig:
    # Architecture
    hidden_dim: int = 256
    n_hidden:   int = 3

    # VAE for behavior policy (paper: hidden=750, latent=2*act_dim)
    vae_hidden_dim: int = 750
    # latent_dim: set to 2*act_dim in __init__ if 0
    vae_latent_dim: int = 0

    # Training
    lr:         float = 3e-4
    lr_vae:     float = 3e-4
    batch_size: int   = 256
    gamma:      float = 0.99
    tau:        float = 0.005       # soft target update

    # SAC temperature (fixed for offline RL)
    # AntMaze/PointMaze: 0.1; Locomotion: 0.05
    alpha_sac:  float = 0.05

    # MMD constraint threshold
    # Locomotion: 0.05 (Table 1); AntMaze/PointMaze: 0.1
    delta_mmd:  float = 0.05

    # MMD kernel: 'laplacian' (default) or 'gaussian'
    # Paper uses Laplacian with bandwidth σ=20
    kernel:     str   = "laplacian"
    kernel_bw:  float = 20.0       # bandwidth σ for MMD kernel

    # Number of samples for MMD estimation (paper: 4)
    n_mmd_samples: int = 4

    # Lagrange multiplier initialization and step size
    log_lambda_init: float = 0.0   # start at λ=1
    lr_lambda:       float = 1e-3

    # Number of VAE samples at inference (best-Q selection)
    n_vae_samples:   int = 10

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
    def __init__(self, state_dim: int, act_dim: int,
                 hidden_dim: int, n_hidden: int):
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
    """SAC-style stochastic actor with tanh squashing."""
    def __init__(self, state_dim: int, act_dim: int,
                 hidden_dim: int, n_hidden: int):
        super().__init__()
        self.trunk   = _mlp(state_dim, hidden_dim, hidden_dim, n_hidden - 1)
        self.mean_fc = nn.Linear(hidden_dim, act_dim)
        self.lstd_fc = nn.Linear(hidden_dim, act_dim)

    def _dist(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(s)
        return self.mean_fc(h), self.lstd_fc(h).clamp(LOG_STD_MIN, LOG_STD_MAX).exp()

    def sample(self, s: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, std = self._dist(s)
        eps = torch.randn_like(mean)
        x = mean + std * eps
        a = torch.tanh(x)
        log_pi = (Normal(mean, std).log_prob(x)
                  - torch.log(1.0 - a.pow(2) + 1e-6)).sum(-1)
        return a, log_pi

    def sample_n(self, s: torch.Tensor, n: int
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """[B,n,act_dim], [B,n]"""
        B = s.shape[0]
        s_rep = s.unsqueeze(1).expand(-1, n, -1).reshape(B * n, -1)
        a, lp = self.sample(s_rep)
        return a.reshape(B, n, -1), lp.reshape(B, n)

    @torch.no_grad()
    def mean_action(self, s: torch.Tensor) -> torch.Tensor:
        mean, _ = self._dist(s)
        return torch.tanh(mean)


class BehaviorVAE(nn.Module):
    """
    Conditional VAE for modeling the behavior policy π_β(a|s).
    Architecture follows the BEAR paper: hidden=750, latent=2*act_dim.

    Encoder: (s, a) → (μ_z, σ_z)
    Decoder: (s, z) → a  (tanh-squashed)
    """
    def __init__(self, state_dim: int, act_dim: int,
                 hidden_dim: int, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(state_dim + act_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),           nn.ReLU(),
        )
        self.mu_fc  = nn.Linear(hidden_dim, latent_dim)
        self.std_fc = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),              nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )

    def encode(self, s: torch.Tensor, a: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        h   = self.enc(torch.cat([s, a], dim=-1))
        mu  = self.mu_fc(h)
        std = self.std_fc(h).clamp(-4.0, 15.0).exp()  # paper: clamp log_std
        return mu, std

    def decode(self, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.dec(torch.cat([s, z], dim=-1)))

    def forward(self, s: torch.Tensor, a: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (a_recon, mu_z, std_z) for ELBO computation."""
        mu, std = self.encode(s, a)
        z       = mu + std * torch.randn_like(std)
        a_recon = self.decode(s, z)
        return a_recon, mu, std

    @torch.no_grad()
    def sample(self, s: torch.Tensor) -> torch.Tensor:
        """Sample actions from the modeled behavior policy."""
        z = torch.randn(s.shape[0], self.latent_dim, device=s.device).clamp(-0.5, 0.5)
        return self.decode(s, z)

    @torch.no_grad()
    def sample_n(self, s: torch.Tensor, n: int) -> torch.Tensor:
        """Sample n actions per state. Returns [B, n, act_dim]."""
        B   = s.shape[0]
        s_r = s.unsqueeze(1).expand(-1, n, -1).reshape(B * n, -1)
        z   = torch.randn(B * n, self.latent_dim, device=s.device).clamp(-0.5, 0.5)
        return self.decode(s_r, z).reshape(B, n, -1)


# ─────────────────────────────────────────────────────────────────────────────
# MMD kernels
# ─────────────────────────────────────────────────────────────────────────────

def _mmd(
    x: torch.Tensor,        # [B, n, d]  samples from distribution 1
    y: torch.Tensor,        # [B, m, d]  samples from distribution 2
    kernel: str = "laplacian",
    bw: float   = 20.0,
) -> torch.Tensor:
    """
    Biased MMD^2 estimator (Kumar 2019 Appendix A).
    Returns scalar per batch element → shape [B].
    """
    n, m = x.shape[1], y.shape[1]

    def _k(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a,b: [B, ?, d]; compute kernel matrix [B, ?_a, ?_b]
        diff = a.unsqueeze(2) - b.unsqueeze(1)           # [B, ?a, ?b, d]
        dist = diff.pow(2).sum(-1)                        # [B, ?a, ?b]
        if kernel == "laplacian":
            return torch.exp(-(dist + 1e-8).sqrt() / bw)   # +1e-8 for sqrt stability
        return torch.exp(-dist / (2.0 * bw ** 2))

    k_xx = _k(x, x).sum((1, 2)) / (n * n)               # [B]
    k_yy = _k(y, y).sum((1, 2)) / (m * m)               # [B]
    k_xy = _k(x, y).sum((1, 2)) / (n * m)               # [B]
    return (k_xx + k_yy - 2.0 * k_xy).clamp(min=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class BEARAgent:
    """
    BEAR agent for continuous-action offline RL.

    Uses the same observation interface as IQL/CQL:
      dict with 'observation' + 'desired_goal' (PointMaze),
      or flat array (Hopper, Walker2d).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        cfg: Optional[BEARConfig] = None,
    ):
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.cfg     = cfg if cfg is not None else BEARConfig()
        self.device  = torch.device(self.cfg.device)

        torch.manual_seed(self.cfg.seed)
        self._build_networks(obs_dim)

        self.state_mean: Optional[torch.Tensor] = None
        self.state_std:  Optional[torch.Tensor] = None
        self.loss_history: Dict[str, list] = {"q": [], "vae": [], "actor": []}

    def _build_networks(self, state_dim: int) -> None:
        self.obs_dim = state_dim
        sd, ad = state_dim, self.act_dim
        hd, nh = self.cfg.hidden_dim, self.cfg.n_hidden
        lat     = self.cfg.vae_latent_dim if self.cfg.vae_latent_dim > 0 else 2 * ad

        self.q_net = DoubleQNetwork(sd, ad, hd, nh).to(self.device)
        self.q_tgt = DoubleQNetwork(sd, ad, hd, nh).to(self.device)
        self.actor = GaussianActor(sd, ad, hd, nh).to(self.device)
        self.vae   = BehaviorVAE(sd, ad, self.cfg.vae_hidden_dim, lat).to(self.device)

        self.q_tgt.load_state_dict(self.q_net.state_dict())

        # Lagrange multiplier log_λ (constrained to ≥ 0 via exp)
        self.log_lam = torch.tensor(
            [self.cfg.log_lambda_init], requires_grad=True, device=self.device)

        self.q_opt   = optim.Adam(self.q_net.parameters(), lr=self.cfg.lr)
        self.a_opt   = optim.Adam(self.actor.parameters(), lr=self.cfg.lr)
        self.vae_opt = optim.Adam(self.vae.parameters(),   lr=self.cfg.lr_vae)
        self.lam_opt = optim.Adam([self.log_lam],           lr=self.cfg.lr_lambda)

    # ── observation processing ────────────────────────────────────────────────

    def _obs_to_state(self, obs: Any) -> np.ndarray:
        if isinstance(obs, dict):
            o = np.asarray(obs["observation"],  dtype=np.float32).reshape(-1)
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
        return _t(obs), _t(dataset["act"]), _t(dataset["rew"]), _t(next_obs), _t(dataset["done"])

    # ── training ──────────────────────────────────────────────────────────────

    def _soft_update(self) -> None:
        for p, pt in zip(self.q_net.parameters(), self.q_tgt.parameters()):
            pt.data.copy_(self.cfg.tau * p.data + (1.0 - self.cfg.tau) * pt.data)

    def train(self, dataset: Dict[str, Any], n_epochs: int = 200) -> None:
        """Train BEAR on the offline dataset."""
        S, A, R, S2, D = self._dataset_to_tensors(dataset)
        N = S.shape[0]
        if N == 0:
            print("[BEAR] empty dataset — skipping.", flush=True)
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
        α_s = self.cfg.alpha_sac

        for epoch in range(int(n_epochs)):
            idx = rng.choice(N, size=bs, replace=(N < bs))
            s, a, r, s2, d = S[idx], A[idx], R[idx], S2[idx], D[idx]

            # ── VAE training (behavior policy) ───────────────────────────────
            a_recon, mu_z, std_z = self.vae(s, a)
            recon_loss = F.mse_loss(a_recon, a)
            kl_loss    = -0.5 * (1.0 + 2.0 * std_z.log()
                                  - mu_z.pow(2) - std_z.pow(2)).sum(-1).mean()
            vae_loss   = recon_loss + 0.5 * kl_loss
            self.vae_opt.zero_grad()
            vae_loss.backward()
            self.vae_opt.step()

            # ── Q update (SAC backup, no conservatism needed — MMD handles it) ─
            with torch.no_grad():
                a2, lp2 = self.actor.sample(s2)
                q2_min  = self.q_tgt.min(s2, a2)
                q_tgt   = r + self.cfg.gamma * (1.0 - d) * (q2_min - α_s * lp2)
            q1, q2   = self.q_net(s, a)
            q_loss   = F.mse_loss(q1, q_tgt) + F.mse_loss(q2, q_tgt)
            self.q_opt.zero_grad()
            q_loss.backward()
            self.q_opt.step()

            # ── Actor update with MMD constraint ─────────────────────────────
            # Standard SAC-style actor update: maximize Q(s, a_new) - λ·MMD
            # MMD constraint enforces π stays close to π_β (Kumar 2019, eq. 4).
            n_mmd = self.cfg.n_mmd_samples

            # MMD between current policy and VAE behavior policy
            a_pi, _  = self.actor.sample_n(s, n_mmd)            # [B, n, ad]
            with torch.no_grad():
                a_vae = self.vae.sample_n(s, n_mmd)             # [B, n, ad]

            mmd = _mmd(a_pi, a_vae,
                       kernel=self.cfg.kernel, bw=self.cfg.kernel_bw).mean()

            # Q-maximization: sample a single action per state from the actor
            # (standard SAC actor loss, differentiable through reparameterization)
            a_new, _ = self.actor.sample(s)                      # [B, ad]
            q_actor  = self.q_net.min(s, a_new)
            lam      = self.log_lam.exp().detach()
            a_loss   = -q_actor.mean() + lam * mmd

            self.a_opt.zero_grad()
            a_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.a_opt.step()

            # ── Dual variable update (λ ← max(0, λ + η·(MMD - ε))) ──────────
            lam_loss = -(self.log_lam.exp() * (mmd.detach() - self.cfg.delta_mmd))
            self.lam_opt.zero_grad()
            lam_loss.backward()
            self.lam_opt.step()
            # Clamp log_λ to avoid negative λ
            with torch.no_grad():
                self.log_lam.clamp_(min=-10.0, max=10.0)

            self._soft_update()

            self.loss_history["q"].append(float(q_loss))
            self.loss_history["vae"].append(float(vae_loss))
            self.loss_history["actor"].append(float(a_loss))

    # ── inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def act(self, obs: Any) -> np.ndarray:
        """
        BEAR deployment: sample n_vae_samples actions from the VAE,
        return the one with the highest Q-value (best-of-N selection,
        as described in the paper).
        """
        s = torch.as_tensor(self._obs_to_state(obs),
                            dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.state_mean is not None:
            s = (s - self.state_mean) / self.state_std
        n   = self.cfg.n_vae_samples
        a_s = self.vae.sample_n(s, n).squeeze(0)          # [n, act_dim]
        q_s = self.q_net.min(
            s.expand(n, -1), a_s
        )                                                  # [n]
        best = q_s.argmax()
        return np.clip(a_s[best].cpu().numpy(), -1.0, 1.0).astype(np.float32)

    def __call__(self, obs: Any) -> np.ndarray:
        return self.act(obs)
