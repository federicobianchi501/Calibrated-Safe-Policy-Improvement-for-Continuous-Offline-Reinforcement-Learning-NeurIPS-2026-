# =============================================================================
# Calibrated Safe Policy Improvement for Continuous Offline Reinforcement Learning
# Federico Bianchi, Alessandro Farinelli, Alberto Castellini
# NeurIPS 2026
# =============================================================================
"""
Numba forward passes for the dynamics ensemble and the PPO actor.

Weights of a fitted EnsembleDynamicsSB3 and an SB3 PPO policy are copied into
float32 numpy arrays; njit kernels compute per-member deltas, USAD and the
deterministic PPO action. Kernels are compiled with fastmath, so outputs
match torch up to float32 rounding. Training and serialization stay with the
torch modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# Weight containers
# ---------------------------------------------------------------------------


@dataclass
class EnsembleNumbaWeights:
    """Float32 weights and input normalization of a K-member MLP ensemble."""

    # K members. Each member is an MLP with `n_hidden` hidden layers:
    #   2 hidden: in -> H1 -> H2 -> obs_dim                  (3 Linear)
    #   3 hidden: in -> H1 -> H2 -> H3 -> obs_dim            (4 Linear)
    #   4 hidden: in -> H1 -> H2 -> H3 -> H4 -> obs_dim      (5 Linear)
    # W3/b3 are always the output layer (into obs_dim).
    # W_mid/b_mid carry the third hidden layer when n_hidden in {3,4}.
    # W_mid2/b_mid2 carry the fourth hidden layer when n_hidden==4.
    # Unused mid arrays are zero-shaped sentinels (K,0,0) / (K,0).
    W1: np.ndarray  # [K, H1, in_dim]
    b1: np.ndarray  # [K, H1]
    W2: np.ndarray  # [K, H2, H1]
    b2: np.ndarray  # [K, H2]
    W_mid: np.ndarray  # [K, H3, H2] when n_hidden in {3,4}
    b_mid: np.ndarray  # [K, H3]    when n_hidden in {3,4}
    W_mid2: np.ndarray  # [K, H4, H3] when n_hidden==4
    b_mid2: np.ndarray  # [K, H4]    when n_hidden==4
    W3: np.ndarray  # [K, obs_dim, H_last]
    b3: np.ndarray  # [K, obs_dim]
    input_mean: np.ndarray  # [in_dim] (zeros if no normalization)
    input_inv_std: np.ndarray  # [in_dim] (ones if no normalization)
    has_norm: bool
    obs_dim: int
    act_dim: int
    in_dim: int
    K: int
    n_hidden: int  # 2, 3, or 4


@dataclass
class PPOActorNumbaWeights:
    """Float32 weights of the SB3 PPO actor mean network."""

    # actor: Linear(obs_dim->H1) + Tanh + Linear(H1->H2) + Tanh + Linear(H2->act_dim)
    W1: np.ndarray  # [H1, obs_dim]
    b1: np.ndarray  # [H1]
    W2: np.ndarray  # [H2, H1]
    b2: np.ndarray  # [H2]
    W3: np.ndarray  # [act_dim, H2]
    b3: np.ndarray  # [act_dim]
    obs_dim: int
    act_dim: int


# ---------------------------------------------------------------------------
# Extractors (torch -> numpy, called once at planner init)
# ---------------------------------------------------------------------------


def extract_ensemble_weights(ens: Any) -> EnsembleNumbaWeights:
    """
    Read weights of an EnsembleDynamicsSB3 into flat numpy arrays.

    Supports
        2 hidden layers: in -> H1 -> H2 -> obs_dim                  (3 Linear)
        3 hidden layers: in -> H1 -> H2 -> H3 -> obs_dim            (4 Linear)
        4 hidden layers: in -> H1 -> H2 -> H3 -> H4 -> obs_dim      (5 Linear)
    with ReLU activations between Linears.
    """
    import torch.nn as nn

    members = list(getattr(ens, "ensemble"))
    K = len(members)
    if K == 0:
        raise ValueError("ensemble has no members")

    net0 = members[0].net
    linear_idx = [i for i, layer in enumerate(net0) if isinstance(layer, nn.Linear)]
    n_linear = len(linear_idx)
    n_hidden = n_linear - 1
    if n_hidden not in (2, 3, 4):
        raise ValueError(
            f"Numba backend supports MLP with 2, 3, or 4 hidden layers; got {n_linear} Linear layers."
        )

    # Identify per-depth layer indices
    in_idx = linear_idx[0]
    h1_idx = linear_idx[1]
    h2_idx = linear_idx[2] if n_hidden >= 3 else linear_idx[-1]
    h3_idx = linear_idx[3] if n_hidden == 4 else None
    out_idx = linear_idx[-1]

    in_dim = int(net0[in_idx].in_features)
    H1 = int(net0[in_idx].out_features)
    H2 = int(net0[h1_idx].out_features)
    H3 = int(net0[h2_idx].out_features) if n_hidden >= 3 else 0
    H4 = int(net0[h3_idx].out_features) if n_hidden == 4 else 0
    obs_dim = int(net0[out_idx].out_features)

    W1 = np.zeros((K, H1, in_dim), dtype=np.float32)
    b1 = np.zeros((K, H1), dtype=np.float32)
    W2 = np.zeros((K, H2, H1), dtype=np.float32)
    b2 = np.zeros((K, H2), dtype=np.float32)
    if n_hidden == 2:
        H_last = H2
    elif n_hidden == 3:
        H_last = H3
    else:
        H_last = H4
    W3 = np.zeros((K, obs_dim, H_last), dtype=np.float32)
    b3 = np.zeros((K, obs_dim), dtype=np.float32)
    if n_hidden >= 3:
        W_mid = np.zeros((K, H3, H2), dtype=np.float32)
        b_mid = np.zeros((K, H3), dtype=np.float32)
    else:
        W_mid = np.zeros((K, 0, 0), dtype=np.float32)
        b_mid = np.zeros((K, 0), dtype=np.float32)
    if n_hidden == 4:
        W_mid2 = np.zeros((K, H4, H3), dtype=np.float32)
        b_mid2 = np.zeros((K, H4), dtype=np.float32)
    else:
        W_mid2 = np.zeros((K, 0, 0), dtype=np.float32)
        b_mid2 = np.zeros((K, 0), dtype=np.float32)

    for k, m in enumerate(members):
        net = m.net
        W1[k] = net[in_idx].weight.detach().cpu().numpy().astype(np.float32)
        b1[k] = net[in_idx].bias.detach().cpu().numpy().astype(np.float32)
        W2[k] = net[h1_idx].weight.detach().cpu().numpy().astype(np.float32)
        b2[k] = net[h1_idx].bias.detach().cpu().numpy().astype(np.float32)
        if n_hidden >= 3:
            W_mid[k] = net[h2_idx].weight.detach().cpu().numpy().astype(np.float32)
            b_mid[k] = net[h2_idx].bias.detach().cpu().numpy().astype(np.float32)
        if n_hidden == 4:
            W_mid2[k] = net[h3_idx].weight.detach().cpu().numpy().astype(np.float32)
            b_mid2[k] = net[h3_idx].bias.detach().cpu().numpy().astype(np.float32)
        W3[k] = net[out_idx].weight.detach().cpu().numpy().astype(np.float32)
        b3[k] = net[out_idx].bias.detach().cpu().numpy().astype(np.float32)

    in_mean_raw = getattr(ens, "input_mean", None)
    in_std_raw = getattr(ens, "input_std", None)
    if in_mean_raw is not None and in_std_raw is not None:
        input_mean = np.asarray(in_mean_raw, dtype=np.float32).reshape(-1)
        input_std = np.asarray(in_std_raw, dtype=np.float32).reshape(-1)
        input_inv_std = (1.0 / np.maximum(input_std, 1e-8)).astype(np.float32)
        has_norm = True
    else:
        input_mean = np.zeros((in_dim,), dtype=np.float32)
        input_inv_std = np.ones((in_dim,), dtype=np.float32)
        has_norm = False

    act_dim = in_dim - obs_dim
    return EnsembleNumbaWeights(
        W1=W1, b1=b1, W2=W2, b2=b2,
        W_mid=W_mid, b_mid=b_mid,
        W_mid2=W_mid2, b_mid2=b_mid2,
        W3=W3, b3=b3,
        input_mean=input_mean, input_inv_std=input_inv_std,
        has_norm=has_norm, obs_dim=obs_dim, act_dim=act_dim, in_dim=in_dim, K=K,
        n_hidden=n_hidden,
    )


def extract_ppo_actor_weights(ppo: Any) -> PPOActorNumbaWeights:
    """
    Read the deterministic mean head of an SB3 PPO ActorCriticPolicy.

    Expected structure:
        policy.mlp_extractor.policy_net[0] = Linear(obs_dim, H1)
        policy.mlp_extractor.policy_net[1] = Tanh
        policy.mlp_extractor.policy_net[2] = Linear(H1, H2)
        policy.mlp_extractor.policy_net[3] = Tanh
        policy.action_net               = Linear(H2, act_dim)
    """
    pnet = ppo.policy.mlp_extractor.policy_net
    lin1 = pnet[0]
    lin2 = pnet[2]
    action_net = ppo.policy.action_net

    W1 = lin1.weight.detach().cpu().numpy().astype(np.float32)
    b1 = lin1.bias.detach().cpu().numpy().astype(np.float32)
    W2 = lin2.weight.detach().cpu().numpy().astype(np.float32)
    b2 = lin2.bias.detach().cpu().numpy().astype(np.float32)
    W3 = action_net.weight.detach().cpu().numpy().astype(np.float32)
    b3 = action_net.bias.detach().cpu().numpy().astype(np.float32)

    obs_dim = int(lin1.in_features)
    act_dim = int(action_net.out_features)

    return PPOActorNumbaWeights(
        W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3,
        obs_dim=obs_dim, act_dim=act_dim,
    )


# ---------------------------------------------------------------------------
# njit kernels
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=True, inline="always")
def _linear_relu(x, W, b, out):
    # out = relu(W @ x + b);  W: [H, D], x: [D], b: [H], out: [H]
    H = W.shape[0]
    D = W.shape[1]
    for i in range(H):
        s = b[i]
        for j in range(D):
            s += W[i, j] * x[j]
        out[i] = s if s > 0.0 else 0.0


@njit(cache=True, fastmath=True, inline="always")
def _linear_tanh(x, W, b, out):
    # out = tanh(W @ x + b)
    H = W.shape[0]
    D = W.shape[1]
    for i in range(H):
        s = b[i]
        for j in range(D):
            s += W[i, j] * x[j]
        out[i] = np.tanh(s)


@njit(cache=True, fastmath=True, inline="always")
def _linear(x, W, b, out):
    # out = W @ x + b
    H = W.shape[0]
    D = W.shape[1]
    for i in range(H):
        s = b[i]
        for j in range(D):
            s += W[i, j] * x[j]
        out[i] = s


@njit(cache=True, fastmath=True)
def ensemble_forward_all(
    obs, action,
    W1, b1, W2, b2, W3, b3,
    input_mean, input_inv_std, has_norm,
    deltas_out,
    _h1_buf, _h2_buf, _x_buf,
):
    """
    Per-member delta predictions of a 2-hidden-layer ensemble.

    Writes deltas_out [K, obs_dim]. Input is (obs, action), normalized when
    `has_norm`. Scratch buffers (_h1_buf, _h2_buf, _x_buf) are caller-owned.
    """
    K = W1.shape[0]
    in_dim = W1.shape[2]
    obs_dim = W3.shape[1]
    ds = obs.shape[0]

    # pack input (obs, action)
    for j in range(ds):
        _x_buf[j] = obs[j]
    for j in range(in_dim - ds):
        _x_buf[ds + j] = action[j]

    if has_norm:
        for j in range(in_dim):
            _x_buf[j] = (_x_buf[j] - input_mean[j]) * input_inv_std[j]

    for k in range(K):
        _linear_relu(_x_buf, W1[k], b1[k], _h1_buf)
        _linear_relu(_h1_buf, W2[k], b2[k], _h2_buf)
        # final linear into deltas_out[k]
        for i in range(obs_dim):
            s = b3[k, i]
            for j in range(_h2_buf.shape[0]):
                s += W3[k, i, j] * _h2_buf[j]
            deltas_out[k, i] = s


@njit(cache=True, fastmath=True)
def ensemble_forward_all_3h(
    obs, action,
    W1, b1, W2, b2, W_mid, b_mid, W3, b3,
    input_mean, input_inv_std, has_norm,
    deltas_out,
    _h1_buf, _h2_buf, _h3_buf, _x_buf,
):
    """
    `ensemble_forward_all` for 3-hidden-layer MLPs.

    x -> W1 -> ReLU -> W2 -> ReLU -> W_mid -> ReLU -> W3 -> deltas.
    """
    K = W1.shape[0]
    in_dim = W1.shape[2]
    obs_dim = W3.shape[1]
    ds = obs.shape[0]

    for j in range(ds):
        _x_buf[j] = obs[j]
    for j in range(in_dim - ds):
        _x_buf[ds + j] = action[j]

    if has_norm:
        for j in range(in_dim):
            _x_buf[j] = (_x_buf[j] - input_mean[j]) * input_inv_std[j]

    for k in range(K):
        _linear_relu(_x_buf, W1[k], b1[k], _h1_buf)
        _linear_relu(_h1_buf, W2[k], b2[k], _h2_buf)
        _linear_relu(_h2_buf, W_mid[k], b_mid[k], _h3_buf)
        for i in range(obs_dim):
            s = b3[k, i]
            for j in range(_h3_buf.shape[0]):
                s += W3[k, i, j] * _h3_buf[j]
            deltas_out[k, i] = s


@njit(cache=True, fastmath=True)
def ensemble_forward_all_4h(
    obs, action,
    W1, b1, W2, b2, W_mid, b_mid, W_mid2, b_mid2, W3, b3,
    input_mean, input_inv_std, has_norm,
    deltas_out,
    _h1_buf, _h2_buf, _h3_buf, _h4_buf, _x_buf,
):
    """
    `ensemble_forward_all` for 4-hidden-layer MLPs.

    x -> W1 -> ReLU -> W2 -> ReLU -> W_mid -> ReLU -> W_mid2 -> ReLU -> W3 -> deltas.
    """
    K = W1.shape[0]
    in_dim = W1.shape[2]
    obs_dim = W3.shape[1]
    ds = obs.shape[0]

    for j in range(ds):
        _x_buf[j] = obs[j]
    for j in range(in_dim - ds):
        _x_buf[ds + j] = action[j]

    if has_norm:
        for j in range(in_dim):
            _x_buf[j] = (_x_buf[j] - input_mean[j]) * input_inv_std[j]

    for k in range(K):
        _linear_relu(_x_buf, W1[k], b1[k], _h1_buf)
        _linear_relu(_h1_buf, W2[k], b2[k], _h2_buf)
        _linear_relu(_h2_buf, W_mid[k], b_mid[k], _h3_buf)
        _linear_relu(_h3_buf, W_mid2[k], b_mid2[k], _h4_buf)
        for i in range(obs_dim):
            s = b3[k, i]
            for j in range(_h4_buf.shape[0]):
                s += W3[k, i, j] * _h4_buf[j]
            deltas_out[k, i] = s


@njit(cache=True, fastmath=True)
def usad_from_deltas(deltas):
    """max_{i<j} ||deltas[i] - deltas[j]||_2. deltas: [K, D]."""
    K = deltas.shape[0]
    D = deltas.shape[1]
    maxd = 0.0
    for i in range(K):
        for j in range(i + 1, K):
            s = 0.0
            for d in range(D):
                x = deltas[i, d] - deltas[j, d]
                s += x * x
            if s > maxd:
                maxd = s
    return np.sqrt(maxd)


@njit(cache=True, fastmath=True)
def mean_delta_from_deltas(deltas, out):
    """out = mean_k deltas[k]. deltas: [K, D], out: [D]."""
    K = deltas.shape[0]
    D = deltas.shape[1]
    inv = 1.0 / float(K)
    for d in range(D):
        s = 0.0
        for k in range(K):
            s += deltas[k, d]
        out[d] = s * inv


@njit(cache=True, fastmath=True)
def ppo_actor_mean(
    obs,
    W1, b1, W2, b2, W3, b3,
    out,
    _h1_buf, _h2_buf,
):
    """
    PPO actor mean (Tanh activations), written into `out`.

    Clipped to [-1, 1], as `PPO.predict(..., deterministic=True)` clips to the
    action space bounds.
    """
    _linear_tanh(obs, W1, b1, _h1_buf)
    _linear_tanh(_h1_buf, W2, b2, _h2_buf)
    _linear(_h2_buf, W3, b3, out)
    for i in range(out.shape[0]):
        v = out[i]
        if v < -1.0:
            out[i] = -1.0
        elif v > 1.0:
            out[i] = 1.0


# ---------------------------------------------------------------------------
# Python-side convenience wrappers (one-off queries, not in the hot loop)
# ---------------------------------------------------------------------------


class EnsembleNumbaProxy:
    """
    Ensemble interface used by mcts_parametric (_predict_all, uncertainty_usad,
    sample_next_obs_vec) backed by the 2-hidden-layer numba kernel.

    For one-off queries; the numba planners call the kernels directly.
    """

    def __init__(self, w: EnsembleNumbaWeights):
        self.w = w
        self._h1 = np.zeros((w.W1.shape[1],), dtype=np.float32)
        self._h2 = np.zeros((w.W2.shape[1],), dtype=np.float32)
        self._x = np.zeros((w.in_dim,), dtype=np.float32)
        self._deltas = np.zeros((w.K, w.obs_dim), dtype=np.float32)
        self._mean_delta = np.zeros((w.obs_dim,), dtype=np.float32)

    def _predict_all(self, obs, action) -> np.ndarray:
        """Per-member deltas [K, obs_dim] (a copy)."""
        obs_np = np.asarray(obs, dtype=np.float32).reshape(-1)
        act_np = np.asarray(action, dtype=np.float32).reshape(-1)
        ensemble_forward_all(
            obs_np, act_np,
            self.w.W1, self.w.b1, self.w.W2, self.w.b2, self.w.W3, self.w.b3,
            self.w.input_mean, self.w.input_inv_std, self.w.has_norm,
            self._deltas, self._h1, self._h2, self._x,
        )
        return self._deltas.copy()

    def uncertainty_usad(self, obs, action) -> float:
        """USAD disagreement at (obs, action)."""
        deltas = self._predict_all(obs, action)
        return float(usad_from_deltas(deltas))

    def sample_next_obs_vec(self, obs, action, *, model_index: Optional[int] = None,
                            rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Next obs from member `model_index` (-1: ensemble mean; None: random)."""
        obs_np = np.asarray(obs, dtype=np.float32).reshape(-1)
        deltas = self._predict_all(obs_np, action)
        K = deltas.shape[0]
        if model_index == -1:
            delta = deltas.mean(axis=0)
        else:
            mi = model_index
            if mi is None:
                if rng is None:
                    rng = np.random.default_rng()
                mi = int(rng.integers(0, K))
            mi = int(np.clip(mi, 0, K - 1))
            delta = deltas[mi]
        return (obs_np + delta).astype(np.float32)
