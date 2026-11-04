#!/usr/bin/env python3
# =============================================================================
# Calibrated Safe Policy Improvement for Continuous Offline Reinforcement Learning
# Federico Bianchi, Alessandro Farinelli, Alberto Castellini
# NeurIPS 2026
# =============================================================================
"""
Cal-SPI benchmark on the MuJoCo locomotion domains (Hopper, HalfCheetah, Walker2d, Ant).

One run covers a (domain, dataset size, seed) cell: dataset collection with the PPO
baseline, trajectory-level train/calibration split, ensemble training, calibrator
fitting, planner construction (backends original, batched, numba, numba_scaled, array),
offline RL baselines, and episodic evaluation with per-episode checkpoints.

Observations are flat arrays; the ensemble sees them as {"observation": obs}. An episode
counts as a success when it is not terminated early.

Methods: ppo_only, mcts_only_no_gate, calibrated_usad, calibrated_variance, iql, td3bc,
morel_full, mopo_mse, cql, bear, cal_ql.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from safe_rl.envs.continuous_env_factory import DOMAIN_CONFIGS, make_continuous_env
from safe_rl.offline.ensemble_dynamics_sb3 import EnsembleDynamicsSB3, EnsembleTrainConfig
from safe_rl.offline.model_env_continuous import ModelContinuousEnv, ModelContinuousEnvConfig


def _make_model_env(ens, model_cfg: ModelContinuousEnvConfig):
    """Build the learned-dynamics environment used by the planners."""
    return ModelContinuousEnv(ens, model_cfg)
from safe_rl.offline.uncertainty_calibration import (
    UncertaintyCalibrator,
    UncertaintyCalibratorConfig,
)
from safe_rl.algorithms.offline.cal_spi.mcts_parametric import (
    OfflineMCTSConfig,
    OfflineMCTS_DPW_PPO_SPIBB_Parametric,
    OfflineMCTS_DPW_Ensemble_UNSAFE,
)
from safe_rl.algorithms.offline.cal_spi.mcts_batched import (
    OfflineMCTS_DPW_PPO_Batched,
)
from safe_rl.algorithms.offline.iql import IQLAgent, IQLConfig
from safe_rl.algorithms.offline.td3bc import TD3BCAgent, TD3BCConfig
from safe_rl.algorithms.offline.morel_full import MORelFullAgent, MORelFullConfig
from safe_rl.algorithms.offline.mopo import MOPOMSEAgent, MOPOMSEConfig
from safe_rl.algorithms.offline.cql import CQLAgent, CQLConfig
from safe_rl.algorithms.offline.bear import BEARAgent, BEARConfig
from safe_rl.algorithms.offline.cal_ql import CalQLAgent, CalQLConfig


# ---------------------------------------------------------------------------
# Domain-specific baseline hyperparameters
# ---------------------------------------------------------------------------
# Values follow the locomotion settings of the original papers:
#   IQL    (Kostrikov et al., 2021): alpha=0.7, beta=3.0, 3 hidden layers
#   TD3+BC (Fujimoto & Gu, 2021):    alpha_bc=2.5, 2 hidden layers
#   MOReL  (Kidambi et al., 2020):   tau = 95th percentile of USAD on D_train, kappa=50
#   MOPO   (Yu et al., 2020):        rollout_length 5 (1 on walker2d), lam=1.0

def make_iql_cfg(domain: str, seed: int) -> IQLConfig:
    """IQL config with the locomotion hyperparameters."""
    return IQLConfig(
        hidden_dim=256,
        n_hidden=3,
        lr=3e-4,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        beta=3.0,             # 10.0 in maze navigation
        alpha=0.7,            # 0.9 in maze navigation
        clip_score=100.0,
        normalize_states=True,
        seed=seed,
    )


def make_td3bc_cfg(domain: str, seed: int) -> TD3BCConfig:
    """TD3+BC config; the same hyperparameters on every locomotion domain."""
    return TD3BCConfig(
        hidden_dim=256,
        n_hidden=2,
        lr=3e-4,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        alpha_bc=2.5,
        normalize_states=True,
        seed=seed,
    )


def make_morel_cfg(domain: str, seed: int, rollout_batch: int) -> MORelFullConfig:
    """MOReL config.

    tau is the 95th percentile of USAD on D_train; transitions above it enter the HALT
    state with reward -kappa and done=True. rollout_length=5 is shorter than in the
    original method (100-1000) because the policy is trained with SAC bootstrapping.
    """
    return MORelFullConfig(
        tau=0.2,              # used if the USAD percentile cannot be computed
        tau_percentile=95.0,
        kappa=50.0,
        rollout_length=5,
        rollout_batch=rollout_batch,
        hidden_dim=256,
        n_hidden=2,
        lr=3e-4,
        batch_size=256,
        gamma=0.99,
        tau_soft=0.005,
        sac_alpha=0.2,
        seed=seed,
    )


def make_mopo_mse_cfg(domain: str, seed: int, rollout_batch: int) -> MOPOMSEConfig:
    """MOPO config on the MSE ensemble; rollout_length 1 on walker2d, 5 elsewhere, lam=1.0."""
    rollout_length = 1 if domain == "walker2d" else 5
    return MOPOMSEConfig(
        lam=1.0,
        rollout_length=rollout_length,
        rollout_batch=rollout_batch,
        hidden_dim=256,
        n_hidden=2,
        lr=3e-4,
        batch_size=256,
        gamma=0.99,
        tau_soft=0.005,
        sac_alpha=0.2,
        seed=seed,
    )


def make_cql_cfg(domain: str, seed: int) -> CQLConfig:
    """CQL config with the locomotion hyperparameters (Kumar et al., 2020)."""
    return CQLConfig(
        hidden_dim=256,
        n_hidden=3,
        lr=3e-4,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        alpha_sac=0.05,
        alpha_cql=5.0,
        n_action_samples=10,
        normalize_states=True,
        seed=seed,
    )


def make_bear_cfg(domain: str, seed: int) -> BEARConfig:
    """BEAR config with the locomotion hyperparameters (Kumar et al., 2019)."""
    return BEARConfig(
        hidden_dim=256,
        n_hidden=3,
        vae_hidden_dim=750,
        vae_latent_dim=0,  # set to 2*act_dim in agent __init__
        lr=3e-4,
        lr_vae=3e-4,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        alpha_sac=0.05,
        delta_mmd=0.05,
        kernel="laplacian",
        kernel_bw=20.0,
        n_mmd_samples=4,
        n_vae_samples=10,
        normalize_states=True,
        seed=seed,
    )


def make_cal_ql_cfg(domain: str, seed: int) -> CalQLConfig:
    """Cal-QL config (Nakamoto et al., 2023) on top of the CQL settings above."""
    return CalQLConfig(
        hidden_dim=256,
        n_hidden=3,
        lr=3e-4,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        alpha_sac=0.05,
        alpha_cql=5.0,
        n_action_samples=10,
        normalize_states=True,
        k_mc=5,
        lambda_cal=1.0,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def ensure_dir(p: str) -> None:
    """Create directory p if it does not exist."""
    os.makedirs(p, exist_ok=True)


def make_morel_reward_fn(domain: str):
    """Return a batched reward_fn(obs, act, next_obs) -> (rew, done) for MOReL rollouts.

    Uses the same reward and health check as ModelContinuousEnv.
    """
    dc = DOMAIN_CONFIGS[domain]
    v_idx   = int(dc["vel_obs_idx"])
    z_idx   = int(dc["z_obs_idx"])
    ang_idx = int(dc["angle_obs_idx"])
    z_lo, z_hi = dc["healthy_z_range"]
    healthy_r  = float(dc["healthy_reward"])
    ctrl_cost  = float(dc["ctrl_cost_weight"])
    healthy_ang = float(dc["healthy_angle"])

    def reward_fn(obs, act, next_obs):
        x_vel    = next_obs[:, v_idx]
        ctrl     = ctrl_cost * np.sum(act ** 2, axis=-1)
        z        = next_obs[:, z_idx]
        ang      = next_obs[:, ang_idx]
        healthy  = (z >= z_lo) & (z <= z_hi) & (np.abs(ang) <= healthy_ang)
        rew  = (x_vel + healthy_r * healthy - ctrl).astype(np.float32)
        done = (~healthy).astype(np.float32)
        return rew, done

    return reward_fn


def _jsonable(x):
    """Convert dataclasses and numpy values to JSON-serializable objects."""
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return {"__ndarray__": True, "shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, (set, tuple)):
        return list(x)
    return x


def sha256_file(path: str) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------
# Layout inside the output directory:
#   ep_ckpt/<method>.jsonl  one line per completed episode (append-only)
#   results_partial.jsonl   one line per completed method (append-only)
#   models/<method>.pt      trained baseline weights
#
# On re-run, completed methods are skipped, partially evaluated methods resume
# from the last saved episode, and trained baselines are loaded instead of retrained.

_PARTIAL_FILE      = "results_partial.jsonl"
_EP_CKPT_DIR       = "ep_ckpt"
_MODELS_DIR        = "models"
_SYNC_INTERVAL_SEC = 10 * 3600   # minimum interval between explicit fsync calls


class _EpisodeCheckpoint:
    """Per-episode JSONL checkpoint for one method's evaluation.

    Each line holds one completed episode; a truncated last line is ignored on load.
    Writes are flushed to the OS on every append, and os.fsync is called at most once
    every _SYNC_INTERVAL_SEC seconds.
    """

    def __init__(self, outdir: str, method: str):
        import time as _time
        ep_dir = os.path.join(outdir, _EP_CKPT_DIR)
        os.makedirs(ep_dir, exist_ok=True)
        self.path = os.path.join(ep_dir, f"{method}.jsonl")
        self.episodes: List[Dict] = []
        self._last_sync: float = _time.monotonic()
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self.episodes.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # ignore truncated last line

    def n_done(self) -> int:
        """Number of completed episodes."""
        return len(self.episodes)

    def append(self, ep_data: Dict) -> None:
        """Record one completed episode in memory and on disk."""
        import time as _time
        self.episodes.append(ep_data)
        with open(self.path, "a") as f:
            f.write(json.dumps(ep_data) + "\n")
            f.flush()
            now = _time.monotonic()
            if now - self._last_sync >= _SYNC_INTERVAL_SEC:
                os.fsync(f.fileno())
                self._last_sync = now

    def to_metrics(self) -> Dict[str, Any]:
        """Aggregate per-episode data into the same metrics dict as _eval_policy_episodic."""
        if not self.episodes:
            return {}
        R    = np.array([e["return"]        for e in self.episodes], dtype=np.float64)
        RD   = np.array([e["return_disc"]   for e in self.episodes], dtype=np.float64)
        L    = np.array([e["ep_len"]        for e in self.episodes], dtype=np.float64)
        S    = np.array([e["survived"]      for e in self.episodes], dtype=np.float64)
        XVEL = np.array([e["xvel"]          for e in self.episodes], dtype=np.float64)
        FB   = np.array([e["fallback_frac"] for e in self.episodes], dtype=np.float64)
        COV  = np.array([e["coverage"]      for e in self.episodes], dtype=np.float64)
        return {
            "return_mean":            float(R.mean()),
            "return_std":             float(R.std()),
            "return_median":          float(np.median(R)),
            "return_min":             float(R.min()),
            "return_max":             float(R.max()),
            "cvar_15":                float(cvar_lower_tail(R, 0.15)),
            "discounted_return_mean": float(RD.mean()),
            "ep_len_mean":            float(L.mean()),
            "success_rate":           float(S.mean()),
            "xvel_mean":              float(XVEL.mean()),
            "root_fallback_rate":     float(FB.mean()),
            "root_coverage_mean":     float(COV.mean()),
        }


def _load_completed_methods(outdir: str) -> Dict[str, Dict]:
    """Load completed method rows from partial checkpoint. Returns {method_name: row}."""
    path = os.path.join(outdir, _PARTIAL_FILE)
    if not os.path.exists(path):
        return {}
    completed: Dict[str, Dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    row = json.loads(line)
                    completed[row["method"]] = row
                except (json.JSONDecodeError, KeyError):
                    pass
    return completed


def _save_partial_result(outdir: str, row: Dict) -> None:
    """Append a completed method row to results_partial.jsonl and fsync it."""
    path = os.path.join(outdir, _PARTIAL_FILE)
    serialized = {}
    for k, v in row.items():
        if isinstance(v, (np.integer, np.floating)):
            serialized[k] = v.item()
        elif isinstance(v, np.ndarray):
            serialized[k] = v.tolist()
        else:
            serialized[k] = v
    with open(path, "a") as f:
        f.write(json.dumps(serialized) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _save_baseline_model(outdir: str, method: str, agent) -> None:
    """Save every nn.Module state_dict of a baseline agent plus its state normalization stats."""
    import torch.nn as nn
    models_dir = os.path.join(outdir, _MODELS_DIR)
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"{method}.pt")
    state: Dict = {}
    for name, val in vars(agent).items():
        if isinstance(val, nn.Module):
            state[name] = val.state_dict()
    # State normalization stats (IQL, TD3+BC, CQL and derived agents).
    for attr in ("state_mean", "state_std"):
        v = getattr(agent, attr, None)
        if v is not None:
            state[f"_norm_{attr}"] = v
    torch.save(state, path)
    print(f"[CKPT] Saved {method} model → {path}", flush=True)


def _load_baseline_model(outdir: str, method: str, agent) -> bool:
    """Load saved state_dicts into agent. Returns True if a checkpoint was found."""
    import torch.nn as nn
    path = os.path.join(outdir, _MODELS_DIR, f"{method}.pt")
    if not os.path.exists(path):
        return False
    state = torch.load(path, map_location="cpu")
    for name, val in vars(agent).items():
        if isinstance(val, nn.Module) and name in state:
            val.load_state_dict(state[name])
    for attr in ("state_mean", "state_std"):
        key = f"_norm_{attr}"
        if key in state:
            setattr(agent, attr, state[key])
    print(f"[CKPT] Loaded {method} model from {path}", flush=True)
    return True


def cvar_lower_tail(x: np.ndarray, alpha: float = 0.15) -> float:
    """Mean of the values at or below the alpha-quantile (lower-tail CVaR)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    q = float(np.quantile(x, alpha))
    m = x <= q
    return float(np.mean(x[m])) if np.any(m) else q


def _discounted_add(ret_d: float, r: float, gamma: float, t: int) -> float:
    """Return ret_d + gamma**t * r."""
    return float(ret_d + (float(gamma) ** int(t)) * float(r))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

# State for fork-based parallel evaluation. Set in the parent before forking;
# workers inherit it via copy-on-write.
_PARENT_ENV_FACTORY = None
_PARENT_ACT_FN = None
_PARENT_CKPT_PATH = None  # JSONL file that workers append episodes to
_PARENT_BASE_SEED = 0
_WORKER_ENV = None


def _worker_init():
    """Pool initializer: each child builds its own env via the inherited factory."""
    global _WORKER_ENV
    _WORKER_ENV = _PARENT_ENV_FACTORY()


def _worker_append_progress(ep_data: Dict[str, Any]) -> None:
    """Append a completed episode line to the shared JSONL file.

    Each line is far below PIPE_BUF (4096 bytes), so concurrent append-mode writes
    from several workers do not interleave on Linux.
    """
    if _PARENT_CKPT_PATH is None:
        return
    line = json.dumps(ep_data) + "\n"
    with open(_PARENT_CKPT_PATH, "a") as f:
        f.write(line)
        f.flush()


def _reseed_planner(act_fn, base_seed: int, ep_index: int) -> None:
    """Reseed the planner's generator from (base_seed, ep_index).

    Each episode gets an independent action-sampling stream that is the same in
    sequential and fork-parallel evaluation and reproducible on re-run and resume.
    Objects without an `rng` attribute are left untouched.
    """
    obj = getattr(act_fn, "__self__", act_fn)
    if hasattr(obj, "rng"):
        obj.rng = np.random.default_rng([int(base_seed), int(ep_index)])


def _worker_run_episode(args):
    """Run a single episode in a forked worker. Returns an ep_data dict."""
    actual_ei, ep_seed, max_steps, gamma = args
    env = _WORKER_ENV
    act_fn = _PARENT_ACT_FN
    _reseed_planner(act_fn, _PARENT_BASE_SEED, actual_ei)
    obs, _ = env.reset(seed=int(ep_seed))
    ep_ret = 0.0
    ep_ret_d = 0.0
    ep_len = 0
    ep_survived = True
    ep_xvel_sum = 0.0
    fb_count = 0
    cov_sum = 0.0
    cov_count = 0
    for t in range(int(max_steps)):
        a = act_fn(obs)
        _act_obj = getattr(act_fn, "__self__", act_fn)
        fb_count += int(bool(getattr(_act_obj, "last_root_fallback", False)))
        cov_sum += float(getattr(_act_obj, "last_root_coverage", 0.0))
        cov_count += 1
        obs, r, terminated, truncated, info = env.step(a)
        ep_ret += float(r)
        ep_ret_d = _discounted_add(ep_ret_d, float(r), float(gamma), t)
        ep_len = t + 1
        if isinstance(info, dict):
            ep_xvel_sum += float(info.get("x_velocity", 0.0))
        if terminated:
            ep_survived = False
        if terminated or truncated:
            break
    ep_data = {
        "ep":            actual_ei,
        "return":        ep_ret,
        "return_disc":   ep_ret_d,
        "ep_len":        ep_len,
        "survived":      1.0 if ep_survived else 0.0,
        "xvel":          ep_xvel_sum / max(1, ep_len),
        "fallback_frac": fb_count / max(1, cov_count),
        "coverage":      cov_sum / max(1, cov_count),
    }
    # Each worker appends its episode as soon as it finishes, so the line count
    # of the JSONL file is the number of completed episodes.
    _worker_append_progress(ep_data)
    return ep_data


def _eval_policy_episodic(
    env,
    act_fn,
    n_episodes: int,
    seed: int,
    max_steps: int,
    gamma: float,
    ckpt: Optional["_EpisodeCheckpoint"] = None,
    env_factory: Optional[Callable[[], Any]] = None,
    n_workers: int = 1,
    seed_stride: int = 1,
) -> Dict[str, Any]:
    """Evaluate a policy for n_episodes and return aggregated metrics.

    The environment seed of episode i is `seed * seed_stride + i`. With stride 1,
    cells with adjacent seeds share most initial states; a stride >= n_episodes gives
    every cell a disjoint range. All methods of a cell see the same initial states,
    so comparisons within a cell are paired.

    Success means the episode was not terminated early. Fallback rate and coverage
    are read from the planner's `last_root_fallback` / `last_root_coverage` attributes.

    With `ckpt`, episodes are checkpointed one by one and a re-run resumes after the
    last completed episode. With `n_workers > 1` and an `env_factory`, episodes run in
    a fork-based process pool, one environment per worker.
    """
    n_already = ckpt.n_done() if ckpt is not None else 0
    n_todo    = max(0, int(n_episodes) - n_already)

    if n_already > 0:
        print(f"[CKPT] Resuming from episode {n_already}/{n_episodes}", flush=True)

    if n_todo == 0 and ckpt is not None:
        return ckpt.to_metrics()

    # ── Parallel branch: fork-based pool, one env per worker, act_fn via copy-on-write ──
    if n_workers > 1 and env_factory is not None and n_todo > 1:
        import multiprocessing as _mp
        global _PARENT_ENV_FACTORY, _PARENT_ACT_FN, _PARENT_CKPT_PATH, _PARENT_BASE_SEED
        _PARENT_ENV_FACTORY = env_factory
        _PARENT_ACT_FN = act_fn
        _PARENT_BASE_SEED = int(seed)
        _PARENT_CKPT_PATH = ckpt.path if ckpt is not None else None
        tasks = [
            (n_already + ei, int(seed * seed_stride + n_already + ei),
             int(max_steps), float(gamma))
            for ei in range(n_todo)
        ]
        n_workers_eff = min(int(n_workers), len(tasks))
        print(f"[EVAL] Parallel eval: {len(tasks)} episodes across {n_workers_eff} workers", flush=True)
        ctx = _mp.get_context("fork")
        with ctx.Pool(processes=n_workers_eff, initializer=_worker_init) as pool:
            results = pool.map(_worker_run_episode, tasks)
        # Clear the globals so later forks do not inherit this call's state.
        _PARENT_ENV_FACTORY = None
        _PARENT_ACT_FN = None
        _PARENT_CKPT_PATH = None
        if ckpt is not None:
            # Workers wrote the episodes to disk; reload instead of appending again.
            ckpt.episodes = []
            ckpt._load()
            return ckpt.to_metrics()
        R    = np.asarray([d["return"]        for d in results], dtype=np.float64)
        RD   = np.asarray([d["return_disc"]   for d in results], dtype=np.float64)
        L    = np.asarray([d["ep_len"]        for d in results], dtype=np.float64)
        S    = np.asarray([d["survived"]      for d in results], dtype=np.float64)
        XVEL = np.asarray([d["xvel"]          for d in results], dtype=np.float64)
        FB   = np.asarray([d["fallback_frac"] for d in results], dtype=np.float64)
        COV  = np.asarray([d["coverage"]      for d in results], dtype=np.float64)
        return {
            "return_mean":            float(R.mean()),
            "return_std":             float(R.std()),
            "return_median":          float(np.median(R)),
            "return_min":             float(R.min()),
            "return_max":             float(R.max()),
            "cvar_15":                float(cvar_lower_tail(R, 0.15)),
            "discounted_return_mean": float(RD.mean()),
            "ep_len_mean":            float(L.mean()),
            "success_rate":           float(S.mean()),
            "xvel_mean":              float(XVEL.mean()),
            "root_fallback_rate":     float(FB.mean()),
            "root_coverage_mean":     float(COV.mean()),
        }

    R, RD, L, S = [], [], [], []
    FB, COV = [], []
    XVEL = []

    for ei in range(n_todo):
        actual_ei = n_already + ei  # global episode index, stable across resume
        _reseed_planner(act_fn, seed, actual_ei)
        obs, _ = env.reset(seed=int(seed * seed_stride + actual_ei))
        ep_ret = 0.0
        ep_ret_d = 0.0
        ep_len = 0
        ep_survived = True
        ep_xvel_sum = 0.0
        fb_count = 0
        cov_sum  = 0.0
        cov_count = 0

        for t in range(int(max_steps)):
            a = act_fn(obs)
            # For a bound method the planner attributes live on __self__.
            _act_obj = getattr(act_fn, "__self__", act_fn)
            fb_count  += int(bool(getattr(_act_obj, "last_root_fallback",  False)))
            cov_sum   += float(getattr(_act_obj, "last_root_coverage", 0.0))
            cov_count += 1

            obs, r, terminated, truncated, info = env.step(a)
            ep_ret   += float(r)
            ep_ret_d  = _discounted_add(ep_ret_d, float(r), float(gamma), t)
            ep_len    = t + 1

            if isinstance(info, dict):
                ep_xvel_sum += float(info.get("x_velocity", 0.0))

            if terminated:
                ep_survived = False

            if terminated or truncated:
                break

        ep_data = {
            "ep":           actual_ei,
            "return":       ep_ret,
            "return_disc":  ep_ret_d,
            "ep_len":       ep_len,
            "survived":     1.0 if ep_survived else 0.0,
            "xvel":         ep_xvel_sum / max(1, ep_len),
            "fallback_frac": fb_count / max(1, cov_count),
            "coverage":     cov_sum / max(1, cov_count),
        }
        R.append(ep_data["return"])
        RD.append(ep_data["return_disc"])
        L.append(ep_data["ep_len"])
        S.append(ep_data["survived"])
        XVEL.append(ep_data["xvel"])
        FB.append(ep_data["fallback_frac"])
        COV.append(ep_data["coverage"])

        if ckpt is not None:
            ckpt.append(ep_data)

    # With a checkpoint, aggregate over resumed and new episodes together.
    if ckpt is not None:
        return ckpt.to_metrics()

    R    = np.asarray(R,    dtype=np.float64)
    RD   = np.asarray(RD,   dtype=np.float64)
    L    = np.asarray(L,    dtype=np.float64)
    S    = np.asarray(S,    dtype=np.float64)
    XVEL = np.asarray(XVEL, dtype=np.float64)
    FB   = np.asarray(FB,   dtype=np.float64)
    COV  = np.asarray(COV,  dtype=np.float64)

    return {
        "return_mean":          float(R.mean()),
        "return_std":           float(R.std()),
        "return_median":        float(np.median(R)),
        "return_min":           float(R.min()),
        "return_max":           float(R.max()),
        "cvar_15":              float(cvar_lower_tail(R, 0.15)),
        "discounted_return_mean": float(RD.mean()),
        "ep_len_mean":          float(L.mean()),
        "success_rate":         float(S.mean()),
        "xvel_mean":            float(XVEL.mean()),
        "root_fallback_rate":   float(FB.mean()),
        "root_coverage_mean":   float(COV.mean()),
    }


# ---------------------------------------------------------------------------
# PPO helpers
# ---------------------------------------------------------------------------

def run_ppo_training(
    *,
    domain: str,
    seed: int,
    total_timesteps: int,
    out_path: str,
    net: str = "64,64",
) -> str:
    """Train a PPO behavior policy in a subprocess and return the checkpoint path."""
    train_script = os.path.join(os.path.dirname(__file__), "train_ppo_continuous.py")
    ensure_dir(os.path.dirname(out_path) or ".")
    cmd = [
        sys.executable, train_script,
        "--domain", domain,
        "--seed", str(seed),
        "--out", out_path,
        "--total_timesteps", str(total_timesteps),
        "--net", net,
    ]
    print(f"[PPO TRAIN] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    if not os.path.exists(out_path):
        raise RuntimeError(f"PPO checkpoint not created: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Dataset collection
# ---------------------------------------------------------------------------

def collect_dataset(
    env,
    ppo: PPO,
    n_trajs: int,
    max_steps: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Collect n_trajs episodes with the stochastic PPO policy.

    Returns flat arrays obs, actions, rewards, obs_next, dones; episode i is reset with
    seed `seed * 1000 + i`. `dones` marks both termination and truncation.
    """
    obs_list, act_list, rew_list, obs_next_list, done_list = [], [], [], [], []

    for i in range(n_trajs):
        obs, _ = env.reset(seed=int(seed * 1000 + i))
        for _ in range(max_steps):
            a, _ = ppo.predict(obs, deterministic=False)
            obs_next, r, terminated, truncated, _ = env.step(a)
            obs_list.append(obs.copy())
            act_list.append(np.asarray(a, dtype=np.float32).ravel())
            rew_list.append(float(r))
            obs_next_list.append(obs_next.copy())
            done_list.append(bool(terminated or truncated))
            obs = obs_next
            if terminated or truncated:
                break

    return {
        "obs":      np.asarray(obs_list,      dtype=np.float32),
        "actions":  np.asarray(act_list,       dtype=np.float32),
        "rewards":  np.asarray(rew_list,       dtype=np.float32),
        "obs_next": np.asarray(obs_next_list,  dtype=np.float32),
        "dones":    np.asarray(done_list,      dtype=np.bool_),
    }


def split_train_calib(
    dataset: Dict[str, np.ndarray],
    n_trajs: int,
    calib_frac: float,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Trajectory-level split of `dataset` into (train, calibration).

    Episodes are assigned whole, so no trajectory crosses the boundary; a
    calib_frac fraction of episodes (at least one) goes to calibration. All transitions
    of the calibration episodes are returned without subsampling. The D_bin / D_wilks /
    D_thr partition is drawn later, inside `fit_calibrator`.
    """
    rng = np.random.default_rng(seed)

    dones = dataset["dones"]
    N = int(dones.shape[0])
    end_idx = np.where(dones)[0]
    if end_idx.size == 0 or end_idx[-1] != N - 1:
        end_idx = np.concatenate([end_idx, np.array([N - 1])])
    start_idx = np.concatenate([np.array([0]), end_idx[:-1] + 1])
    episodes = list(zip(start_idx.tolist(), end_idx.tolist()))
    n_ep = len(episodes)
    n_cal_ep = max(1, int(round(n_ep * float(calib_frac))))

    ep_order = rng.permutation(n_ep)
    cal_ep_ids = sorted(ep_order[:n_cal_ep].tolist())
    tr_ep_ids = sorted(ep_order[n_cal_ep:].tolist())

    def _indices_of_episodes(ids: list) -> np.ndarray:
        if len(ids) == 0:
            return np.empty(0, dtype=np.int64)
        pieces = [np.arange(episodes[i][0], episodes[i][1] + 1, dtype=np.int64) for i in ids]
        return np.concatenate(pieces, axis=0)

    tr_idx = _indices_of_episodes(tr_ep_ids)
    cal_idx = _indices_of_episodes(cal_ep_ids)

    def _sel(d, i):
        return {k: v[i] for k, v in d.items()}

    return _sel(dataset, tr_idx), _sel(dataset, cal_idx)


def triple_split_calib_iid(
    n: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly partition `n` calibration indices into three disjoint blocks.

    D_bin fits the quantile bin edges on u, D_wilks the per-bin Wilks upper tolerance
    limits B_b given those edges, and D_thr the threshold eps = Quantile_{q_eps}(B(u_n))
    given B. The first two blocks have n // 3 elements; D_thr takes the remainder.

    Returns:
        (idx_bin, idx_wilks, idx_thr) as int64 arrays.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(int(n))
    n1 = n // 3
    n2 = n // 3
    idx_bin = perm[:n1]
    idx_wilks = perm[n1 : n1 + n2]
    idx_thr = perm[n1 + n2 :]
    return (
        np.asarray(idx_bin, dtype=np.int64),
        np.asarray(idx_wilks, dtype=np.int64),
        np.asarray(idx_thr, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Ensemble / calibration helpers
# ---------------------------------------------------------------------------

def train_ensemble(
    dataset: Dict[str, np.ndarray],
    obs_dim: int,
    act_dim: int,
    seed: int,
    epochs: int = 200,
    normalize_inputs: bool = True,
    bootstrap: bool = True,
    ensemble_size: int = 4,
    hidden_dims: Optional[Tuple[int, ...]] = None,
    target_normalize: bool = False,
    scale_aware_uncertainty: bool = False,
) -> EnsembleDynamicsSB3:
    """Train the dynamics ensemble on flat transitions (wrapped as {"observation": obs}).

    `target_normalize` or `scale_aware_uncertainty` selects EnsembleDynamicsV2.
    `target_normalize` standardizes the regression target per dimension during
    training and folds the de-normalization into the output layer, so the result is
    still a raw-delta predictor and only the loss weighting across output dimensions
    changes. Both are off by default, which gives the base MSE ensemble.
    """
    kwargs = dict(
        seed=seed,
        epochs=epochs,
        normalize_inputs=normalize_inputs,
        bootstrap=bootstrap,
        ensemble_size=ensemble_size,
    )
    if hidden_dims is not None:
        kwargs["hidden_dims"] = tuple(hidden_dims)
    cfg = EnsembleTrainConfig(**kwargs)
    if target_normalize or scale_aware_uncertainty:
        from safe_rl.offline.ensemble_dynamics_v2 import EnsembleDynamicsV2
        ens = EnsembleDynamicsV2(obs_dim=obs_dim, act_dim=act_dim, cfg=cfg,
                                 normalize_targets=bool(target_normalize),
                                 scale_aware_uncertainty=bool(scale_aware_uncertainty))
    else:
        ens = EnsembleDynamicsSB3(obs_dim=obs_dim, act_dim=act_dim, cfg=cfg)
    obs_d      = {"observation": dataset["obs"]}
    obs_next_d = {"observation": dataset["obs_next"]}
    ens.fit_from_transitions(obs_d, dataset["actions"], obs_next_d, verbose=True)
    return ens


def wilks_min_n_per_bin(n_bins_eff: int, delta: float, alpha: float = 0.05) -> int:
    """Minimum per-bin sample size for the Wilks tolerance limit.

    The bin maximum satisfies P(F(e_(n)) >= 1 - delta) = 1 - (1 - delta)^n. With a
    Bonferroni correction over B bins this requires
        n_b >= log(alpha / B) / log(1 - delta).
    Returns the ceiling of this threshold, or 0 for invalid arguments.
    """
    import math
    if n_bins_eff <= 0 or delta <= 0 or delta >= 1:
        return 0
    alpha_b = alpha / max(1, n_bins_eff)
    if alpha_b <= 0:
        return int(1e9)
    n_min = math.log(alpha_b) / math.log(1.0 - delta)
    return int(math.ceil(n_min))


def wilks_optimal_bins(N_cal: int, delta: float = 0.05, alpha: float = 0.05) -> int:
    """Largest bin count B* admissible for N_cal calibration samples.

    B* is the largest B with
        N_cal >= (B / delta) * log(B / alpha),
    i.e. the largest number of bins whose per-bin bounds hold with confidence
    1 - alpha / B each. B* grows as O(N_cal / log N_cal). Found by binary search
    on [1, N_cal].
    """
    import math
    if N_cal <= 0:
        return 1
    lo, hi = 1, N_cal
    while lo < hi:
        mid = (lo + hi + 1) // 2
        required = (mid / delta) * math.log(mid / alpha)
        if required <= N_cal:
            lo = mid
        else:
            hi = mid - 1
    return max(1, lo)


def wilks_eps_check(
    u_cal: np.ndarray,
    calibrator: UncertaintyCalibrator,
    eps: float,
    delta: float,
    alpha: float = 0.05,
) -> Tuple[float, bool, int, int]:
    """Check the per-bin Wilks sample-size condition on u_cal.

    Returns:
        (eps_out, is_valid, n_b_min, wilks_threshold): eps_out is eps if every
        effective bin has at least wilks_threshold samples and -inf otherwise
        (conservative collapse); n_b_min is the smallest bin count.
    """
    edges = calibrator.bin_edges
    n_bins_eff = len(calibrator.bin_bounds)
    threshold = wilks_min_n_per_bin(n_bins_eff, delta, alpha)

    bin_idx = np.clip(np.digitize(u_cal, edges[1:-1], right=False), 0, n_bins_eff - 1)
    counts = np.array([int(np.sum(bin_idx == b)) for b in range(n_bins_eff)])
    n_b_min = int(counts.min()) if counts.size > 0 else 0

    is_valid = n_b_min >= threshold
    # eps = -inf makes B(u) <= eps false everywhere, so A_safe is empty and the
    # policy falls back to PPO. +inf would certify every action.
    eps_out = eps if is_valid else float("-inf")
    return eps_out, is_valid, n_b_min, threshold


def calib_min_bins_check(
    n_cal: int,
    delta: float,
    bins: int,
    min_bin: int,
    q_eps: float,
    eps_raw: float,
    uncertainty_type: str,
    calib_frac: float,
    ds: int,
    alpha: float = 0.05,
) -> float:
    """Enforce the minimum bin count implied by q_eps and print a calibration summary.

    The minimum is calib_min_bins = ceil(1 / (1 - q_eps)), e.g. 20 for q_eps = 0.95.
    With quantile binning each bin holds a 1/B fraction of D_cal. If
    1/B >= 1 - q_eps, eps equals the bound of the last bin, and since B(u) is clamped
    to that bound for u above the calibration range, the gate would certify every
    out-of-distribution action. With B >= calib_min_bins, eps lies below the last bound.

    Returns:
        eps_raw if bins >= calib_min_bins, otherwise -inf (conservative collapse).
    """
    calib_min_bins = math.ceil(1.0 / (1.0 - q_eps))

    # Smallest N_cal for which B* >= calib_min_bins.
    n_cal_exit = int(math.ceil((calib_min_bins / delta) * math.log(calib_min_bins / alpha)))

    # Number of trajectories needed to reach it, at the current mean episode length.
    avg_ep_len = (n_cal / (calib_frac * max(1, ds))) if (calib_frac * ds) > 0 else 1.0
    n_trajs_exit = int(math.ceil(n_cal_exit / max(1.0, avg_ep_len * calib_frac)))

    # Total-sample Wilks condition for the chosen B.
    n_cal_for_b = int(math.ceil((bins / delta) * math.log(bins / alpha))) if bins > 0 else 0
    wilks_total_ok = n_cal >= n_cal_for_b

    low_data = bins < calib_min_bins
    eps_out = float("-inf") if low_data else eps_raw

    sep = "─" * 58
    print(f"\n[CALIB/{uncertainty_type}] {sep}", flush=True)
    print(
        f"  N_cal={n_cal} | delta={delta} | q_eps={q_eps} | alpha={alpha}",
        flush=True,
    )
    print(
        f"  calib_min_bins     : ceil(1/(1-{q_eps})) = {calib_min_bins}  [auto from q_eps]",
        flush=True,
    )
    print(
        f"  B* (Wilks-optimal) : {bins}"
        f"  (need N_cal>={n_cal_for_b} for B={bins}:"
        f" {'OK' if wilks_total_ok else 'FAIL'})",
        flush=True,
    )
    print(f"  n_b_wilks (per bin): {min_bin}", flush=True)
    if low_data:
        print(
            f"  min_bins check     : B*={bins} < {calib_min_bins}"
            f"  \u2190 LOW-DATA REGIME",
            flush=True,
        )
        print(f"  eps (would be)     : {eps_raw:.4f} \u2192 overridden to -inf", flush=True)
        print(f"  \u2192 Cal-SPI \u2261 PPO (conservative collapse)", flush=True)
        print(
            f"  Exit condition     : N_cal >= {n_cal_exit}"
            f"  (~{n_trajs_exit} trajs at avg_ep_len\u2248{avg_ep_len:.0f})",
            flush=True,
        )
    else:
        print(
            f"  min_bins check     : B*={bins} >= {calib_min_bins}"
            f"  \u2190 GATE ACTIVE",
            flush=True,
        )
        print(f"  eps                : {eps_raw:.4f}", flush=True)
    print(f"[CALIB/{uncertainty_type}] {sep}\n", flush=True)
    return eps_out


def _parse_candidate_bins(spec: Optional[str]) -> Optional[List[int]]:
    """Parse a comma-separated list of bin counts (e.g. "20,40,80").

    Values below 2 and unparsable tokens are dropped. Returns None for an empty spec,
    which selects the default set from `_theory_candidate_bins`.
    """
    if spec is None:
        return None
    s = str(spec).strip()
    if not s:
        return None
    out: List[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = int(tok)
        except ValueError:
            continue
        if v >= 2:
            out.append(v)
    return out if out else None


def _theory_candidate_bins(q_eps: float, K: int = 4) -> List[int]:
    """Default candidate bin counts {B_min, 2 B_min, ..., K B_min}.

    B_min = ceil(1 / (1 - q_eps)); q_eps = 0.95 and K = 4 give {20, 40, 60, 80}.
    """
    Bm = int(math.ceil(1.0 / max(1e-12, 1.0 - float(q_eps))))
    return [int(Bm * (k + 1)) for k in range(int(K))]


def fit_calibrator(
    ens: EnsembleDynamicsSB3,
    calib_dataset: Dict[str, np.ndarray],
    cfg: UncertaintyCalibratorConfig,
    uncertainty_type: str = "usad",
    wilks_alpha: float = 0.05,
    q_eps: float = 0.95,
    triple_split: bool = True,
    candidate_bins: Optional[List[int]] = None,
    single_B_override: Optional[int] = None,
    n_bootstrap: int = 200,
    random_seed: int = 0,
    legacy_mode: bool = False,
) -> Tuple[UncertaintyCalibrator, float, np.ndarray, np.ndarray]:
    """Fit the calibrated bound B(u) and the gate threshold eps on calibration data.

    Args:
        uncertainty_type: "usad" (max pairwise disagreement of the member means) or
            "epistemic" (mean per-dimension variance of the member means).
        triple_split: split the calibration sample into D_bin (bin edges), D_wilks
            (per-bin Wilks bounds) and D_thr (eps). Each candidate B in
            `candidate_bins` (or `single_B_override`, or the default set) is fitted,
            and B is selected with the one-standard-error rule over bootstrap
            estimates of eps. With False, or with fewer than 10 samples per block,
            edges, bounds and eps are fitted jointly on the whole sample, which does
            not give the data independence required by the coverage guarantee.
        legacy_mode: with the triple split, use B = B* (or cfg.n_bins), per-bin
            empirical (1 - delta)-quantiles and a plain alpha / B correction.

    If some effective bin has fewer samples than the Wilks threshold, eps is set to
    -inf (conservative collapse).

    Returns:
        (calibrator, eps, u_calib, e_calib).
    """
    obs_flat  = calib_dataset["obs"]
    actions   = calib_dataset["actions"]
    obs_next  = calib_dataset["obs_next"]

    if uncertainty_type == "usad":
        u_calib = np.array([
            ens.uncertainty_usad({"observation": obs_flat[i]}, actions[i])
            for i in range(len(obs_flat))
        ], dtype=np.float32)
    else:  # "epistemic"
        u_calib = np.array([
            ens.uncertainty_epistemic({"observation": obs_flat[i]}, actions[i])
            for i in range(len(obs_flat))
        ], dtype=np.float32)

    # Error e = ||mu_bar(s, a) - s'||_2. With a scale-aware V2 ensemble, u and e are
    # both computed on deltas divided by their per-dimension scale, so no state
    # component dominates because of its units.
    if hasattr(ens, "calibration_error") and getattr(ens, "scale_aware_uncertainty", False):
        e_calib = np.array([
            ens.calibration_error(obs_flat[i], actions[i], obs_next[i])
            for i in range(len(obs_flat))
        ], dtype=np.float32)
    else:
        e_calib = np.array([
            float(np.linalg.norm(
                ens.predict_next_obs_vec_mean({"observation": obs_flat[i]}, actions[i])
                - obs_next[i]
            ))
            for i in range(len(obs_flat))
        ], dtype=np.float32)

    # D_bin fits bin edges, D_wilks the per-bin bounds on those edges, D_thr the
    # q_eps quantile of B(u). Each block is independent of the statistic fitted on
    # the next, which is what the Wilks coverage guarantee requires.
    idx_bin, idx_wilks, idx_thr = triple_split_calib_iid(
        n=int(u_calib.shape[0]), seed=0
    )
    _need_min = 10
    if (
        triple_split
        and idx_bin.size >= _need_min
        and idx_wilks.size >= _need_min
        and idx_thr.size >= _need_min
        and not legacy_mode
    ):
        # Candidate set: candidate_bins, else single_B_override, else the default set.
        if candidate_bins is not None and len(candidate_bins) > 0:
            cand_list = [int(B) for B in candidate_bins if int(B) >= 2]
        elif single_B_override is not None and int(single_B_override) > 0:
            cand_list = [int(single_B_override)]
        else:
            cand_list = _theory_candidate_bins(float(q_eps))
        cand_list = sorted(set(cand_list))

        # Pass 1: bin edges fitted on D_bin (quantile binning), one set per B.
        u_bin = u_calib[idx_bin]
        bin_edges_per_B: Dict[int, np.ndarray] = {}
        for B in cand_list:
            n_bins_req = max(2, int(B))
            if cfg.binning == "quantile":
                qs = np.linspace(0.0, 1.0, n_bins_req + 1)
                edges_B = np.quantile(u_bin, qs)
            else:
                u_min_b, u_max_b = float(np.min(u_bin)), float(np.max(u_bin))
                if u_max_b <= u_min_b:
                    u_max_b = u_min_b + 1e-12
                edges_B = np.linspace(u_min_b, u_max_b, n_bins_req + 1)
            edges_B = np.asarray(edges_B, dtype=np.float64)
            for i in range(1, len(edges_B)):
                if edges_B[i] <= edges_B[i - 1]:
                    edges_B[i] = edges_B[i - 1] + 1e-12
            bin_edges_per_B[int(B)] = edges_B

        # Passes 2-3: per-bin Wilks bounds on D_wilks, eps_B on D_thr, and
        # one-standard-error selection of B_dagger over the candidates.
        calibrator, eps_raw, sel_diag = UncertaintyCalibrator.fit_with_selection(
            u_wilks=u_calib[idx_wilks],
            e_wilks=e_calib[idx_wilks],
            u_thr=u_calib[idx_thr],
            candidate_bins=cand_list,
            bin_edges_per_B=bin_edges_per_B,
            cfg_template=cfg,
            q_eps=float(q_eps),
            n_bootstrap=int(n_bootstrap),
            random_seed=int(random_seed),
        )

        # Per-candidate table and selected B.
        print(
            f"[CALIB] Triple-split + candidate sweep: N_cal={u_calib.shape[0]} "
            f"(bin={idx_bin.size}, wilks={idx_wilks.size}, thr={idx_thr.size}); "
            f"candidates={cand_list}",
            flush=True,
        )
        print(
            f"[CALIB] {'B':>4} {'min_n_b':>8} {'req_n_b':>8} "
            f"{'alpha_eff':>10} {'eps_B':>9} {'admissible':>11} {'selected':>9}",
            flush=True,
        )
        for rec in sel_diag.get("per_candidate", []):
            print(
                f"[CALIB] {rec['B']:>4d} {rec['min_bin_count']:>8d} "
                f"{rec['required_min_bin_count']:>8d} "
                f"{rec['alpha_eff']:>10.4g} {rec['epsilon_B']:>9.4f} "
                f"{('YES' if rec['admissible'] else '---'):>11} "
                f"{('  *  ' if rec['selected'] else '     '):>9}",
                flush=True,
            )
        if sel_diag.get("collapsed", False):
            print(
                f"[CALIB] B_adm = empty -> conservative collapse (eps=-inf, fallback PPO).",
                flush=True,
            )
        else:
            print(
                f"[CALIB] B_dagger={sel_diag.get('B_dagger')} (B_best={sel_diag.get('B_best')}) "
                f"eps={sel_diag.get('epsilon'):.4f} (eps_min={sel_diag.get('epsilon_min'):.4f}, "
                f"eps_se={sel_diag.get('epsilon_se'):.4f}, n_boot={n_bootstrap}) "
                f"selection_rule=one_standard_error",
                flush=True,
            )
    elif (
        triple_split and legacy_mode
        and idx_bin.size >= _need_min
        and idx_wilks.size >= _need_min
        and idx_thr.size >= _need_min
    ):
        # Legacy mode: B = wilks_optimal_bins(|D_wilks|) unless cfg.n_bins is set,
        # per-bin empirical (1 - delta)-quantile, alpha / B Bonferroni correction
        # without a correction for the candidate set.
        u_bin = u_calib[idx_bin]
        _n_fit = idx_wilks.size
        if int(cfg.n_bins) > 0:
            B_legacy = int(cfg.n_bins)
        else:
            B_legacy = wilks_optimal_bins(_n_fit, float(cfg.delta))
        n_bins_req = max(2, B_legacy)
        if cfg.binning == "quantile":
            qs = np.linspace(0.0, 1.0, n_bins_req + 1)
            edges_fixed = np.quantile(u_bin, qs)
        else:
            u_min_b, u_max_b = float(np.min(u_bin)), float(np.max(u_bin))
            if u_max_b <= u_min_b:
                u_max_b = u_min_b + 1e-12
            edges_fixed = np.linspace(u_min_b, u_max_b, n_bins_req + 1)
        from dataclasses import replace as _dc_replace
        cfg_legacy = _dc_replace(cfg, n_bins=int(B_legacy), mode="quantile")
        calibrator = UncertaintyCalibrator.fit(
            u_calib[idx_wilks], e_calib[idx_wilks],
            cfg=cfg_legacy, bin_edges=edges_fixed,
        )
        eps_raw = float(
            np.quantile(calibrator.bound_batch(u_calib[idx_thr]), float(q_eps))
        )
        print(
            f"[CALIB] LEGACY mode (--calib_legacy_mode 1): "
            f"B={B_legacy} (Wilks-optimal max), mode=quantile, alpha/B Bonferroni; "
            f"N_cal={u_calib.shape[0]} (bin={idx_bin.size}, wilks={idx_wilks.size}, thr={idx_thr.size}); "
            f"eps={eps_raw:.4f}",
            flush=True,
        )
    else:
        # Single-split fit: edges, per-bin bounds and eps on the same sample. Used
        # with --calib_triple_split 0 or when N_cal is too small for three blocks.
        calibrator = UncertaintyCalibrator.fit(u_calib, e_calib, cfg=cfg)
        eps_raw = float(np.quantile(calibrator.bound_batch(u_calib), float(q_eps)))
        if not triple_split:
            reason = "disabled by --calib_triple_split 0"
        else:
            reason = f"N_cal={u_calib.shape[0]} < 3*{_need_min}"
        print(
            f"[CALIB] Legacy single-split fit ({reason}); "
            f"Wilks guarantees require the triple split (Section method).",
            flush=True,
        )

    # Conservative collapse if some bin has too few samples.
    eps, wilks_ok, n_b_min, wilks_thr = wilks_eps_check(
        u_calib, calibrator, eps_raw, cfg.delta, wilks_alpha
    )
    n_bins_eff = len(calibrator.bin_bounds)
    if wilks_ok:
        print(
            f"[CALIB] Wilks OK: n_b_min={n_b_min} >= {wilks_thr} "
            f"(B={n_bins_eff}, delta={cfg.delta}, alpha={wilks_alpha}) "
            f"eps={eps:.4f}",
            flush=True,
        )
    else:
        print(
            f"[CALIB] Wilks FAIL: n_b_min={n_b_min} < {wilks_thr} "
            f"(B={n_bins_eff}, delta={cfg.delta}, alpha={wilks_alpha}) "
            f"-> eps=-inf (conservative collapse)",
            flush=True,
        )

    return calibrator, eps, u_calib, e_calib


# ---------------------------------------------------------------------------
# Method factories
# ---------------------------------------------------------------------------

def _select_planner_class(backend: str):
    """Return the non-CSPI planner class for a backend name.

    'batched' selects the batched planner, which produces the same actions as the
    original planner under fixed_per_sim sampling; any other value selects the
    original planner.
    """
    if str(backend) == "batched":
        return OfflineMCTS_DPW_PPO_Batched
    return OfflineMCTS_DPW_PPO_SPIBB_Parametric


def make_mcts_calibrated(
    ens: EnsembleDynamicsSB3,
    calibrator: UncertaintyCalibrator,
    eps: float,
    ppo: PPO,
    model_cfg: ModelContinuousEnvConfig,
    mcts_cfg: OfflineMCTSConfig,
    seed: int,
    backend: str = "original",
    cspi_margin_mult: float = 1.0,
    cspi_v_rollout_horizon: int = 0,
    cspi_v_rollout_n_samples: int = 1,
    cspi_e_plan_scale: float = 0.5,
    cspi_delta_piB: float = 0.0,
    cspi_frozen_root: bool = True,
    cspi_n_min_root_visits: int = 3,
    use_ppo_leaf_value: bool = False,
):
    """Build the Cal-SPI planner with the USAD gate.

    The planner inserts the baseline action a_b = pi_b(s) at the root, runs MCTS-DPW
    over gate-certified actions, and switches away from a_b only when
    Q_hat_safe(a*) - V_hat_b(s) >= Delta_margin (baseline-relative switch); otherwise
    it returns a_b. `cspi_margin_mult = 0` gives the plain argmax over A_safe.

    `backend` is one of original, batched, numba, numba_scaled, array.
    """
    from safe_rl.algorithms.offline.cal_spi.mcts_cspi import (
        OfflineMCTS_DPW_PPO_SPIBB_CSPI_Parametric,
        OfflineMCTS_DPW_PPO_Batched_CSPI,
    )
    model_env = _make_model_env(ens, model_cfg)
    mcts_cfg.eps_threshold = float(eps)
    mcts_cfg.gate_mode = "calibrated"
    mcts_cfg.uncertainty_type = "usad"

    extra = {}
    if str(backend) == "batched":
        PlannerCls = OfflineMCTS_DPW_PPO_Batched_CSPI
    elif str(backend) == "numba":
        from safe_rl.algorithms.offline.cal_spi.mcts_numba import (
            OfflineMCTS_DPW_PPO_Numba_CSPI,
        )
        PlannerCls = OfflineMCTS_DPW_PPO_Numba_CSPI
    elif str(backend) == "array":
        # Compiled search loop over preallocated buffers; produces the same
        # actions as the numba backend for the same seed.
        from safe_rl.algorithms.offline.cal_spi.mcts_array import (
            OfflineMCTS_DPW_PPO_Array_CSPI,
        )
        PlannerCls = OfflineMCTS_DPW_PPO_Array_CSPI
        # The gate must score the statistic the calibrator bounded, so delta_scale
        # is passed exactly when the ensemble uses scale-aware uncertainty.
        if getattr(ens, "scale_aware_uncertainty", False):
            _sc = getattr(ens, "delta_scale", None)
            if _sc is None:
                raise ValueError(
                    "the ensemble declares a scale-aware uncertainty but carries "
                    "no delta_scale; the planner would gate on the unscaled "
                    "statistic while the calibrator bounded the scaled one"
                )
            extra["delta_scale"] = _sc
    elif str(backend) == "numba_scaled":
        # Scale-aware gate: disagreement is scored on scaled deltas, matching the
        # statistic bounded by a scale-aware calibrator.
        from safe_rl.algorithms.offline.cal_spi.mcts_numba_scaled import (
            OfflineMCTS_DPW_PPO_Numba_CSPI_Scaled,
        )
        PlannerCls = OfflineMCTS_DPW_PPO_Numba_CSPI_Scaled
        extra["delta_scale"] = getattr(ens, "delta_scale", None)
    else:
        PlannerCls = OfflineMCTS_DPW_PPO_SPIBB_CSPI_Parametric

    return PlannerCls(
        model_env=model_env,
        ppo_policy=ppo,
        cfg=mcts_cfg,
        calibrator=calibrator,
        use_gate=True,
        use_ppo_prior=True,
        use_ppo_leaf_value=bool(use_ppo_leaf_value),
        rng_seed=seed,
        cspi_margin_mult=float(cspi_margin_mult),
        cspi_v_rollout_horizon=int(cspi_v_rollout_horizon),
        cspi_v_rollout_n_samples=int(cspi_v_rollout_n_samples),
        cspi_e_plan_scale=float(cspi_e_plan_scale),
        cspi_delta_piB=float(cspi_delta_piB),
        cspi_frozen_root=bool(cspi_frozen_root),
        cspi_n_min_root_visits=int(cspi_n_min_root_visits),
        **extra,
    )


def make_mcts_calibrated_var(
    ens: EnsembleDynamicsSB3,
    calibrator: UncertaintyCalibrator,
    eps: float,
    ppo: PPO,
    model_cfg: ModelContinuousEnvConfig,
    mcts_cfg: OfflineMCTSConfig,
    seed: int,
    backend: str = "original",
    cspi_margin_mult: float = 1.0,
    cspi_v_rollout_horizon: int = 0,
    cspi_v_rollout_n_samples: int = 1,
    cspi_e_plan_scale: float = 0.5,
    cspi_delta_piB: float = 0.0,
    cspi_frozen_root: bool = True,
    cspi_n_min_root_visits: int = 3,
):
    """Build the Cal-SPI planner with the epistemic-variance gate; see make_mcts_calibrated."""
    from safe_rl.algorithms.offline.cal_spi.mcts_cspi import (
        OfflineMCTS_DPW_PPO_SPIBB_CSPI_Parametric,
        OfflineMCTS_DPW_PPO_Batched_CSPI,
    )
    model_env = _make_model_env(ens, model_cfg)
    mcts_cfg.eps_threshold = float(eps)
    mcts_cfg.gate_mode     = "calibrated"
    mcts_cfg.uncertainty_type = "epistemic"

    extra = {}
    if str(backend) == "batched":
        PlannerCls = OfflineMCTS_DPW_PPO_Batched_CSPI
    elif str(backend) == "numba":
        from safe_rl.algorithms.offline.cal_spi.mcts_numba import (
            OfflineMCTS_DPW_PPO_Numba_CSPI,
        )
        PlannerCls = OfflineMCTS_DPW_PPO_Numba_CSPI
    elif str(backend) == "array":
        # Compiled search loop over preallocated buffers; produces the same
        # actions as the numba backend for the same seed.
        from safe_rl.algorithms.offline.cal_spi.mcts_array import (
            OfflineMCTS_DPW_PPO_Array_CSPI,
        )
        PlannerCls = OfflineMCTS_DPW_PPO_Array_CSPI
        # The gate must score the statistic the calibrator bounded, so delta_scale
        # is passed exactly when the ensemble uses scale-aware uncertainty.
        if getattr(ens, "scale_aware_uncertainty", False):
            _sc = getattr(ens, "delta_scale", None)
            if _sc is None:
                raise ValueError(
                    "the ensemble declares a scale-aware uncertainty but carries "
                    "no delta_scale; the planner would gate on the unscaled "
                    "statistic while the calibrator bounded the scaled one"
                )
            extra["delta_scale"] = _sc
    elif str(backend) == "numba_scaled":
        # Scale-aware gate: disagreement is scored on scaled deltas, matching the
        # statistic bounded by a scale-aware calibrator.
        from safe_rl.algorithms.offline.cal_spi.mcts_numba_scaled import (
            OfflineMCTS_DPW_PPO_Numba_CSPI_Scaled,
        )
        PlannerCls = OfflineMCTS_DPW_PPO_Numba_CSPI_Scaled
        extra["delta_scale"] = getattr(ens, "delta_scale", None)
    else:
        PlannerCls = OfflineMCTS_DPW_PPO_SPIBB_CSPI_Parametric

    return PlannerCls(
        model_env=model_env,
        ppo_policy=ppo,
        cfg=mcts_cfg,
        calibrator=calibrator,
        use_gate=True,
        use_ppo_prior=True,
        use_ppo_leaf_value=False,
        rng_seed=seed,
        cspi_margin_mult=float(cspi_margin_mult),
        cspi_v_rollout_horizon=int(cspi_v_rollout_horizon),
        cspi_v_rollout_n_samples=int(cspi_v_rollout_n_samples),
        cspi_e_plan_scale=float(cspi_e_plan_scale),
        cspi_delta_piB=float(cspi_delta_piB),
        cspi_frozen_root=bool(cspi_frozen_root),
        cspi_n_min_root_visits=int(cspi_n_min_root_visits),
        **extra,
    )


def make_mcts_no_gate(
    ens: EnsembleDynamicsSB3,
    ppo: PPO,
    model_cfg: ModelContinuousEnvConfig,
    mcts_cfg: OfflineMCTSConfig,
    seed: int,
) -> OfflineMCTS_DPW_Ensemble_UNSAFE:
    """Build the unconstrained MCTS planner (no gate, no PPO prior, zero leaf value)."""
    mcts_cfg_unsafe = OfflineMCTSConfig(
        num_sims=mcts_cfg.num_sims,
        max_depth=mcts_cfg.max_depth,
        gamma=mcts_cfg.gamma,
        dpw_alpha=mcts_cfg.dpw_alpha,
        dpw_k=mcts_cfg.dpw_k,
        c_puct=mcts_cfg.c_puct,
        n_actions_root=mcts_cfg.n_actions_root,
        n_actions_per_node=mcts_cfg.n_actions_per_node,
        leaf_value_mode="zero",
    )
    model_env = _make_model_env(ens, model_cfg)
    act_dim = DOMAIN_CONFIGS[model_cfg.domain]["act_dim"]
    action_low  = np.full(act_dim, -1.0, dtype=np.float32)
    action_high = np.full(act_dim,  1.0, dtype=np.float32)
    return OfflineMCTS_DPW_Ensemble_UNSAFE(
        model_env=model_env,
        cfg=mcts_cfg_unsafe,
        action_low=action_low,
        action_high=action_high,
        rng_seed=seed,
    )


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(args) -> None:
    """Run every requested method on every (seed, dataset size) cell and write results."""
    domain_cfg = DOMAIN_CONFIGS[args.domain]
    obs_dim    = domain_cfg["obs_dim"]
    act_dim    = domain_cfg["act_dim"]
    max_steps  = int(args.max_steps)

    ppo_path = str(args.ppo_global_path)

    # ── Behavior policy: one PPO checkpoint shared by all cells ──────────
    if not os.path.exists(ppo_path):
        print(f"[BENCH] PPO not found at {ppo_path}, training...", flush=True)
        run_ppo_training(
            domain=args.domain,
            seed=0,
            total_timesteps=int(args.ppo_timesteps),
            out_path=ppo_path,
        )
    else:
        print(f"[BENCH] PPO already at {ppo_path}", flush=True)

    ppo = PPO.load(ppo_path)
    ppo_act_fn = lambda obs: ppo.predict(obs, deterministic=True)[0]

    rows = []
    should_run = set(args.run_methods)

    for seed in args.seeds:
        print(f"\n[BENCH] === seed={seed} ===", flush=True)

        env_data = make_continuous_env(args.domain, seed=seed)
        env_eval = make_continuous_env(args.domain, seed=seed)
        # Each fork-parallel evaluation worker builds its own env from this factory.
        _domain_local = args.domain
        _seed_local = seed
        def _eval_env_factory():
            return make_continuous_env(_domain_local, seed=_seed_local)

        for ds in sorted(args.dataset_trajs):
            outdir = str(args.outdir)
            ensure_dir(outdir)

            print(f"\n[BENCH] ds={ds} seed={seed}", flush=True)

            # ── Dataset ───────────────────────────────────────────────────
            # Collection is skipped when --force_retrain is off, the ensemble and
            # calibrator caches exist (epistemic too if requested), and every
            # requested method needs no dataset at evaluation time.
            _ens_path_pre = os.path.join(str(args.outdir), "ensemble.pt")
            _cal_path_pre = os.path.join(str(args.outdir), "calibrator_usad.pkl")
            _cal_npz_pre  = os.path.join(str(args.outdir), "calibration_data_usad.npz")
            _cal_var_path_pre = os.path.join(str(args.outdir), "calibrator_epistemic.pkl")
            _cal_var_npz_pre  = os.path.join(str(args.outdir), "calibration_data_epistemic.npz")
            _methods_no_data = {"ppo_only", "calibrated_usad", "calibrated_variance",
                                "mcts_only_no_gate"}
            _need_var = "calibrated_variance" in set(args.run_methods)
            _all_cache_present = (
                os.path.exists(_ens_path_pre)
                and os.path.exists(_cal_path_pre)
                and os.path.exists(_cal_npz_pre)
                and (not _need_var or (
                    os.path.exists(_cal_var_path_pre) and os.path.exists(_cal_var_npz_pre)
                ))
            )
            _skip_collect = (
                (not args.force_retrain)
                and _all_cache_present
                and (set(args.run_methods) <= _methods_no_data)
            )

            collect_seed = int(seed) * 100_000 + int(ds)
            if _skip_collect:
                print(
                    f"[BENCH] Skipping data collection: cache hit (ensemble + calibrator) "
                    f"and run_methods={sorted(args.run_methods)} are dataset-free at eval.",
                    flush=True,
                )
                # Empty splits with the keys of `split_train_calib`; never read,
                # since the ensemble and calibrators are loaded from cache.
                ds_sub = None
                D_train = {
                    "obs":      np.zeros((0, obs_dim), dtype=np.float32),
                    "actions":  np.zeros((0, act_dim), dtype=np.float32),
                    "obs_next": np.zeros((0, obs_dim), dtype=np.float32),
                    "rewards":  np.zeros((0,), dtype=np.float32),
                    "dones":    np.zeros((0,), dtype=np.float32),
                }
                D_calib = {k: v.copy() for k, v in D_train.items()}
            else:
                # The dataset is saved next to the ensemble so later runs reuse the
                # same transitions. PPO samples actions stochastically, so the torch
                # and numpy global RNGs are seeded before collection.
                _ds_path = os.path.join(outdir, "dataset.npz")
                if os.path.exists(_ds_path) and not args.force_retrain:
                    print(f"[BENCH] Loading dataset from {_ds_path}", flush=True)
                    _z = np.load(_ds_path)
                    ds_sub = {k: _z[k] for k in
                              ("obs", "actions", "rewards", "obs_next", "dones")}
                else:
                    print(
                        f"[BENCH] Collecting {ds} trajectories (collect_seed={collect_seed})...",
                        flush=True,
                    )
                    torch.manual_seed(int(collect_seed))
                    np.random.seed(int(collect_seed) % (2 ** 32))
                    ds_sub = collect_dataset(
                        env_data, ppo, n_trajs=ds, max_steps=max_steps, seed=collect_seed
                    )
                    np.savez_compressed(_ds_path, **ds_sub)
                    print(f"[BENCH] Dataset saved → {_ds_path} "
                          f"({len(ds_sub['obs'])} transitions)", flush=True)
                D_train, D_calib = split_train_calib(
                    ds_sub, ds, calib_frac=float(args.calib_frac), seed=seed
                )

            # ── Ensemble training ─────────────────────────────────────────
            ens_path = os.path.join(outdir, "ensemble.pt")
            ensemble_hidden = tuple(int(h) for h in args.ensemble_hidden_dims.split(",") if h.strip())
            if os.path.exists(ens_path) and not args.force_retrain:
                print(f"[BENCH] Loading ensemble from {ens_path}", flush=True)
                state = torch.load(ens_path, map_location="cpu")
                # Architecture from the checkpoint; 512x512 and K=4 if not stored.
                ckpt_hidden = tuple(state.get("hidden_dims", (512, 512)))
                ckpt_K      = int(state.get("ensemble_size", 4))
                ens = EnsembleDynamicsSB3(
                    obs_dim=obs_dim, act_dim=act_dim,
                    cfg=EnsembleTrainConfig(
                        ensemble_size=ckpt_K, hidden_dims=ckpt_hidden,
                    ),
                )
                for k, m in enumerate(ens.ensemble):
                    m.load_state_dict(state["weights"][k] if "weights" in state else state[k])
                # Input normalization stats, if stored.
                if state.get("norm_mean") is not None:
                    ens.input_mean = state["norm_mean"]
                    ens.input_std  = state["norm_std"]
                    print(f"[BENCH] Ensemble norm stats loaded", flush=True)
                # Per-dimension delta scale, if stored, so the cached ensemble keeps
                # the metric its calibrator was fitted on.
                if state.get("delta_scale") is not None:
                    ens.delta_scale = np.asarray(state["delta_scale"], np.float32)
                    ens.scale_aware_uncertainty = bool(args.uncertainty_scale_aware)
                print(f"[BENCH] Loaded ensemble: K={ckpt_K} hidden={ckpt_hidden}", flush=True)
            else:
                print(f"[BENCH] Training ensemble on {len(D_train['obs'])} transitions "
                      f"(K={args.ensemble_size}, hidden={ensemble_hidden}, "
                      f"epochs={args.ensemble_epochs}, normalize={args.ensemble_normalize}, "
                      f"bootstrap={args.ensemble_bootstrap})...", flush=True)
                ens = train_ensemble(
                    D_train, obs_dim=obs_dim, act_dim=act_dim, seed=seed,
                    epochs=int(args.ensemble_epochs),
                    normalize_inputs=bool(args.ensemble_normalize),
                    bootstrap=bool(args.ensemble_bootstrap),
                    ensemble_size=int(args.ensemble_size),
                    hidden_dims=ensemble_hidden,
                    target_normalize=bool(args.ensemble_target_normalize),
                    scale_aware_uncertainty=bool(args.uncertainty_scale_aware),
                )
                ensure_dir(outdir)
                save_state = {
                    "weights": {k: m.state_dict() for k, m in enumerate(ens.ensemble)},
                    "norm_mean": ens.input_mean,
                    "norm_std":  ens.input_std,
                    "hidden_dims": list(ensemble_hidden),
                    "ensemble_size": int(args.ensemble_size),
                    "delta_scale": getattr(ens, "delta_scale", None),
                }
                torch.save(save_state, ens_path)

            # With scale-aware uncertainty the calibrator bounds disagreement divided
            # by the per-dimension delta scale; without that scale the gate would
            # score a different statistic, so the run stops here.
            if bool(args.uncertainty_scale_aware) and getattr(ens, "delta_scale", None) is None:
                raise RuntimeError(
                    "--uncertainty_scale_aware was requested but the ensemble "
                    "carries no delta_scale (a cached checkpoint predating the "
                    "scale-aware path?). Rerun with --force_retrain rather than "
                    "gating on the unscaled statistic."
                )
            print(f"[BENCH] Gate statistic: "
                  f"{'scale-aware USAD' if bool(args.uncertainty_scale_aware) else 'USAD'}"
                  f" (delta_scale {'present' if getattr(ens, 'delta_scale', None) is not None else 'absent'})",
                  flush=True)

            # ── Calibrator (USAD) ─────────────────────────────────────────
            cal_path = os.path.join(outdir, "calibrator_usad.pkl")
            if os.path.exists(cal_path) and not args.force_retrain:
                with open(cal_path, "rb") as f:
                    calibrator, eps = pickle.load(f)
                print(f"[BENCH] Loaded USAD calibrator (eps={eps:.4f})", flush=True)
                # Re-check the Wilks and min-bins conditions on the saved u_cal.
                npz_path = os.path.join(outdir, "calibration_data_usad.npz")
                if os.path.exists(npz_path) and np.isfinite(eps):
                    u_cal_saved = np.load(npz_path)["u"]
                    eps, wilks_ok, n_b_min, wilks_thr = wilks_eps_check(
                        u_cal_saved, calibrator, eps,
                        float(args.calib_delta), 0.05,
                    )
                    if not wilks_ok:
                        print(
                            f"[CALIB] Wilks FAIL (post-load): n_b_min={n_b_min} < {wilks_thr} "
                            f"-> eps=-inf (conservative collapse)",
                            flush=True,
                        )
                    _bins_cached = len(calibrator.bin_bounds)
                    _min_bin_cached = wilks_min_n_per_bin(_bins_cached, float(args.calib_delta))
                    eps = calib_min_bins_check(
                        n_cal=len(u_cal_saved), delta=float(args.calib_delta),
                        bins=_bins_cached, min_bin=_min_bin_cached,
                        q_eps=float(args.q_eps),
                        eps_raw=eps,
                        uncertainty_type="USAD(cached)",
                        calib_frac=float(args.calib_frac), ds=ds,
                    )
            else:
                _delta = float(args.calib_delta)
                _n_cal = len(D_calib["obs"])
                # With the triple split the per-bin bounds are fitted on
                # D_wilks, about N_cal / 3 samples.
                _n_fit = (_n_cal // 3) if int(args.calib_triple_split) else _n_cal
                # fit_calibrator selects B over the candidate set with the
                # one-standard-error rule. --calib_bins 0 uses the default set (or
                # --calib_candidate_bins); a positive value fixes a single B.
                _bins = int(args.calib_bins)
                # min_bin_size applies to the single-split path only;
                # fit_with_selection uses 1 (no bin merging).
                _min_bin = (
                    wilks_min_n_per_bin(_bins, _delta)
                    if int(args.calib_min_samples) == 0
                    else int(args.calib_min_samples)
                )
                cal_cfg = UncertaintyCalibratorConfig(
                    n_bins=_bins,
                    delta=_delta,
                    min_bin_size=_min_bin,
                    mode=str(args.calibrator_mode),
                    alpha=float(args.calibrator_alpha),
                )
                _cand_bins_usad = _parse_candidate_bins(args.calib_candidate_bins)
                _single_B_usad = int(args.calib_bins) if int(args.calib_bins) > 0 else None
                calibrator, eps_raw_usad, u_cal, e_cal = fit_calibrator(
                    ens, D_calib, cal_cfg, uncertainty_type="usad",
                    q_eps=float(args.q_eps),
                    triple_split=bool(int(args.calib_triple_split)),
                    candidate_bins=_cand_bins_usad,
                    single_B_override=_single_B_usad,
                    n_bootstrap=int(args.calib_n_bootstrap),
                    random_seed=int(seed),
                    legacy_mode=bool(int(args.calib_legacy_mode)),
                )
                # Min-bins check on the selected B_dagger. With candidate selection
                # the collapse is already in eps_raw_usad (-inf when no candidate
                # is admissible); the check then only prints the summary.
                _bins_eff = int(getattr(calibrator, "meta", {}).get("B_dagger", len(calibrator.bin_bounds)) or len(calibrator.bin_bounds))
                eps = calib_min_bins_check(
                    n_cal=_n_cal, delta=_delta, bins=_bins_eff, min_bin=_min_bin,
                    q_eps=float(args.q_eps),
                    eps_raw=eps_raw_usad,
                    uncertainty_type="USAD",
                    calib_frac=float(args.calib_frac), ds=ds,
                )
                with open(cal_path, "wb") as f:
                    pickle.dump((calibrator, eps), f)
                np.savez(
                    os.path.join(outdir, "calibration_data_usad.npz"),
                    u=u_cal, e=e_cal,
                )

            # ── Calibrator (epistemic variance, for calibrated_variance) ──
            cal_var_path = os.path.join(outdir, "calibrator_epistemic.pkl")
            if os.path.exists(cal_var_path) and not args.force_retrain:
                with open(cal_var_path, "rb") as f:
                    calibrator_var, eps_var = pickle.load(f)
                print(f"[BENCH] Loaded epistemic calibrator (eps_var={eps_var:.4f})", flush=True)
                # Re-check the Wilks and min-bins conditions on the saved u_cal.
                npz_var_path = os.path.join(outdir, "calibration_data_epistemic.npz")
                if os.path.exists(npz_var_path) and np.isfinite(eps_var):
                    u_cal_var_saved = np.load(npz_var_path)["u"]
                    eps_var, wilks_ok_v, n_b_min_v, wilks_thr_v = wilks_eps_check(
                        u_cal_var_saved, calibrator_var, eps_var,
                        float(args.calib_delta), 0.05,
                    )
                    if not wilks_ok_v:
                        print(
                            f"[CALIB] Wilks FAIL (post-load epistemic): n_b_min={n_b_min_v} < {wilks_thr_v} "
                            f"-> eps_var=-inf (conservative collapse)",
                            flush=True,
                        )
                    _bins_cached_v = len(calibrator_var.bin_bounds)
                    _min_bin_cached_v = wilks_min_n_per_bin(_bins_cached_v, float(args.calib_delta))
                    eps_var = calib_min_bins_check(
                        n_cal=len(u_cal_var_saved), delta=float(args.calib_delta),
                        bins=_bins_cached_v, min_bin=_min_bin_cached_v,
                        q_eps=float(args.q_eps),
                        eps_raw=eps_var,
                        uncertainty_type="EpistemicVar(cached)",
                        calib_frac=float(args.calib_frac), ds=ds,
                    )
            else:
                _delta_v = float(args.calib_delta)
                _n_cal_v = len(D_calib["obs"])
                # Same B selection and bin sizing as the USAD branch.
                _n_fit_v = (_n_cal_v // 3) if int(args.calib_triple_split) else _n_cal_v
                _bins_v = int(args.calib_bins)
                _min_bin_v = (
                    wilks_min_n_per_bin(_bins_v, _delta_v)
                    if int(args.calib_min_samples) == 0
                    else int(args.calib_min_samples)
                )
                cal_cfg_var = UncertaintyCalibratorConfig(
                    n_bins=_bins_v,
                    delta=_delta_v,
                    min_bin_size=_min_bin_v,
                    mode=str(args.calibrator_mode),
                    alpha=float(args.calibrator_alpha),
                )
                _cand_bins_var = _parse_candidate_bins(args.calib_candidate_bins)
                _single_B_var = int(args.calib_bins) if int(args.calib_bins) > 0 else None
                calibrator_var, eps_raw_var, u_cal_var, e_cal_var = fit_calibrator(
                    ens, D_calib, cal_cfg_var, uncertainty_type="epistemic",
                    q_eps=float(args.q_eps),
                    triple_split=bool(int(args.calib_triple_split)),
                    candidate_bins=_cand_bins_var,
                    single_B_override=_single_B_var,
                    n_bootstrap=int(args.calib_n_bootstrap),
                    random_seed=int(seed),
                    legacy_mode=bool(int(args.calib_legacy_mode)),
                )
                _bins_eff_v = int(getattr(calibrator_var, "meta", {}).get("B_dagger", len(calibrator_var.bin_bounds)) or len(calibrator_var.bin_bounds))
                eps_var = calib_min_bins_check(
                    n_cal=_n_cal_v, delta=_delta_v, bins=_bins_eff_v, min_bin=_min_bin_v,
                    q_eps=float(args.q_eps),
                    eps_raw=eps_raw_var,
                    uncertainty_type="EpistemicVar",
                    calib_frac=float(args.calib_frac), ds=ds,
                )
                with open(cal_var_path, "wb") as f:
                    pickle.dump((calibrator_var, eps_var), f)
                np.savez(
                    os.path.join(outdir, "calibration_data_epistemic.npz"),
                    u=u_cal_var, e=e_cal_var,
                )

            # ── Config for model env / MCTS ───────────────────────────────
            model_cfg_mcts = ModelContinuousEnvConfig.for_domain(args.domain)
            model_cfg_mcts.truncate_on_time_limit = False  # rollouts end at max_depth
            model_cfg_mcts.max_steps = int(args.max_depth)

            mcts_cfg = OfflineMCTSConfig(
                num_sims=int(args.num_sims),
                max_depth=int(args.max_depth),
                gamma=float(args.gamma),
                dpw_k=float(args.dpw_k),
                dpw_alpha=float(args.dpw_alpha),
                c_puct=float(args.c_puct),
                ppo_std_scale=float(args.ppo_std_scale),
                n_actions_root=int(args.n_actions_root),
                n_actions_per_node=int(args.n_actions_per_node),
                leaf_value_mode="zero",
                gate_inside_tree=True,
            )

            # ── Load partial results for this (ds, seed) outdir ──────────────
            completed_methods = _load_completed_methods(outdir)
            if completed_methods:
                n_done = len(completed_methods)
                print(f"[CKPT] Found {n_done} completed method(s) in {outdir}: "
                      f"{list(completed_methods.keys())}", flush=True)

            def row_base(eps_used: Optional[float] = None):
                """Fields shared by every result row of this cell."""
                return {
                    "domain": args.domain,
                    "seed": seed,
                    "ds": ds,
                    "n_train": len(D_train["obs"]),
                    "n_calib": len(D_calib["obs"]),
                    "eps": float(eps_used) if eps_used is not None else float(eps),
                }

            def run_eval(
                method_name: str,
                act_fn,
                eps_used: Optional[float] = None,
                planner_obj: Optional[Any] = None,
            ) -> None:
                """Evaluate act_fn as method_name unless skipped or already completed."""
                if method_name not in should_run:
                    return
                if method_name in completed_methods:
                    print(f"[CKPT] {method_name} already done — loading from checkpoint", flush=True)
                    rows.append(completed_methods[method_name])
                    return
                print(f"[BENCH] Evaluating {method_name}...", flush=True)
                # Reset the switch statistics so the row covers this evaluation only.
                if planner_obj is not None and hasattr(planner_obj, "cspi_reset_stats"):
                    planner_obj.cspi_reset_stats()
                ckpt = _EpisodeCheckpoint(outdir, method_name)
                metrics = _eval_policy_episodic(
                    env_eval, act_fn,
                    n_episodes=int(args.eval_episodes),
                    seed=int(seed),
                    max_steps=int(max_steps),
                    gamma=float(args.gamma),
                    ckpt=ckpt,
                    env_factory=_eval_env_factory,
                    n_workers=int(args.eval_workers),
                    seed_stride=int(getattr(args, 'eval_seed_stride', 1)),
                )
                row = {**row_base(eps_used), "method": method_name, **metrics}
                if planner_obj is not None and hasattr(planner_obj, "cspi_stats"):
                    try:
                        row["cspi_stats"] = dict(planner_obj.cspi_stats())
                    except Exception as _e:
                        row["cspi_stats"] = {"error": str(_e)}
                rows.append(row)
                _save_partial_result(outdir, row)
                cspi_tail = ""
                if "cspi_stats" in row and isinstance(row["cspi_stats"], dict):
                    s = row["cspi_stats"]
                    cspi_tail = (
                        f" [CSPI] switch={s.get('n_switch', 0)}/"
                        f"{s.get('n_calls', 0)} "
                        f"fb_gate={s.get('n_fallback_gate', 0)} "
                        f"fb_margin={s.get('n_fallback_margin', 0)} "
                        f"root_safe_avg={s.get('root_safe_actions_avg', 0.0):.1f} "
                        f"root_visits_min={s.get('root_visits_min_over_calls', 0)} "
                        f"margin_avg={s.get('margin_avg_on_switch', 0.0):.3f}"
                    )
                print(
                    f"  [RESULT] {method_name}: "
                    f"return={metrics['return_mean']:.2f}±{metrics['return_std']:.2f} "
                    f"survival={metrics['success_rate']:.2f} "
                    f"xvel={metrics['xvel_mean']:.3f} "
                    f"coverage={metrics['root_coverage_mean']:.3f} "
                    f"fallback={metrics['root_fallback_rate']:.3f}"
                    + cspi_tail,
                    flush=True,
                )

            # ── PPO baseline ──────────────────────────────────────────────
            run_eval("ppo_only", ppo_act_fn)

            # ── Cal-SPI (calibrated USAD) ─────────────────────────────────
            # Switches from the baseline action a_b to a* only when a_b is
            # gate-certified and Q_hat_safe(a*) - V_hat_b(s_root) >= Delta_margin.
            if "calibrated_usad" in should_run:
                import copy
                mcts_cal = make_mcts_calibrated(
                    ens, calibrator, eps, ppo,
                    copy.copy(model_cfg_mcts), copy.copy(mcts_cfg), seed,
                    backend=str(args.mcts_backend),
                    cspi_margin_mult=float(args.cspi_margin_mult),
                    cspi_v_rollout_horizon=int(args.cspi_v_rollout_horizon),
                    cspi_v_rollout_n_samples=int(args.cspi_v_rollout_n_samples),
                    cspi_e_plan_scale=float(args.cspi_e_plan_scale),
                    cspi_delta_piB=float(args.cspi_delta_piB),
                    cspi_frozen_root=bool(int(args.cspi_frozen_root)),
                    cspi_n_min_root_visits=int(args.cspi_n_min_root_visits),
                    use_ppo_leaf_value=bool(int(args.use_ppo_leaf_value)),
                )
                run_eval("calibrated_usad", mcts_cal.act, eps_used=float(eps),
                         planner_obj=mcts_cal)

            # ── Cal-SPI (calibrated epistemic variance) ───────────────────
            if "calibrated_variance" in should_run:
                import copy
                mcts_cal_var = make_mcts_calibrated_var(
                    ens, calibrator_var, eps_var, ppo, copy.copy(model_cfg_mcts), copy.copy(mcts_cfg), seed,
                    backend=str(args.mcts_backend),
                    cspi_margin_mult=float(args.cspi_margin_mult),
                    cspi_v_rollout_horizon=int(args.cspi_v_rollout_horizon),
                    cspi_v_rollout_n_samples=int(args.cspi_v_rollout_n_samples),
                    cspi_e_plan_scale=float(args.cspi_e_plan_scale),
                    cspi_delta_piB=float(args.cspi_delta_piB),
                    cspi_frozen_root=bool(int(args.cspi_frozen_root)),
                    cspi_n_min_root_visits=int(args.cspi_n_min_root_visits),
                )
                run_eval("calibrated_variance", mcts_cal_var.act, eps_used=float(eps_var),
                         planner_obj=mcts_cal_var)

            # ── Offline RL baselines ──────────────────────────────────────
            baseline_epochs = int(args.baseline_epochs)

            def run_baseline(method_name: str, agent, agent_act_fn) -> None:
                """Train (or load from cache) a baseline agent and evaluate it."""
                if method_name not in should_run:
                    return
                if method_name in completed_methods:
                    print(f"[CKPT] {method_name} already done — loading from checkpoint", flush=True)
                    rows.append(completed_methods[method_name])
                    return
                if not _load_baseline_model(outdir, method_name, agent):
                    print(f"[BENCH] Training baseline {method_name} ({baseline_epochs} steps)...",
                          flush=True)
                    agent.train(_baseline_ds, n_epochs=baseline_epochs)
                    _save_baseline_model(outdir, method_name, agent)
                else:
                    print(f"[BENCH] Evaluating baseline {method_name} (model loaded from cache)...",
                          flush=True)
                ckpt = _EpisodeCheckpoint(outdir, method_name)
                metrics = _eval_policy_episodic(
                    env_eval, agent_act_fn,
                    n_episodes=int(args.eval_episodes),
                    seed=int(seed),
                    max_steps=int(max_steps),
                    gamma=float(args.gamma),
                    ckpt=ckpt,
                    env_factory=_eval_env_factory,
                    n_workers=int(args.eval_workers),
                    seed_stride=int(getattr(args, 'eval_seed_stride', 1)),
                )
                row = {**row_base(), "method": method_name, **metrics}
                rows.append(row)
                _save_partial_result(outdir, row)
                print(
                    f"  [RESULT] {method_name}: "
                    f"return={metrics['return_mean']:.2f}±{metrics['return_std']:.2f} "
                    f"survival={metrics['success_rate']:.2f}",
                    flush=True,
                )

            # Dataset format shared by the baseline agents. The goal arrays have
            # shape (N, 0), so concatenating them to the observation is a no-op.
            _n = len(D_train["obs"])
            _goal_zeros = np.zeros((_n, 0), dtype=np.float32)
            _baseline_ds = {
                "obs_observation":  D_train["obs"],
                "obs_desired_goal": _goal_zeros,
                "act":              D_train["actions"],
                "rew":              D_train["rewards"],
                "next_observation": D_train["obs_next"],
                "next_desired_goal": _goal_zeros,
                "done":             D_train["dones"].astype(np.float32),
            }

            _rb = min(1000, len(D_train["obs"]))  # model rollout batch size

            # IQL
            if "iql" in should_run:
                iql_agent = IQLAgent(obs_dim, act_dim, cfg=make_iql_cfg(args.domain, seed))
                run_baseline("iql", iql_agent, iql_agent.act)

            # TD3+BC
            if "td3bc" in should_run:
                td3_agent = TD3BCAgent(obs_dim, act_dim, cfg=make_td3bc_cfg(args.domain, seed))
                run_baseline("td3bc", td3_agent, td3_agent.act)

            # MOReL-full
            if "morel_full" in should_run:
                mrl_agent = MORelFullAgent(
                    ens, obs_dim, act_dim,
                    cfg=make_morel_cfg(args.domain, seed, _rb),
                    reward_fn=make_morel_reward_fn(args.domain),
                )
                run_baseline("morel_full", mrl_agent, mrl_agent.act)

            # MOPO-MSE
            if "mopo_mse" in should_run:
                mopo_agent = MOPOMSEAgent(
                    ens, obs_dim, act_dim,
                    cfg=make_mopo_mse_cfg(args.domain, seed, _rb),
                    reward_fn=make_morel_reward_fn(args.domain),
                )
                run_baseline("mopo_mse", mopo_agent, mopo_agent.act)

            # CQL
            if "cql" in should_run:
                cql_agent = CQLAgent(obs_dim, act_dim, cfg=make_cql_cfg(args.domain, seed))
                run_baseline("cql", cql_agent, cql_agent.act)

            # BEAR
            if "bear" in should_run:
                bear_agent = BEARAgent(obs_dim, act_dim, cfg=make_bear_cfg(args.domain, seed))
                run_baseline("bear", bear_agent, bear_agent.act)

            # Cal-QL
            if "cal_ql" in should_run:
                cal_ql_agent = CalQLAgent(obs_dim, act_dim, cfg=make_cal_ql_cfg(args.domain, seed))
                run_baseline("cal_ql", cal_ql_agent, cal_ql_agent.act)

            # ── MCTS without gate (reference planner) ─────────────────────
            if "mcts_only_no_gate" in should_run:
                import copy
                mcts_cfg_nogate = copy.copy(mcts_cfg)
                # Budget: --nogate_num_sims if positive, else min(200, num_sims).
                _ng = int(getattr(args, 'nogate_num_sims', 0))
                mcts_cfg_nogate.num_sims = (_ng if _ng > 0
                                            else min(200, int(mcts_cfg.num_sims)))
                mcts_no_gate = make_mcts_no_gate(ens, ppo, copy.copy(model_cfg_mcts), mcts_cfg_nogate, seed)
                run_eval("mcts_only_no_gate", mcts_no_gate.act)

        env_data.close()
        env_eval.close()

    # ── Save results ───────────────────────────────────────────────────────
    if rows:
        df = pd.DataFrame(rows)
        out_csv = os.path.join(args.outdir, "results.csv")
        df.to_csv(out_csv, index=False)
        print(f"\n[BENCH] Results saved → {out_csv}", flush=True)
        print(df.to_string(max_rows=60), flush=True)
    else:
        print("[BENCH] No results produced.", flush=True)


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def parse_args():
    """Parse the command-line arguments."""
    ap = argparse.ArgumentParser(description="Cal-SPI benchmark on MuJoCo locomotion domains")

    # Domain
    ap.add_argument("--domain", type=str, required=True,
                    choices=sorted(DOMAIN_CONFIGS),
                    help="MuJoCo domain")

    # Output
    ap.add_argument("--outdir", type=str, default="results_continuous",
                    help="Root output directory")

    # PPO
    ap.add_argument("--ppo_global_path", type=str, required=True,
                    help="Path to pre-trained PPO checkpoint (.zip)")
    ap.add_argument("--ppo_timesteps", type=int, default=70_000,
                    help="PPO training timesteps if the checkpoint is missing")

    # Dataset
    ap.add_argument("--dataset_trajs", type=int, nargs="+", default=[5, 50, 500, 1000])
    ap.add_argument("--seeds",          type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--max_steps",      type=int, default=1000,
                    help="Max steps per episode")

    # MCTS and evaluation
    ap.add_argument("--num_sims",  type=int,   default=1000)
    ap.add_argument("--max_depth", type=int,   default=10)
    ap.add_argument("--gamma",     type=float, default=0.99)
    ap.add_argument("--eval_episodes", type=int, default=20)
    ap.add_argument("--eval_seed_stride", type=int, default=1,
                    help="Environment seed of episode i is seed*stride + i. With 1, "
                         "adjacent seeds share most initial states; a value >= "
                         "--eval_episodes gives every cell a disjoint range. All methods "
                         "of a cell see the same initial states.")
    ap.add_argument("--mcts_backend", type=str, default="array",
                    choices=["original", "batched", "numba", "numba_scaled", "array"],
                    help="MCTS planner backend: 'original' (per-sample torch forward), "
                         "'batched' (per-node batched torch forward, same actions as "
                         "'original' under fixed_per_sim sampling), 'numba' (compiled "
                         "ensemble and PPO forward; same algorithm, equal up to float32 "
                         "rounding), 'numba_scaled' (numba with the scale-aware gate), "
                         "'array' (compiled search over preallocated buffers, same "
                         "actions as 'numba').")
    ap.add_argument("--eval_workers", type=int, default=1,
                    help="Fork-based workers for episodic evaluation (1 = sequential)")

    # Calibration
    ap.add_argument("--calib_frac",        type=float, default=0.2)
    ap.add_argument("--calib_bins",        type=int,   default=0,
                    help="Fixed bin count B (0 = select B from --calib_candidate_bins "
                         "or the default set)")
    ap.add_argument("--calib_candidate_bins", type=str, default="",
                    help=(
                        "Candidate bin counts for one-standard-error selection, comma-separated "
                        "(e.g. '20,40,80'). Empty = [B_min, 2 B_min, 3 B_min, 4 B_min] with "
                        "B_min=ceil(1/(1-q_eps)). Per-bin confidence alpha_eff = alpha/(|cand|*B)."
                    ))
    ap.add_argument("--calib_n_bootstrap", type=int, default=200,
                    help="Bootstrap resamples for the standard error of eps_B in the "
                         "one-standard-error rule (0 = plain argmin).")
    ap.add_argument("--calib_legacy_mode", type=int, default=0,
                    help="0 = candidate selection with the one-standard-error rule; "
                         "1 = single B = wilks_optimal_bins(|D_wilks|), mode='quantile', "
                         "alpha_eff=alpha/B, no correction over candidates.")
    ap.add_argument("--calib_delta",       type=float, default=0.05)
    ap.add_argument("--calib_min_samples", type=int,   default=0,
                    help="min_bin_size of the single-split fit (0 = Wilks n_b for the chosen B)")
    ap.add_argument("--q_eps",             type=float, default=0.95,
                    help=(
                        "Quantile level of the gate threshold: eps=quantile_{q_eps}(B(u_calib)). "
                        "Also sets B_min=ceil(1/(1-q_eps)) for the candidate set "
                        "(q_eps=0.95 gives B_min=20)."
                    ))

    # Offline RL baselines: gradient updates per agent
    ap.add_argument("--baseline_epochs", type=int, default=1_000_000)

    # Methods to run
    ap.add_argument(
        "--run_methods", type=str, nargs="+",
        default=["ppo_only", "mcts_only_no_gate", "calibrated_usad"],
        help="Methods to evaluate",
    )
    ap.add_argument("--nogate_num_sims", type=int, default=0,
                    help="Simulation budget for mcts_only_no_gate "
                         "(0 = min(200, num_sims)).")

    # Misc
    ap.add_argument("--force_retrain", action="store_true",
                    help="Retrain ensemble/calibrator even if cached")
    # ── MCTS / APW hyperparameters ─────────────────────────────────────────
    ap.add_argument("--dpw_k",            type=float, default=2.0,
                    help="APW widening constant k: max_children=floor(k*N^alpha)")
    ap.add_argument("--dpw_alpha",        type=float, default=0.5,
                    help="APW exponent alpha")
    ap.add_argument("--c_puct",           type=float, default=1.5,
                    help="PUCT exploration constant")
    ap.add_argument("--n_actions_root",   type=int,   default=96,
                    help="Maximum number of actions at the root")
    ap.add_argument("--ppo_std_scale",    type=float, default=0.1,
                    help="Std of the Gaussian around the PPO mean from which new actions "
                         "are sampled")
    ap.add_argument("--n_actions_per_node", type=int, default=24,
                    help="Maximum number of actions at non-root nodes")
    ap.add_argument("--ensemble_epochs",    type=int,  default=200,
                    help="Ensemble training epochs")
    # Baseline-relative switch
    ap.add_argument("--cspi_margin_mult", type=float, default=1.0,
                    help="Multiplier on Delta_margin in the baseline-relative switch "
                         "(0 = plain argmax over A_safe).")
    ap.add_argument("--cspi_v_rollout_horizon", type=int, default=0,
                    help="Model steps of the pi_b rollout that estimates V_b(s_root) "
                         "(0 = max_depth, the horizon of the Q estimates).")
    ap.add_argument("--cspi_v_rollout_n_samples", type=int, default=1,
                    help="Number of pi_b rollouts averaged to estimate V_b(s_root)")
    ap.add_argument("--cspi_e_plan_scale", type=float, default=0.5,
                    help="E_plan term of the margin, as a fraction of the model-error "
                         "radius (heuristic bound on the finite-budget planning error).")
    ap.add_argument("--cspi_delta_piB", type=float, default=0.0,
                    help="Additive slack delta_piB in the margin for the suboptimality "
                         "of pi_b (0 = near-optimal baseline assumption).")
    ap.add_argument("--cspi_frozen_root", type=int, default=1,
                    help="Frozen-root regime (1=on, 0=off): the n_actions_root root "
                         "candidates, a_b included, are sampled once at the start of "
                         "act(), their safe subset is fixed, and no root actions are "
                         "added during search. The finite-budget planner bound assumes it.")
    ap.add_argument("--cspi_n_min_root_visits", type=int, default=3,
                    help="Minimum visits per safe root action before PUCT selection "
                         "applies (frozen-root regime only).")
    ap.add_argument("--use_ppo_leaf_value", type=int, default=0,
                    help="0: leaf value 0. 1: bootstrap the leaf with V_PPO(s_leaf).")
    ap.add_argument("--calibrator_mode", type=str, default="quantile",
                    choices=["quantile", "wilks_strict"],
                    help="Per-bin bound: 'quantile' = empirical (1-delta)-quantile; "
                         "'wilks_strict' = order-statistic upper tolerance limit with "
                         "per-bin confidence 1-alpha/B.")
    ap.add_argument("--calibrator_alpha", type=float, default=0.05,
                    help="Overall Wilks confidence level alpha for wilks_strict; "
                         "split as alpha/B across bins")
    ap.add_argument("--calib_triple_split", type=int, default=1,
                    choices=[0, 1],
                    help="1: fit bin edges, per-bin Wilks bounds and eps on the disjoint "
                         "blocks D_bin/D_wilks/D_thr, as the coverage guarantee requires. "
                         "0: fit all three on the full calibration sample.")
    ap.add_argument("--ensemble_normalize", type=int,  default=0,
                    help="Standardize (s,a) inputs of the ensemble (1=on)")
    ap.add_argument("--ensemble_bootstrap", type=int,  default=0,
                    help="Train each ensemble member on a bootstrap resample (1=on)")
    ap.add_argument("--uncertainty_scale_aware", type=int, default=0,
                    choices=[0, 1],
                    help="Divide USAD and the calibration error by the per-dimension "
                         "delta scale, so every state component weighs equally in the "
                         "calibrated bound. eps is then in scaled units.")
    ap.add_argument("--ensemble_target_normalize", type=int, default=0,
                    choices=[0, 1],
                    help="Standardize the regression target per dimension during "
                         "ensemble training; the de-normalization is folded into the "
                         "output layer, so the model still predicts raw deltas.")
    ap.add_argument("--ensemble_size",      type=int,  default=4,
                    help="Number of ensemble members K")
    ap.add_argument("--ensemble_hidden_dims", type=str, default="512,512",
                    help="Comma-separated hidden layer sizes of each ensemble member")

    return ap.parse_args()


def _log_config(args) -> None:
    """Print every argument as a sorted key=value list on one [CONFIG] line."""
    print("[CONFIG] " + " ".join(
        f"{k}={v}" for k, v in sorted(vars(args).items())), flush=True)


if __name__ == "__main__":
    args = parse_args()
    _log_config(args)
    run_benchmark(args)
