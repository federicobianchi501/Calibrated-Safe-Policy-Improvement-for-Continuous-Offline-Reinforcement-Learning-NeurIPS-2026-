#!/usr/bin/env python3
# =============================================================================
# Anonymous authors. Affiliations and acknowledgements will be added on
# acceptance of the associated paper. Submitted to NeurIPS 2026 (double-blind).
# =============================================================================
"""
Point Maze benchmark driver.

Pipeline per (seed, dataset size):
  - Train one global PPO behavior policy (single checkpoint, reused across cells).
  - Evaluate the PPO baseline once per seed (flat vs. dataset size).
  - Sample max_ds trajectory seeds per random seed; for each dataset size ds,
    regenerate rollouts on traj_seeds[:ds] so smaller datasets are strict
    prefixes of larger ones.
  - Split each dataset into train / calibration by trajectory, with the
    calibration ids fixed once per seed (computed on max_ds) and intersected
    with the current ds.
  - Train one ensemble and one calibrator per ds; plan with one MCTS-DPW
    instance per method.
  - Evaluate each planner on the true environment with a seed that does not
    depend on ds.
  - The model env used inside MCTS sets truncate_on_time_limit=False so the
    effective planning horizon is exactly max_depth.

The "mcts_unsafe_model_only" method runs MCTS-DPW on the learned model with
no calibrated gate, no PPO prior, no PPO value, and no fallback; it isolates
unconstrained model exploitation.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import torch
import json
import pickle
import hashlib
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from safe_rl.envs.pointmaze_factory import make_pointmaze_env
from safe_rl.offline.model_env_pointmaze import ModelPointMazeEnv, ModelEnvConfig
from safe_rl.offline.ensemble_dynamics_sb3 import EnsembleDynamicsSB3, EnsembleTrainConfig
from safe_rl.offline.uncertainty_calibration import UncertaintyCalibrator, UncertaintyCalibratorConfig
from safe_rl.algorithms.offline.cal_spi.mcts_parametric import (
    OfflineMCTSConfig,
    OfflineMCTS_DPW_PPO_SPIBB_Parametric,
    OfflineMCTS_DPW_Ensemble_UNSAFE,
)
from safe_rl.algorithms.offline.iql import IQLAgent, IQLConfig
from safe_rl.algorithms.offline.td3bc import TD3BCAgent, TD3BCConfig
from safe_rl.algorithms.offline.morel_full import MORelFullAgent, MORelFullConfig
from safe_rl.algorithms.offline.mopo import MOPOMSEAgent, MOPOMSEConfig
from safe_rl.algorithms.offline.cql import CQLAgent, CQLConfig
from safe_rl.algorithms.offline.bear import BEARAgent, BEARConfig
from safe_rl.algorithms.offline.cal_ql import CalQLAgent, CalQLConfig


# --- Debug visualizer (optional; skipped gracefully when not found) ---
try:
    from mcts_debug_visualizer import (
        LoggingMCTS,
        run_debug_episode,
        plot_debug_episode,
    )
    _DEBUG_VIZ_AVAILABLE = True
except ImportError:
    _DEBUG_VIZ_AVAILABLE = False
    print("[DEBUG VIZ] mcts_debug_visualizer not found — debug plots disabled.", flush=True)

# -------------------------
# Helpers
# -------------------------

def rollout_trajectory(env, act_fn, *, seed: int, max_steps: int):
    """
    Run one episode and store full 2D trajectory:
      - achieved_goal positions
      - desired_goal
      - start position
      - success / termination info
    """
    obs, info = env.reset(seed=int(seed))

    pts = []
    rewards = []
    dones = []
    infos = []

    start_ag = np.asarray(obs["achieved_goal"], dtype=np.float32).copy()
    goal = np.asarray(obs["desired_goal"], dtype=np.float32).copy()

    # include initial point
    pts.append(start_ag.copy())

    for t in range(int(max_steps)):
        a = act_fn(obs)
        obs, r, terminated, truncated, info = env.step(a)

        ag = np.asarray(obs["achieved_goal"], dtype=np.float32).copy()
        pts.append(ag)

        rewards.append(float(r))
        dones.append(bool(terminated or truncated))
        infos.append(dict(info) if isinstance(info, dict) else {})

        if terminated or truncated:
            break

    pts = np.asarray(pts, dtype=np.float32)
    return {
        "xy": pts,                       # shape [T+1, 2]
        "start": start_ag,
        "goal": goal,
        "rewards": np.asarray(rewards, dtype=np.float32),
        "done_flags": np.asarray(dones, dtype=np.int32),
        "n_steps": int(len(rewards)),
        "success": bool(infos[-1].get("success", False)) if len(infos) > 0 else False,
        "terminated": bool(dones[-1]) if len(dones) > 0 else False,
    }


def save_traj_npz(path: str, traj: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))

    payload = {
        "xy": np.asarray(traj["xy"], dtype=np.float32),
        "start": np.asarray(traj["start"], dtype=np.float32),
        "goal": np.asarray(traj["goal"], dtype=np.float32),
        "n_steps": np.asarray([int(traj.get("n_steps", 0))], dtype=np.int32),
        "success": np.asarray([float(traj.get("success", 0.0))], dtype=np.float32),
    }

    # Optional fields: save only if present
    if "rewards" in traj:
        payload["rewards"] = np.asarray(traj["rewards"], dtype=np.float32)
    if "done_flags" in traj:
        payload["done_flags"] = np.asarray(traj["done_flags"], dtype=np.int32)
    if "terminated" in traj:
        payload["terminated"] = np.asarray([int(traj["terminated"])], dtype=np.int32)
    if "return" in traj:
        payload["return"] = np.asarray([float(traj["return"])], dtype=np.float32)
    if "discounted_return" in traj:
        payload["discounted_return"] = np.asarray([float(traj["discounted_return"])], dtype=np.float32)
    if "return_wo_lc" in traj:
        payload["return_wo_lc"] = np.asarray([float(traj["return_wo_lc"])], dtype=np.float32)
    if "time_to_success" in traj:
        tts = traj["time_to_success"]
        payload["time_to_success"] = np.asarray(
            [float(tts) if tts is not None else np.nan], dtype=np.float32
        )

    np.savez_compressed(path, **payload)


def plot_trajectories_2d(trajs_by_method: Dict[str, List[Dict[str, Any]]], outpath: str, title: str = ""):
    """
    One 2D plot:
      - target as star
      - start as black circle
      - one or more trajectories per method
    """
    ensure_dir(os.path.dirname(outpath))

    style = {
        "ppo_only": dict(label="Baseline", color="gold"),
        "mcts_only_no_gate": dict(label="MCTS-DPW", color="blue"),
        "morel_usad": dict(label="SPI-U", color="purple"),
        "calibrated_usad": dict(label="CSPI-U", color="green"),
        "calibrated_variance": dict(label="CSPI-V", color="red"),
    }

    plt.figure(figsize=(7, 7))
    ax = plt.gca()

    plotted_start = False
    plotted_goal = False

    for method, trajs in trajs_by_method.items():
        if len(trajs) == 0:
            continue
        st = style.get(method, dict(label=method, color=None))

        for k, traj in enumerate(trajs):
            xy = np.asarray(traj["xy"], dtype=np.float32)
            start = np.asarray(traj["start"], dtype=np.float32)
            goal = np.asarray(traj["goal"], dtype=np.float32)

            ax.plot(
                xy[:, 0],
                xy[:, 1],
                linewidth=2.0,
                alpha=0.9,
                color=st["color"],
                label=st["label"] if k == 0 else None,
            )

            # mark final point
            ax.scatter(
                xy[-1, 0],
                xy[-1, 1],
                s=30,
                color=st["color"],
                alpha=0.9,
            )

            if not plotted_start:
                ax.scatter(
                    start[0], start[1],
                    s=80,
                    color="black",
                    marker="o",
                    label="Start",
                    zorder=5,
                )
                plotted_start = True

            if not plotted_goal:
                ax.scatter(
                    goal[0], goal[1],
                    s=160,
                    color="orange",
                    marker="*",
                    label="Target",
                    zorder=6,
                )
                plotted_goal = True

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _jsonable(x):
    # Convert numpy / torch / dataclass to json-friendly
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
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _to_f32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)

def _dbg_stats(name: str, x: np.ndarray) -> None:
    x = np.asarray(x)
    flat = x.reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        print(f"[CALIB DBG] {name}: no finite values. shape={x.shape} dtype={x.dtype}", flush=True)
        return
    q0, q1, q50, q99, q100 = np.quantile(finite, [0.0, 0.01, 0.5, 0.99, 1.0])
    print(
        f"[CALIB DBG] {name}: shape={x.shape} "
        f"min={q0:.6g} p01={q1:.6g} med={q50:.6g} p99={q99:.6g} max={q100:.6g} "
        f"mean={finite.mean():.6g} std={finite.std():.6g}",
        flush=True,
    )

def _dbg_shape(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray) -> None:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        print(f"[CALIB DBG] SHAPE MISMATCH: {name_a}.shape={a.shape} vs {name_b}.shape={b.shape}", flush=True)

def make_env(continuing_task: int, reward_type: str, seed: int, living_cost: float = 0.0, success_bonus: float = 0.0, reward_mode: str = "current", step_penalty: float = 1.0, dist_penalty: float = 0.2):
    return make_pointmaze_env(
        int(seed),
        continuing_task=bool(int(continuing_task)),
        reward_type=str(reward_type),
        living_cost=float(living_cost),
        success_bonus=float(success_bonus),
        reward_mode=str(reward_mode),
        step_penalty=float(step_penalty),
        dist_penalty=float(dist_penalty),
    )

def run_ppo_training_to_path(
    *,
    train_ppo_script: str,
    seed: int,
    total_timesteps: int,
    continuing_task: int,
    reward_type: str,
    out_path: str,
    reward_mode: str,
    step_penalty: float,
    dist_penalty: float,
    success_bonus: float,
    living_cost: float,
) -> str:
    ensure_dir(os.path.dirname(out_path) or ".")
    cmd = [
        sys.executable,
        train_ppo_script,
        "--seed", str(int(seed)),
        "--out", str(out_path),
        "--continuing_task", str(int(continuing_task)),
        "--reward_type", str(reward_type),
        "--reward_mode", str(reward_mode),
        "--living_cost", str(float(living_cost)),
        "--success_bonus", str(float(success_bonus)),
        "--step_penalty", str(float(step_penalty)),
        "--dist_penalty", str(float(dist_penalty)),
        "--total_timesteps", str(int(total_timesteps)),
    ]
    print(f"[PPO TRAIN-ONCE] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    if not os.path.exists(out_path):
        raise RuntimeError(f"PPO checkpoint not found after training: {out_path}")
    return out_path

# -------------------------
# Metrics
# -------------------------

import numpy as np

def cvar_lower_tail(x: np.ndarray, alpha: float = 0.15) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    q = float(np.quantile(x, alpha))
    m = x <= q
    return float(np.mean(x[m])) if np.any(m) else q

def _discounted_add(ret_d: float, r: float, gamma: float, t: int) -> float:
    return float(ret_d + (float(gamma) ** int(t)) * float(r))


# ---------------------------------------------------------------------------
# Episode-level checkpoint utilities (power-outage safe resume)
# ---------------------------------------------------------------------------
# Files written per outdir (e.g. results_pointmaze/dsN/seedK/):
#   ep_ckpt/<method>.jsonl   — one JSON line per completed episode (append-only)
#   results_partial.jsonl    — one JSON line per completed method (append-only)
#   models/<method>.pt       — trained baseline model weights

_PM_PARTIAL_FILE      = "results_partial.jsonl"
_PM_EP_CKPT_DIR       = "ep_ckpt"
_PM_MODELS_DIR        = "models"
_PM_SYNC_INTERVAL_SEC = 10 * 3600   # os.fsync at most once every 10 hours


class _PMEpisodeCheckpoint:
    """Per-episode JSONL checkpoint for one PointMaze method evaluation.

    to_metrics() returns (summary, arr, {}) matching _eval_policy_episodic's
    3-tuple so callers need no special-casing.
    """

    def __init__(self, outdir: str, method: str):
        import time as _time
        ep_dir = os.path.join(outdir, _PM_EP_CKPT_DIR)
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
        return len(self.episodes)

    def append(self, ep_data: Dict) -> None:
        import time as _time
        self.episodes.append(ep_data)
        with open(self.path, "a") as f:
            f.write(json.dumps(ep_data) + "\n")
            f.flush()
            now = _time.monotonic()
            if now - self._last_sync >= _PM_SYNC_INTERVAL_SEC:
                os.fsync(f.fileno())
                self._last_sync = now

    def to_metrics(self):
        """Return (summary_dict, arr_dict, empty_traj_payload) from checkpoint data."""
        if not self.episodes:
            return {}, {"R": np.array([]), "RD": np.array([]), "R0": np.array([])}, {}
        R    = np.array([e["return"]        for e in self.episodes], dtype=np.float64)
        RD   = np.array([e["return_disc"]   for e in self.episodes], dtype=np.float64)
        R0   = np.array([e["return_wo_lc"]  for e in self.episodes], dtype=np.float64)
        L    = np.array([e["ep_len"]        for e in self.episodes], dtype=np.float64)
        S    = np.array([e["success"]       for e in self.episodes], dtype=np.float64)
        TTS  = np.array([
            float(e["tts"]) if e["tts"] is not None else np.nan
            for e in self.episodes
        ], dtype=np.float64)
        FB   = np.array([e["fallback_frac"] for e in self.episodes], dtype=np.float64)
        COV  = np.array([e["coverage"]      for e in self.episodes], dtype=np.float64)

        tts_succ = TTS[np.isfinite(TTS)]
        tts_mean = float(np.mean(tts_succ)) if tts_succ.size > 0 else float("nan")
        tts_med  = float(np.median(tts_succ)) if tts_succ.size > 0 else float("nan")

        summary = {
            "return_mean":                float(np.mean(R)),
            "return_std":                 float(np.std(R)),
            "cvar15":                     cvar_lower_tail(R, 0.15),
            "discounted_return_mean":     float(np.mean(RD)),
            "discounted_return_std":      float(np.std(RD)),
            "discounted_cvar15":          cvar_lower_tail(RD, 0.15),
            "success_rate":               float(np.mean(S)),
            "episode_len_mean":           float(np.mean(L)),
            "episode_len_std":            float(np.std(L)),
            "time_to_success_all_mean":   float(np.mean(TTS)),
            "time_to_success_mean":       tts_mean,
            "time_to_success_median":     tts_med,
            "root_fallback_rate":         float(np.mean(FB)),
            "root_coverage_mean":         float(np.mean(COV)),
            "return_wo_lc_mean":          float(np.mean(R0)),
            "return_wo_lc_std":           float(np.std(R0)),
            "return_wo_lc_cvar15":        cvar_lower_tail(R0, 0.15),
            "lc_paid_mean":               float(np.mean(R0 - R)),
            "lc_paid_std":                float(np.std(R0 - R)),
        }
        arr = {"R": R, "RD": RD, "R0": R0}
        return summary, arr, {}  # no traj_payload from checkpoint


def _pm_load_completed_methods(outdir: str) -> Dict[str, Dict]:
    """Load completed method rows from partial checkpoint. Returns {method: row}."""
    path = os.path.join(outdir, _PM_PARTIAL_FILE)
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


def _pm_save_partial_result(outdir: str, row: Dict) -> None:
    """Append a completed method result to the partial checkpoint."""
    path = os.path.join(outdir, _PM_PARTIAL_FILE)
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


def _pm_save_baseline_model(outdir: str, method: str, agent) -> None:
    """Save all nn.Module state_dicts from a baseline agent."""
    import torch.nn as nn
    models_dir = os.path.join(outdir, _PM_MODELS_DIR)
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"{method}.pt")
    state: Dict = {}
    for name, val in vars(agent).items():
        if isinstance(val, nn.Module):
            state[name] = val.state_dict()
    for attr in ("state_mean", "state_std"):
        v = getattr(agent, attr, None)
        if v is not None:
            state[f"_norm_{attr}"] = v
    torch.save(state, path)
    print(f"[CKPT] Saved {method} model → {path}", flush=True)


def _pm_load_baseline_model(outdir: str, method: str, agent) -> bool:
    """Load saved state_dicts into agent. Returns True if checkpoint found."""
    import torch.nn as nn
    path = os.path.join(outdir, _PM_MODELS_DIR, f"{method}.pt")
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


def _eval_policy_episodic(
    env,
    act_fn,
    n_episodes: int,
    seed: int,
    max_steps: int,
    gamma: float,
    capture_trajs: bool = False,
    capture_episode_indices: Optional[Set[int]] = None,
    ckpt: Optional["_PMEpisodeCheckpoint"] = None,
):
    n_already = ckpt.n_done() if ckpt is not None else 0
    n_todo    = max(0, int(n_episodes) - n_already)

    if n_already > 0:
        print(f"[CKPT] Resuming from episode {n_already}/{n_episodes}", flush=True)

    if n_todo == 0 and ckpt is not None:
        return ckpt.to_metrics()

    R, RD, L, S, R0 = [], [], [], [], []
    TTS_ALL, TTS_SUCC = [], []
    FB, COV = [], []

    # episode-index -> trajectory payload
    traj_payload: Dict[int, Dict[str, Any]] = {}

    for ei in range(n_todo):
        actual_ei = n_already + ei
        obs, _ = env.reset(seed=int(seed + actual_ei))
        ep_ret = 0.0
        ep_ret_d = 0.0
        ep_len = 0
        ep_succ = 0.0
        t_first_success = None

        fb_count = 0
        cov_sum = 0.0
        cov_count = 0

        capture_this_ep = bool(capture_trajs) and (
            capture_episode_indices is None or int(actual_ei) in capture_episode_indices
        )

        traj_xy = []
        traj_start = None
        traj_goal = None

        if capture_this_ep:
            traj_start = np.asarray(obs["achieved_goal"], dtype=np.float32).copy()
            traj_goal = np.asarray(obs["desired_goal"], dtype=np.float32).copy()
            traj_xy.append(traj_start.copy())

        for t in range(int(max_steps)):
            a = act_fn(obs)

            # diagnostics hooks
            fb_count += int(bool(getattr(act_fn, "last_root_fallback", False)))
            cov_sum += float(getattr(act_fn, "last_root_coverage", 0.0))
            cov_count += 1

            obs, r, terminated, truncated, info = env.step(a)

            if capture_this_ep:
                traj_xy.append(np.asarray(obs["achieved_goal"], dtype=np.float32).copy())

            ep_ret += float(r)
            ep_ret_d = _discounted_add(ep_ret_d, float(r), float(gamma), t)
            ep_len = t + 1

            if isinstance(info, dict) and float(info.get("success", 0.0)) > 0.0:
                ep_succ = 1.0
                if t_first_success is None:
                    t_first_success = t

            if terminated or truncated:
                break

        lc = float(getattr(env, "living_cost", 0.0))
        ep_ret_wo_lc = float(ep_ret + lc * ep_len)

        R.append(ep_ret)
        R0.append(ep_ret_wo_lc)
        RD.append(ep_ret_d)
        L.append(ep_len)
        S.append(ep_succ)
        TTS_ALL.append(float(t_first_success) if t_first_success is not None else float(max_steps))
        TTS_SUCC.append(float(t_first_success) if t_first_success is not None else np.nan)
        FB.append(fb_count / max(1, cov_count))
        COV.append(cov_sum / max(1, cov_count))

        if ckpt is not None:
            ckpt.append({
                "ep":           actual_ei,
                "return":       ep_ret,
                "return_disc":  ep_ret_d,
                "return_wo_lc": ep_ret_wo_lc,
                "ep_len":       ep_len,
                "success":      ep_succ,
                "tts":          float(t_first_success) if t_first_success is not None else None,
                "fallback_frac": fb_count / max(1, cov_count),
                "coverage":     cov_sum / max(1, cov_count),
            })

        if capture_this_ep:
            traj_payload[int(actual_ei)] = {
                "xy": np.asarray(traj_xy, dtype=np.float32),   # shape [T+1, 2]
                "start": np.asarray(traj_start, dtype=np.float32),
                "goal": np.asarray(traj_goal, dtype=np.float32),
                "success": float(ep_succ),
                "n_steps": int(ep_len),
                "return": float(ep_ret),
                "discounted_return": float(ep_ret_d),
                "return_wo_lc": float(ep_ret_wo_lc),
                "time_to_success": (
                    float(t_first_success) if t_first_success is not None else np.nan
                ),
            }

    R = np.asarray(R, dtype=np.float64)
    R0 = np.asarray(R0, dtype=np.float64)
    RD = np.asarray(RD, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    TTS_ALL = np.asarray(TTS_ALL, dtype=np.float64)
    TTS_SUCC = np.asarray(TTS_SUCC, dtype=np.float64)
    FB = np.asarray(FB, dtype=np.float64)
    COV = np.asarray(COV, dtype=np.float64)

    tts_succ_vals = TTS_SUCC[np.isfinite(TTS_SUCC)]
    tts_mean = float(np.mean(tts_succ_vals)) if tts_succ_vals.size > 0 else float("nan")
    tts_med = float(np.median(tts_succ_vals)) if tts_succ_vals.size > 0 else float("nan")

    summary = {
        "return_mean": float(np.mean(R)),
        "return_std": float(np.std(R)),
        "cvar15": cvar_lower_tail(R, 0.15),

        "discounted_return_mean": float(np.mean(RD)),
        "discounted_return_std": float(np.std(RD)),
        "discounted_cvar15": cvar_lower_tail(RD, 0.15),

        "success_rate": float(np.mean(S)),

        "episode_len_mean": float(np.mean(L)),
        "episode_len_std": float(np.std(L)),

        "time_to_success_all_mean": float(np.mean(TTS_ALL)),
        "time_to_success_mean": float(tts_mean),
        "time_to_success_median": float(tts_med),

        "root_fallback_rate": float(np.mean(FB)),
        "root_coverage_mean": float(np.mean(COV)),

        "return_wo_lc_mean": float(np.mean(R0)),
        "return_wo_lc_std": float(np.std(R0)),
        "return_wo_lc_cvar15": cvar_lower_tail(R0, 0.15),

        # = living_cost * episode_len, per episode, only meaningful when env exposes living_cost
        "lc_paid_mean": float(np.mean(R0 - R)),
        "lc_paid_std": float(np.std(R0 - R)),
    }

    # If checkpoint used, aggregate from all episodes (pre-existing + new)
    if ckpt is not None:
        return ckpt.to_metrics()

    arr = {
        "R": R,
        "RD": RD,
        "R0": R0,
    }

    return summary, arr, traj_payload

def _delta_metrics(delta: np.ndarray, prefix: str) -> dict:
    d = np.asarray(delta, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(d)),
        f"{prefix}_std": float(np.std(d)),
        f"{prefix}_cvar15": cvar_lower_tail(d, 0.15),
        f"{prefix}_p_neg": float(np.mean(d < 0.0)),
    }

def eval_ppo_only(
    env_real,
    ppo,
    n_episodes: int,
    seed: int,
    max_steps: int,
    gamma: float,
    capture_trajs: bool = False,
    capture_episode_indices: Optional[Set[int]] = None,
):
    def act_fn(obs):
        a, _ = ppo.predict(obs, deterministic=True)
        return a

    epi_sum, epi_arr, traj_payload = _eval_policy_episodic(
        env_real,
        act_fn,
        n_episodes,
        seed,
        max_steps,
        gamma,
        capture_trajs=capture_trajs,
        capture_episode_indices=capture_episode_indices,
    )
    return epi_sum, epi_arr, traj_payload

def eval_planner(
    env_real,
    planner,
    n_episodes: int,
    seed: int,
    max_steps: int,
    gamma: float,
    capture_trajs: bool = False,
    capture_episode_indices: Optional[Set[int]] = None,
    ckpt: Optional["_PMEpisodeCheckpoint"] = None,
):
    def act_fn(obs):
        a = planner.act(obs)
        act_fn.last_root_fallback = getattr(planner, "last_root_fallback", False)
        act_fn.last_root_coverage = getattr(planner, "last_root_coverage", 0.0)
        return a

    act_fn.last_root_fallback = getattr(planner, "last_root_fallback", False)
    act_fn.last_root_coverage = getattr(planner, "last_root_coverage", 0.0)

    epi_sum, epi_arr, traj_payload = _eval_policy_episodic(
        env_real,
        act_fn,
        n_episodes,
        seed,
        max_steps,
        gamma,
        capture_trajs=capture_trajs,
        capture_episode_indices=capture_episode_indices,
        ckpt=ckpt,
    )
    return epi_sum, epi_arr, traj_payload

# -------------------------
# Dataset seeds + collection (REGENERATE per ds, nested via prefix)
# -------------------------

def trajectory_seed_list(exp_seed: int, max_ds: int) -> List[int]:
    base = int(exp_seed) * 1_000_000 + 77_777
    return [base + i for i in range(int(max_ds))]

def shuffle_traj_seeds_once(traj_seeds: List[int], exp_seed: int) -> List[int]:
    """
    Deterministic shuffle to remove 'prefix bias' while preserving prefix nesting:
    ds always uses traj_seeds[:ds], but traj_seeds is a fixed random permutation per exp_seed.
    """
    rng = np.random.default_rng(int(exp_seed) + 123_456)  # fixed per exp seed
    ts = list(traj_seeds)
    rng.shuffle(ts)
    return ts

def collect_dataset_from_traj_seeds(
    *,
    env,
    ppo: PPO,
    traj_seeds: List[int],
    max_steps: int,
) -> Dict[str, np.ndarray]:
    obs_obs, obs_ag, obs_dg = [], [], []
    next_obs_obs, next_obs_ag, next_obs_dg = [], [], []
    acts, rews, dones, traj_ids = [], [], [], []

    for ti, s0 in enumerate(traj_seeds):
        obs, _ = env.reset(seed=int(s0))
        for _t in range(int(max_steps)):
            a, _ = ppo.predict(obs, deterministic=False)
            a = _to_f32(a).reshape(-1)
            a = np.clip(a, -1.0, 1.0).astype(np.float32) 

            obs2, r, terminated, truncated, _info = env.step(a)
            done = bool(terminated or truncated)

            obs_obs.append(_to_f32(obs["observation"]).reshape(-1))
            obs_ag.append(_to_f32(obs["achieved_goal"]).reshape(-1))
            obs_dg.append(_to_f32(obs["desired_goal"]).reshape(-1))

            next_obs_obs.append(_to_f32(obs2["observation"]).reshape(-1))
            next_obs_ag.append(_to_f32(obs2["achieved_goal"]).reshape(-1))
            next_obs_dg.append(_to_f32(obs2["desired_goal"]).reshape(-1))

            acts.append(a)
            rews.append(float(r))
            dones.append(1.0 if done else 0.0)
            traj_ids.append(int(ti))

            obs = obs2
            if done:
                break

    return {
        "obs_observation": np.asarray(obs_obs, dtype=np.float32),
        "obs_achieved_goal": np.asarray(obs_ag, dtype=np.float32),
        "obs_desired_goal": np.asarray(obs_dg, dtype=np.float32),
        "next_observation": np.asarray(next_obs_obs, dtype=np.float32),
        "next_achieved_goal": np.asarray(next_obs_ag, dtype=np.float32),
        "next_desired_goal": np.asarray(next_obs_dg, dtype=np.float32),
        "act": np.asarray(acts, dtype=np.float32),
        "rew": np.asarray(rews, dtype=np.float32),
        "done": np.asarray(dones, dtype=np.float32),
        "traj_id": np.asarray(traj_ids, dtype=np.int64),
    }

def choose_calib_traj_ids(seed: int, max_ds: int, calib_frac: float) -> Set[int]:
    rng = np.random.default_rng(int(seed + 88_888))
    ids = np.arange(int(max_ds), dtype=np.int64)
    rng.shuffle(ids)
    n_cal = max(1, int(np.ceil(float(calib_frac) * float(max_ds))))
    return set(int(x) for x in ids[:n_cal])

def split_train_calib_by_traj_ids(
    *,
    data: Dict[str, np.ndarray],
    calib_traj_ids_global: Set[int],
    ds: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    traj = np.asarray(data["traj_id"], dtype=np.int64)
    calib_ids = set(i for i in calib_traj_ids_global if int(i) < int(ds))

    mask_cal = np.array([int(t) in calib_ids for t in traj], dtype=bool)
    mask_tr = ~mask_cal


    def sub(mask):
        # Keep traj_id so calibration can compute n_calib_traj robustly.
        # It won't affect model training/calibration because extra keys are ignored.
        out = {k: v[mask] for k, v in data.items()}
        return out

    return sub(mask_tr), sub(mask_cal)

def _obs_i(data: Dict[str, np.ndarray], i: int) -> Dict[str, np.ndarray]:
    return {
        "observation": data["obs_observation"][i],
        "achieved_goal": data["obs_achieved_goal"][i],
        "desired_goal": data["obs_desired_goal"][i],
    }

# -------------------------
# Ensemble / model env
# -------------------------

def build_ensemble(obs_dim: int, act_dim: int) -> Any:
    # Paper Cal-SPI Point Maze ensemble (Tab. calspi-config in the appendix):
    # K=4, MLP[512,512], 50 training epochs, no bootstrap, no input
    # normalization. The underlying EnsembleTrainConfig ships with bootstrap
    # and normalization enabled by default for legacy reasons, so we override
    # here to match the paper.
    cfg = EnsembleTrainConfig(
        ensemble_size=4,
        hidden_dims=(512, 512),
        epochs=50,
        bootstrap=False,
        normalize_inputs=False,
    )
    try:
        return EnsembleDynamicsSB3(obs_dim=obs_dim, act_dim=act_dim, cfg=cfg)
    except TypeError:
        return EnsembleDynamicsSB3(obs_dim, act_dim, cfg)

def fit_ensemble(ens: Any, train: Dict[str, np.ndarray]) -> None:
    obs_dicts = {
        "observation": train["obs_observation"],
        "achieved_goal": train["obs_achieved_goal"],
        "desired_goal": train["obs_desired_goal"],
    }
    next_obs_dicts = {
        "observation": train["next_observation"],
        "achieved_goal": train["next_achieved_goal"],
        "desired_goal": train["next_desired_goal"],
    }
    ens.fit_from_transitions(obs_dicts, train["act"], next_obs_dicts, verbose=True)

def make_model_env(
    ens: Any,
    continuing_task: int,
    reward_type: str,
    max_steps: int,
    goal_thresh: float = 0.45,
    living_cost: float = 0.0,
    success_bonus: float = 0.0,
    reward_mode: str = "current",
    step_penalty: float = 1.0,
    dist_penalty: float = 0.2,
) -> Any:
    cfg = ModelEnvConfig(
        continuing_task=bool(int(continuing_task)),
        reward_type=str(reward_type),
        goal_thresh=float(goal_thresh),
        max_steps=int(max_steps),
        truncate_on_time_limit=False,  # CRITICAL for MCTS
        living_cost=float(living_cost),
        success_bonus=float(success_bonus),
        reward_mode=str(reward_mode),
        step_penalty=float(step_penalty),
        dist_penalty=float(dist_penalty),
    )
    return ModelPointMazeEnv(ensemble=ens, cfg=cfg)

# -------------------------
# Calibration
# -------------------------

def _wilks_optimal_bins(N_cal: int, delta: float = 0.05, alpha: float = 0.05) -> int:
    """
    Wilks-optimal B*: largest B s.t. N_cal >= (B/delta)*ln(B/alpha).
    Eq. ncal-wilks in pac_guarantees.tex.  Mirrors benchmark_continuous.py.
    """
    if N_cal <= 0:
        return 1
    lo, hi = 1, N_cal
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if (mid / delta) * math.log(mid / alpha) <= N_cal:
            lo = mid
        else:
            hi = mid - 1
    return max(1, lo)


def _wilks_min_n_per_bin(n_bins: int, delta: float = 0.05, alpha: float = 0.05) -> int:
    """Per-bin minimum: ceil(ln(alpha/n_bins) / ln(1-delta)).  Bonferroni-corrected."""
    if n_bins <= 0 or delta <= 0 or delta >= 1:
        return 0
    alpha_b = alpha / max(1, n_bins)
    return int(math.ceil(math.log(alpha_b) / math.log(1.0 - delta)))


def fit_calibrator_and_eps(
    *,
    ens: Any,
    calib: Dict[str, np.ndarray],
    kind: str,  # "usad" | "epistemic"
    seed: int,
    eps_quantile: float,
    n_bins: int,
    delta: float,
    min_samples: int,
    max_steps: int,
    stochastic: bool = False,   # if True: e_i = E_{k,eps}[||s'_stoch - s'_true||]
    n_eps: int = 10,            # noise samples per member (only used when stochastic=True)
) -> Tuple[Any, float]:
    N = int(calib["act"].shape[0])

    # ── Wilks-optimal B* (eq. ncal-wilks in pac_guarantees.tex) ──────────────
    # If n_bins=0 (auto mode): compute B* = max B s.t. N >= (B/delta)*ln(B/alpha).
    # Also derives the Wilks per-bin minimum for min_bin_size.
    if int(n_bins) == 0:
        n_bins = _wilks_optimal_bins(N, float(delta))
        _min_bin_size = _wilks_min_n_per_bin(n_bins, float(delta))
    else:
        n_bins = int(n_bins)
        _min_bin_size = int(min_samples)  # legacy: min_samples used as min_bin_size

    # calib_min_bins derived from eps_quantile: ceil(1/(1-eps_quantile)).
    # With B < ceil(1/(1-q)), the q-quantile of B(u) equals B_last, making
    # the gate always pass for OOD actions.  (See calib_min_bins_check docstring.)
    _calib_min_bins = math.ceil(1.0 / (1.0 - float(eps_quantile)))

    if N <= 0:
        cal = UncertaintyCalibrator.fit(
            np.array([0.0], dtype=np.float64),
            np.array([1e9], dtype=np.float64),
            cfg=UncertaintyCalibratorConfig(n_bins=2, delta=float(delta), min_samples=int(min_samples)),
        )
        # IMPORTANT: planner checks cal.meta["low_data_regime"]
        cal.meta = dict(getattr(cal, "meta", {}) or {})
        cal.meta["kind"] = str(kind)
        cal.meta["low_data_regime"] = True
        setattr(cal, "low_data_regime", True)
        return cal, float(1e9)

    u = np.zeros((N,), dtype=np.float64)
    e = np.zeros((N,), dtype=np.float64)

    calib_mode_str = f"stochastic (K×n_eps={len(ens.ensemble)}×{n_eps})" if stochastic else "deterministic (ensemble_mean)"
    print(f"[CALIB] kind={kind} mode={calib_mode_str} N={N}", flush=True)

    for i in range(N):
        obs = _obs_i(calib, i)
        act = calib["act"][i]
        if kind == "usad":
            ui = float(ens.uncertainty_usad(obs, act))
        else:
            ui = float(ens.uncertainty_epistemic(obs, act))

        true_next = np.asarray(calib["next_observation"][i], dtype=np.float32).ravel()

        if stochastic:
            # Stochastic calibration: e_i = E_{k, eps}[ ||s'_stoch - s'_true|| ]
            # Samples K members × n_eps noise draws → mean error over the full
            # transition distribution, not just the ensemble mean.
            # This gives a tighter, more rigorous bound on the error of the
            # actual stochastic transitions used during MCTS planning.
            K = len(ens.ensemble)
            calib_rng = np.random.default_rng(int(seed) + i)
            sample_errors = []
            for k in range(K):
                for _ in range(int(n_eps)):
                    s_hat = ens.sample_next_obs_vec(obs, act, model_index=k, rng=calib_rng)
                    sample_errors.append(
                        float(np.linalg.norm(
                            np.asarray(s_hat, np.float32).ravel() - true_next, ord=2
                        ))
                    )
            ei = float(np.mean(sample_errors))

            # DEBUG (sparse)
            if i in (0, 1, 2, N - 1):
                print(
                    f"[CALIB DBG STOCH] kind={kind} i={i} K={K} n_eps={n_eps} "
                    f"mean_err={ei:.6g} min={min(sample_errors):.6g} max={max(sample_errors):.6g}",
                    flush=True,
                )
        else:
            # Deterministic calibration: e_i = ||mean_k[mu_k] - s'_true||
            # Measures systematic (bias) error of the ensemble mean estimator.
            pred_next = ens.predict_next_obs_vec_mean(obs, act)
            pred_next_arr = np.asarray(pred_next, np.float32).ravel()
            ei = float(np.linalg.norm(pred_next_arr - true_next, ord=2))

            # DEBUG
            if i in (0, 1, 2, N - 1):
                _dbg_shape("pred_next", pred_next_arr, "true_next", true_next)
                print(f"[CALIB DBG] kind={kind} i={i}", flush=True)
                _dbg_stats("pred_next (single)", pred_next_arr)
                _dbg_stats("true_next (single)", true_next)
                _dbg_stats("diff (single)", pred_next_arr - true_next)

        u[i] = ui
        e[i] = ei

    # DEBUG
    _dbg_stats(f"u ({kind})", u)
    _dbg_stats(f"e L2(pred_next - next_observation) ({kind})", e)

    # -------------------------
    # Robust low-data detection for QUANTILE BINNING (paper-friendly)
    # -------------------------
    B = int(n_bins)
    m = int(min_samples)
    N_eff = int(u.shape[0])

    # (A) Hard feasibility: if total points < bins*min_samples, quantile binning cannot meet per-bin minimum.
    low_data_by_counts = (N_eff < B * m)

    # (B) Need at least a couple trajectories to avoid degenerate calibration.
    T_MIN = 2
    if "traj_id" in calib:
        traj_ids_raw = np.asarray(calib["traj_id"]).reshape(-1)
        # Robust mapping even if traj_id are arbitrary (not contiguous / not starting at 0)
        _, inv = np.unique(traj_ids_raw, return_inverse=True)
        n_calib_traj = int(inv.max() + 1) if inv.size > 0 else 0
        counts = np.bincount(inv) if inv.size > 0 else np.asarray([], dtype=np.int64)
        L_med = int(np.median(counts)) if counts.size > 0 else 0
    else:
        # Conservative fallback if traj_id is missing: treat as 1 trajectory of length N_eff
        n_calib_traj = 1
        L_med = int(N_eff)

    # (C) Data-driven effective horizon per trajectory: median traj length, capped by episode horizon.
    # This avoids an arbitrary fixed L_EFF (e.g., 50) while still accounting for correlation.
    L_eff = int(min(int(max_steps), max(1, L_med)))

    low_data_by_traj = (n_calib_traj < T_MIN)
    low_data_by_eff = (n_calib_traj * L_eff < B * m)

    # Optional degeneracy check on u: if too few unique values, quantile binning collapses.
    avg_per_bin = float(N_eff) / float(max(1, B))
    n_unique_u = int(np.unique(u).shape[0])
    low_data_by_unique = (n_unique_u < max(5, B // 2))

    force_low_data = bool(low_data_by_traj or low_data_by_counts or low_data_by_eff or low_data_by_unique)

    _n_cal_for_b = int(math.ceil((n_bins / float(delta)) * math.log(n_bins / 0.05))) if n_bins > 0 else 0
    print(
        f"[CALIB DBG] low-data check kind={kind} | "
        f"N={N_eff} B*={n_bins} (need N>={_n_cal_for_b}) min_bin={_min_bin_size} "
        f"calib_min_bins={_calib_min_bins} (ceil(1/(1-{eps_quantile}))) | "
        f"n_calib_traj={n_calib_traj} (T_MIN={T_MIN}) L_med={L_med} L_eff={L_eff} "
        f"unique_u={n_unique_u} -> low_data={int(force_low_data)}",
        flush=True,
    )

    # If low-data, do NOT fit a real calibrator (claim: B(u) not reliable).
    # Return a dummy calibrator that forces PPO-only via meta["low_data_regime"] and eps=1e9.
    if force_low_data:
        cal = UncertaintyCalibrator.fit(
            np.array([0.0], dtype=np.float64),
            np.array([1e9], dtype=np.float64),
            cfg=UncertaintyCalibratorConfig(n_bins=2, delta=float(delta), min_samples=int(min_samples)),
            meta={"kind": str(kind), "low_data_regime": True},
        )
        cal.meta = dict(getattr(cal, "meta", {}) or {})
        cal.meta["kind"] = str(kind)
        cal.meta["low_data_regime"] = True
        setattr(cal, "low_data_regime", True)
        return cal, float(1e9)

    # ── NEW: Wilks-strict + triple-split + Bonferroni candidate sweep ────────
    # Aligned with continuous benchmark calibration pipeline. See
    # safe_rl/offline/uncertainty_calibration.py: fit_with_selection.
    rng_split = np.random.default_rng(0)
    perm = rng_split.permutation(int(N_eff))
    n1 = int(N_eff) // 3
    idx_bin   = perm[:n1]
    idx_wilks = perm[n1:2 * n1]
    idx_thr   = perm[2 * n1:]

    # Paper-canonical candidate set: B_min = ceil(1/(1-q_eps)) and linear
    # multiples {Bm, 2 Bm, 3 Bm, 4 Bm}. For the paper q_eps = 0.95 this yields
    # the canonical {20, 40, 60, 80} reported in Tab. calspi-config.
    Bm = int(math.ceil(1.0 / max(1e-12, 1.0 - float(eps_quantile))))
    cand_list = [int(Bm * (k + 1)) for k in range(4)]

    # Pass 1: bin edges on D_bin, one set per B
    u_bin = u[idx_bin]
    bin_edges_per_B: dict = {}
    for B in cand_list:
        qs = np.linspace(0.0, 1.0, B + 1)
        edges_B = np.quantile(u_bin, qs).astype(np.float64)
        for i in range(1, len(edges_B)):
            if edges_B[i] <= edges_B[i - 1]:
                edges_B[i] = edges_B[i - 1] + 1e-12
        bin_edges_per_B[int(B)] = edges_B

    cfg_template = UncertaintyCalibratorConfig(
        delta=float(delta),
        binning="quantile",
        mode="wilks_strict",
        alpha=0.05,
    )

    try:
        cal, eps_raw, sel_diag = UncertaintyCalibrator.fit_with_selection(
            u_wilks=u[idx_wilks], e_wilks=e[idx_wilks], u_thr=u[idx_thr],
            candidate_bins=cand_list, bin_edges_per_B=bin_edges_per_B,
            cfg_template=cfg_template,
            q_eps=float(eps_quantile),
            n_bootstrap=200,
            random_seed=int(seed),
        )
    except Exception as _exc:
        print(f"[CALIB DBG] fit_with_selection error: {_exc}; falling back to legacy quantile fit", flush=True)
        cal = UncertaintyCalibrator.fit(
            u, e,
            cfg=UncertaintyCalibratorConfig(n_bins=int(n_bins), delta=float(delta),
                                             binning="quantile",
                                             min_bin_size=int(_min_bin_size),
                                             min_samples=int(min_samples),
                                             monotone=True),
            meta={"kind": str(kind)},
        )
        sel_diag = {"collapsed": False, "B_dagger": int(n_bins),
                    "eps_selected": float(np.quantile(cal.bound_batch(u), float(eps_quantile)))}
        eps_raw = sel_diag["eps_selected"]

    # Console diagnostics (paper-grade)
    print(f"[CALIB] Triple-split + candidate sweep: "
          f"N_cal={int(N_eff)} (bin={idx_bin.size}, wilks={idx_wilks.size}, thr={idx_thr.size}); "
          f"candidates={cand_list}", flush=True)
    print(f"[CALIB] {'B':>4} {'min_n_b':>8} {'req_n_b':>8} {'alpha_eff':>10} "
          f"{'eps_B':>9} {'admissible':>11} {'selected':>9}", flush=True)
    sel_B = sel_diag.get("B_dagger", None)
    for rec in sel_diag.get("per_candidate", []):
        marker = "*" if (sel_B is not None and rec.get("B") == sel_B) else " "
        eB = rec.get("eps_B")
        eB_str = f"{float(eB):9.4f}" if (eB is not None and np.isfinite(float(eB))) else f"{'—':>9}"
        ae = rec.get("alpha_eff", float("nan"))
        print(f"[CALIB] {rec.get('B', 0):>4} {rec.get('min_bin_count', 0):>8} "
              f"{rec.get('required_min_bin_count', 0):>8} {ae:>10.4g} {eB_str} "
              f"{('YES' if rec.get('admissible', False) else '---'):>11} {marker:>9}", flush=True)

    cal.meta = dict(getattr(cal, "meta", {}) or {})
    cal.meta["kind"] = str(kind)
    collapsed = bool(sel_diag.get("collapsed", False))
    cal.meta["low_data_regime"] = collapsed
    setattr(cal, "low_data_regime", collapsed)

    eps_emp = float(eps_raw) if (eps_raw is not None and np.isfinite(float(eps_raw))) else float("inf")
    if not collapsed:
        b = cal.bound_batch(u)
        _dbg_stats(f"b = cal.bound_batch(u) ({kind})", b)
    print(f"[CALIB DBG] kind={kind} q_eps={float(eps_quantile):.4f} -> "
          f"eps={eps_emp:.6g} B_dagger={sel_B} collapsed={int(collapsed)}", flush=True)

    # ── min_bins check (Wilks discriminability, eq. calib_min_bins in paper) ──
    # With B_eff < ceil(1/(1-eps_quantile)), the eps_quantile-quantile of B(u)
    # equals B_last, so OOD actions always pass the gate.  Force low_data_regime.
    # NEW Wilks-strict + triple-split summary (replaces legacy min_bins block)
    _sep = "\u2500" * 54
    print(f"\n[CALIB/{kind.upper()}] {_sep}", flush=True)
    print(f"  N_cal={int(N_eff)} | delta={delta} | q_eps={eps_quantile} | alpha=0.05", flush=True)
    print(f"  candidates B       : {cand_list}", flush=True)
    print(f"  triple split       : D_bin={idx_bin.size}, D_wilks={idx_wilks.size}, D_thr={idx_thr.size}", flush=True)
    if collapsed:
        print(f"  -> COLLAPSE: admissible set empty -> eps=-inf, fallback PPO (Prop. 2)", flush=True)
    else:
        print(f"  B_dagger           : {sel_B}", flush=True)
        print(f"  eps                : {eps_emp:.4f}", flush=True)
        eps_se_val = sel_diag.get('eps_se', float('nan'))
        if eps_se_val is not None and np.isfinite(float(eps_se_val)):
            print(f"  eps_se (bootstrap) : {float(eps_se_val):.4f}", flush=True)
    print(f"[CALIB/{kind.upper()}] {_sep}\n", flush=True)

    if collapsed:
        cal.meta["low_data_regime"] = True
        setattr(cal, "low_data_regime", True)
        return cal, float(1e9)

    return cal, eps_emp

# def fit_calibrator_and_eps(
#     *,
#     ens: Any,
#     calib: Dict[str, np.ndarray],
#     kind: str,  # "usad" | "epistemic"
#     seed: int,
#     eps_quantile: float,
#     n_bins: int,
#     delta: float,
#     min_samples: int,
# ) -> Tuple[Any, float]:
#     # N = int(calib["act"].shape[0])
#     # if N <= 0:
#     #     cal = UncertaintyCalibrator.fit(
#     #         np.array([0.0], dtype=np.float64),
#     #         np.array([1e9], dtype=np.float64),
#     #         cfg=UncertaintyCalibratorConfig(n_bins=2, delta=float(delta), min_samples=int(min_samples)),
#     #     )
#     #     setattr(cal, "low_data_regime", True)
#     #     return cal, float(1e9)
#     N = int(calib["act"].shape[0])
#     if N <= 0:
#         cal = UncertaintyCalibrator.fit(
#             np.array([0.0], dtype=np.float64),
#             np.array([1e9], dtype=np.float64),
#             cfg=UncertaintyCalibratorConfig(n_bins=2, delta=float(delta), min_samples=int(min_samples)),
#         )
#         # IMPORTANT: planner checks cal.meta["low_data_regime"]
#         cal.meta = dict(getattr(cal, "meta", {}) or {})
#         cal.meta["kind"] = str(kind)
#         cal.meta["low_data_regime"] = True
#         return cal, float(1e9)

#     u = np.zeros((N,), dtype=np.float64)
#     e = np.zeros((N,), dtype=np.float64)

#     for i in range(N):
#         obs = _obs_i(calib, i)
#         act = calib["act"][i]
#         if kind == "usad":
#             ui = float(ens.uncertainty_usad(obs, act))
#         else:
#             ui = float(ens.uncertainty_epistemic(obs, act))

#         pred_next = ens.predict_next_obs_vec_mean(obs, act)
#         true_next = np.asarray(calib["next_observation"][i], dtype=np.float32).ravel()
#         # DEBUG
#         if i in (0, 1, 2, N - 1):
#             pred_next_arr = np.asarray(pred_next, np.float32).ravel()
#             _dbg_shape("pred_next", pred_next_arr, "true_next", true_next)
#             print(f"[CALIB DBG] kind={kind} i={i}", flush=True)
#             _dbg_stats("pred_next (single)", pred_next_arr)
#             _dbg_stats("true_next (single)", true_next)
#             _dbg_stats("diff (single)", pred_next_arr - true_next)

#         ei = float(np.linalg.norm(np.asarray(pred_next, np.float32).ravel() - true_next, ord=2))

#         u[i] = ui
#         e[i] = ei
        
#     # DEBUG
#     _dbg_stats(f"u ({kind})", u)
#     _dbg_stats(f"e L2(pred_next - next_observation) ({kind})", e)

#     # -------------------------
#     # Robust low-data detection for QUANTILE BINNING
#     # -------------------------
#     N = int(u.shape[0])
#     B = int(n_bins)
#     m = int(min_samples)

#     # In quantile binning, each bin should have ~N/B points.
#     # If N < B*m, it is impossible (in expectation) to have >= m samples per bin.
#     # This is a robust, statistic-based trigger.
#     avg_per_bin = float(N) / float(max(1, B))
#     low_data_by_counts = (N < B * m)

#     # Additional robust checks (avoid degenerate u distributions):
#     # - If u has too few unique values, quantile binning collapses.
#     # - Use a conservative threshold tied to B.
#     n_unique_u = int(np.unique(u).shape[0])
#     low_data_by_unique = (n_unique_u < max(5, B // 2))

#     force_low_data = bool(low_data_by_counts or low_data_by_unique)

#     print(
#         f"[CALIB DBG] low-data check kind={kind} | "
#         f"N={N} bins={B} min_samples={m} avg_per_bin={avg_per_bin:.2f} "
#         f"unique_u={n_unique_u} -> low_data={int(force_low_data)}",
#         flush=True,
#     )

#     cal = UncertaintyCalibrator.fit(
#         u,
#         e,
#         cfg=UncertaintyCalibratorConfig(
#             n_bins=int(n_bins),
#             delta=float(delta),
#             binning="quantile",
#             min_samples=int(min_samples),
#             monotone=True,
#         ),
#         meta={"kind": str(kind)},
#     )
#     setattr(cal, "low_data_regime", bool(cal.meta.get("low_data_regime", False)))

#     b = cal.bound_batch(u)
#     # DEBUG
#     _dbg_stats(f"b = cal.bound_batch(u) ({kind})", b)
#     eps_emp = float(np.quantile(b, float(eps_quantile)))
#     print(f"[CALIB DBG] kind={kind} eps_quantile={float(eps_quantile):.4f} -> eps_emp={eps_emp:.6g}", flush=True)
#     pass_frac = float(np.mean(np.asarray(b) <= eps_emp))
#     print(f"[CALIB DBG] kind={kind} calib pass fraction ~ {pass_frac:.3f}", flush=True)
#     return cal, eps_emp

# -------------------------
# Plotting
# -------------------------

def plot_metric(df, xcol: str, ycol: str, out_path: str, title: str) -> None:
    plt.figure()
    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method]
        xs = sorted(sub[xcol].unique())
        ys = [sub[sub[xcol] == x][ycol].mean() for x in xs]
        plt.plot(xs, ys, marker="o", label=method)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

PAPER_STYLE = {
    # method_key_in_df : dict(label=..., color=..., marker=..., linestyle=..., markersize=..., linewidth=...)
    "ppo_only": {
        "label": "Baseline",
        "color": "yellow",
        "marker": None,          # no points
        "linestyle": "-",
        "markersize": 0,
        "linewidth": 2.5,
    },
    "mcts_only_no_gate": {
        "label": "MCTS-DPW",
        "color": "blue",
        "marker": "v",           # triangle down
        "linestyle": "-",
        "markersize": 7,
        "linewidth": 2.0,
    },
    "morel_usad": {
        "label": "SPI-U",
        "color": "purple",
        "marker": "s",           # square
        "linestyle": "-",
        "markersize": 6,
        "linewidth": 2.0,
    },
    "calibrated_variance": {
        "label": "CSPI-V",
        "color": "red",
        "marker": "x",
        "linestyle": "-",
        "markersize": 7,
        "linewidth": 2.0,
    },
    "calibrated_usad": {
        "label": "CSPI-U",
        "color": "green",
        "marker": "o",
        "linestyle": "-",
        "markersize": 5,
        "linewidth": 2.0,
    },
}

PAPER_METHOD_ORDER = ["ppo_only", "mcts_only_no_gate", "morel_usad", "calibrated_usad", "calibrated_variance"]


def _paper_aggregate(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Aggregate metric over seeds for each (dataset_trajs, method).
    Expects columns: dataset_trajs, method, seed, metric.
    Returns columns: dataset_trajs, method, mean, std.
    """
    req = {"dataset_trajs", "method", metric}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for plotting: {missing}")

    g = df.groupby(["dataset_trajs", "method"], as_index=False)[metric]
    out = g.agg(mean="mean", std="std")
    # Replace NaN std (single seed) with 0.0 to avoid issues
    out["std"] = out["std"].fillna(0.0)
    return out


def _paper_plot_one(df: pd.DataFrame, metric: str, outpath_svg: str, ylabel: str):
    """
    Paper-ready plot: no title, fixed x label (latex), legend with requested labels/styles.
    Saves SVG (overwrites).
    """
    agg = _paper_aggregate(df, metric)

    # Keep only known methods, enforce order
    methods_present = [m for m in PAPER_METHOD_ORDER if m in set(agg["method"].unique())]

    fig = plt.figure()
    ax = fig.gca()

    for m in methods_present:
        st = PAPER_STYLE[m]
        sub = agg[agg["method"] == m].sort_values("dataset_trajs")
        x = sub["dataset_trajs"].to_numpy()
        y = sub["mean"].to_numpy()

        ax.plot(
            x, y,
            label=st["label"],
            color=st["color"],
            linestyle=st["linestyle"],
            marker=st["marker"],
            markersize=st["markersize"],
            linewidth=st["linewidth"],
        )

    ax.set_xlabel(r"Number of trajectories in dataset $\mathcal{D}$")
    ax.set_ylabel(ylabel)

    ax.legend(frameon=False)
    fig.tight_layout()

    # SVG ignores DPI in raster sense, but matplotlib accepts dpi; keep as requested
    fig.savefig(outpath_svg, format="svg", dpi=300)
    plt.close(fig)


def plot_three_paper_ready(df: pd.DataFrame, outdir: str):
    """
    Writes 3 SVG plots (overwrites each time).
    """
    os.makedirs(outdir, exist_ok=True)

    _paper_plot_one(
        df,
        metric="return_mean",
        outpath_svg=os.path.join(outdir, "plot_mean.svg"),
        ylabel=r"$\bar{\rho}(\pi, M^*)$",
    )
    _paper_plot_one(
        df,
        metric="cvar15",
        outpath_svg=os.path.join(outdir, "plot_cvar.svg"),
        ylabel=r"CVaR $\bar{\rho}(\pi, M^*)$",
    )
    _paper_plot_one(
        df,
        metric="success_rate",
        outpath_svg=os.path.join(outdir, "plot_success.svg"),
        ylabel=r"Success rate",
    )


def save_checkpoint(df: pd.DataFrame, outdir: str):
    """
    Overwrites:
      - results_checkpoint.csv
      - results_checkpoint.json
      - plot_mean.svg, plot_cvar.svg, plot_success.svg
    """
    os.makedirs(outdir, exist_ok=True)

    ckpt_csv = os.path.join(outdir, "results_checkpoint.csv")
    ckpt_json = os.path.join(outdir, "results_checkpoint.json")

    # CSV
    df.to_csv(ckpt_csv, index=False)

    # JSON (nested)
    nested = df_to_nested_json(df)
    with open(ckpt_json, "w") as f:
        json.dump(nested, f, indent=2)

    # Plots
    plot_three_paper_ready(df, outdir)

    print(f"[CHECKPOINT] wrote {ckpt_csv} + {ckpt_json} + updated SVG plots", flush=True)


def df_to_nested_json(df: pd.DataFrame):
    out = {}
    for _, row in df.iterrows():
        s = int(row["seed"])
        d = int(row["dataset_trajs"])
        m = row["method"]

        out.setdefault(s, {})
        out[s].setdefault(d, {})
        out[s][d][m] = {
            k: row[k]
            for k in [
                "return_mean",
                "return_std",
                "cvar15",
                "success_rate",
                "root_fallback_rate",
                "root_coverage_mean",
            ]
        }
    return out

def save_dataset_cache(outdir, seed, ds, data, train, calib, *, traj_seeds_used, calib_traj_ids_global):
    """
    Save full transitions + train/calib split + metadata needed for exact reuse.
    """
    cache_dir = os.path.join(outdir, "cache", f"seed_{seed}", f"ds_{ds}")
    os.makedirs(cache_dir, exist_ok=True)

    full_path = os.path.join(cache_dir, "transitions_full.npz")
    tr_path = os.path.join(cache_dir, "transitions_train.npz")
    cal_path = os.path.join(cache_dir, "transitions_calib.npz")

    # Full dataset (includes traj_id)
    np.savez_compressed(full_path, **data)

    # Train split
    np.savez_compressed(tr_path, **train)

    # Calibration split
    np.savez_compressed(cal_path, **calib)

    meta = {
        "seed": int(seed),
        "dataset_trajs": int(ds),
        "traj_seeds_used": [int(x) for x in traj_seeds_used],  # length = ds
        "calib_traj_ids_global": sorted([int(x) for x in calib_traj_ids_global]),
        # NOTE: effective calib ids are global filtered by < ds; save both so you can reconstruct
        "calib_traj_ids_effective": sorted([int(x) for x in calib_traj_ids_global if int(x) < int(ds)]),
        "counts": {
            "full_transitions": int(data["act"].shape[0]),
            "train_transitions": int(train["act"].shape[0]),
            "calib_transitions": int(calib["act"].shape[0]),
        },
        "files": {
            "transitions_full": os.path.basename(full_path),
            "transitions_train": os.path.basename(tr_path),
            "transitions_calib": os.path.basename(cal_path),
        },
    }

    # Hash files to ensure exact identity across future runs
    meta["sha256"] = {
        "transitions_full": sha256_file(full_path),
        "transitions_train": sha256_file(tr_path),
        "transitions_calib": sha256_file(cal_path),
    }

    with open(os.path.join(cache_dir, "dataset_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=_jsonable)

    print(f"[CACHE] dataset saved at {cache_dir}", flush=True)

def save_calibrator_cache(outdir, seed, ds, *, name: str, calibrator, eps: float):
    cache_dir = os.path.join(outdir, "cache", f"seed_{seed}", f"ds_{ds}")
    os.makedirs(cache_dir, exist_ok=True)

    cal_path = os.path.join(cache_dir, f"calibrator_{name}.pkl")
    eps_path = os.path.join(cache_dir, f"eps_{name}.json")

    with open(cal_path, "wb") as f:
        pickle.dump(calibrator, f)

    with open(eps_path, "w") as f:
        json.dump({"name": str(name), "eps": float(eps)}, f, indent=2)

    meta = {
        "seed": int(seed),
        "dataset_trajs": int(ds),
        "name": str(name),
        "eps": float(eps),
        "sha256": {
            "calibrator_pkl": sha256_file(cal_path),
            "eps_json": sha256_file(eps_path),
        },
        "files": {
            "calibrator_pkl": os.path.basename(cal_path),
            "eps_json": os.path.basename(eps_path),
        },
    }
    with open(os.path.join(cache_dir, f"calibrator_{name}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=_jsonable)

    print(f"[CACHE] calibrator '{name}' saved at {cal_path}", flush=True)

def save_ensemble_cache(outdir, seed, ds, ensemble, *, obs_dim: int, act_dim: int, ensemble_cfg=None):
    cache_dir = os.path.join(outdir, "cache", f"seed_{seed}", f"ds_{ds}")
    os.makedirs(cache_dir, exist_ok=True)

    model_path = os.path.join(cache_dir, "ensemble.pt")

    saved_as = "state_dict"
    try:
        torch.save(ensemble.state_dict(), model_path)
    except Exception:
        saved_as = "object"
        torch.save(ensemble, model_path)

    meta = {
        "seed": int(seed),
        "dataset_trajs": int(ds),
        "obs_dim": int(obs_dim),
        "act_dim": int(act_dim),
        "saved_as": str(saved_as),
        "ensemble_cfg": ensemble_cfg,
        "files": {"ensemble_pt": os.path.basename(model_path)},
        "sha256": {"ensemble_pt": sha256_file(model_path)},
    }

    with open(os.path.join(cache_dir, "ensemble_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=_jsonable)

    print(f"[CACHE] ensemble saved at {model_path}", flush=True)

def save_config_cache(outdir, seed, ds, *, args, base_cfg, model_env_cfg, methods_cfg: dict):
    cache_dir = os.path.join(outdir, "cache", f"seed_{seed}", f"ds_{ds}")
    os.makedirs(cache_dir, exist_ok=True)

    cfg = {
        "seed": int(seed),
        "dataset_trajs": int(ds),
        "args": vars(args),
        "base_cfg": vars(base_cfg),
        "model_env_cfg": _jsonable(model_env_cfg),
        "methods_cfg": {k: vars(v) for k, v in methods_cfg.items()},
    }

    path = os.path.join(cache_dir, "config_full.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2, default=_jsonable)

    print(f"[CACHE] config saved at {path}", flush=True)


# ── Cache loading helpers ─────────────────────────────────────────────────────

def load_dataset_cache(outdir: str, seed: int, ds: int):
    """Load train/calib splits from NPZ cache. Returns (train_dict, calib_dict)."""
    cache_dir = os.path.join(outdir, "cache", f"seed_{seed}", f"ds_{ds}")
    tr_path  = os.path.join(cache_dir, "transitions_train.npz")
    cal_path = os.path.join(cache_dir, "transitions_calib.npz")
    if not os.path.exists(tr_path):
        raise FileNotFoundError(f"[CACHE] Dataset cache missing: {tr_path}")
    train = {k: v for k, v in np.load(tr_path, allow_pickle=True).items()}
    calib = {k: v for k, v in np.load(cal_path, allow_pickle=True).items()}
    print(f"[CACHE] Loaded dataset — train={train['act'].shape[0]} calib={calib['act'].shape[0]} transitions", flush=True)
    return train, calib


def load_ensemble_cache(outdir: str, seed: int, ds: int, obs_dim: int, act_dim: int):
    """Load pre-trained MSE ensemble from PT cache. Returns EnsembleDynamicsSB3."""
    cache_dir  = os.path.join(outdir, "cache", f"seed_{seed}", f"ds_{ds}")
    meta_path  = os.path.join(cache_dir, "ensemble_meta.json")
    model_path = os.path.join(cache_dir, "ensemble.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[CACHE] Ensemble cache missing: {model_path}")
    with open(meta_path) as f:
        meta = json.load(f)
    ens = build_ensemble(obs_dim, act_dim)
    saved_as = meta.get("saved_as", "object")
    if saved_as == "object":
        ens = torch.load(model_path, map_location="cpu")
    else:
        try:
            sd = torch.load(model_path, map_location="cpu")
            ens.load_state_dict(sd)
        except Exception:
            ens = torch.load(model_path, map_location="cpu")
    print(f"[CACHE] Loaded ensemble from {model_path} (saved_as={saved_as})", flush=True)
    return ens


def load_calibrator_cache(outdir: str, seed: int, ds: int, name: str):
    """Load calibrator + eps from PKL/JSON cache. Returns (calibrator, eps)."""
    cache_dir = os.path.join(outdir, "cache", f"seed_{seed}", f"ds_{ds}")
    cal_path  = os.path.join(cache_dir, f"calibrator_{name}.pkl")
    eps_path  = os.path.join(cache_dir, f"eps_{name}.json")
    if not os.path.exists(cal_path):
        raise FileNotFoundError(f"[CACHE] Calibrator cache missing: {cal_path}")
    with open(cal_path, "rb") as f:
        calibrator = pickle.load(f)
    with open(eps_path) as f:
        eps = float(json.load(f)["eps"])
    print(f"[CACHE] Loaded calibrator '{name}' (eps={eps:.4f})", flush=True)
    return calibrator, eps


# ─────────────────────────────────────────────────────────────────────────────

def _sanitize_for_df(x):
    if isinstance(x, (np.generic,)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (dict, list, tuple)):
        try:
            return json.dumps(x, default=str)
        except Exception:
            return str(x)
    return x

# -------------------------
# Main
# -------------------------

def main() -> None:
    
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--shuffle_nesting",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, shuffle traj seeds once per exp seed to remove prefix bias while keeping nesting.",
    )

    ap.add_argument("--outdir", type=str, default="out_pointmaze_gates_full")

    ap.add_argument("--continuing_task", type=int, default=0, choices=[0, 1])
    ap.add_argument("--reward_type", type=str, default="dense", choices=["sparse", "dense"])
    ap.add_argument("--max_steps", type=int, default=300)

    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--dataset_trajs", type=int, nargs="+", default=[5, 50, 100, 300])

    ap.add_argument("--ppo_train_once", type=int, default=1, choices=[0, 1])
    ap.add_argument("--ppo_global_path", type=str, default="out_pointmaze_gates_full/ppo_global.zip")
    ap.add_argument("--train_ppo_script", type=str, required=True)
    ap.add_argument("--ppo_train_seed", type=int, default=0)
    ap.add_argument("--ppo_timesteps", type=int, default=10000)

    ap.add_argument("--calib_frac", type=float, default=0.2)
    ap.add_argument("--calib_bins", type=int, default=30)
    ap.add_argument("--calib_delta", type=float, default=0.05)
    ap.add_argument("--calib_min_samples", type=int, default=50)
    ap.add_argument("--calib_stochastic", type=int, default=0, choices=[0, 1],
                    help="If 1, calibrate e_i using stochastic ensemble samples "
                         "(E_{k,eps}[||s'_stoch - s'_true||]) instead of ensemble mean. "
                         "More rigorous bound, slightly more conservative.")
    ap.add_argument("--calib_n_eps", type=int, default=10,
                    help="Noise samples per member for stochastic calibration (default: 10).")

    ap.add_argument("--eps_quantile", type=float, default=0.95)
    ap.add_argument("--eps_from_calibration", type=int, default=1, choices=[0, 1])
    ap.add_argument("--eps", type=float, default=0.10)
    ap.add_argument("--tau", type=float, default=0.2,
                    help="USAD threshold for the heuristic MOReL gate baseline "
                         "(default 0.2 matches the paper Cal-SPI / MOReL setup).")

    ap.add_argument("--num_sims", type=int, default=200)
    ap.add_argument("--max_depth", type=int, default=20)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--eval_episodes", type=int, default=10,
                    help="Number of evaluation episodes per (ds, seed) cell "
                         "(default 10 matches the paper Point Maze setup).")
    ap.add_argument("--dpw_k", type=float, default=2.0,
                    help="DPW widening constant kappa (max children = floor(kappa * N^alpha))")
    ap.add_argument("--dpw_alpha", type=float, default=0.5,
                    help="DPW widening exponent alpha")
    ap.add_argument("--c_puct", type=float, default=1.5,
                    help="PUCT exploration constant")
    ap.add_argument("--n_actions_root", type=int, default=64,
                    help="Number of root candidate actions sampled at decision time")
    ap.add_argument("--n_actions_per_node", type=int, default=16,
                    help="Per-node cap on candidate actions (non-root)")
    ap.add_argument("--ppo_std_scale", type=float, default=0.1,
                    help="Std-dev scale for PPO-anchored action sampling")
    # Reward configuration. Defaults below match the paper Cal-SPI Point Maze
    # setup (dense + speed_distance + success_bonus=1500). The PPO baseline must
    # be trained with the same reward configuration, which is what
    # train_ppo_pointmaze.py defaults to and what run_benchmark_pointmaze.sh
    # forwards.
    ap.add_argument("--living_cost", type=float, default=0.0)
    ap.add_argument("--success_bonus", type=float, default=1500.0)
    ap.add_argument("--reward_mode", type=str, default="speed_distance",
                    choices=["current", "speed", "speed_distance"])
    ap.add_argument("--step_penalty", type=float, default=1.0)
    ap.add_argument("--dist_penalty", type=float, default=0.2)

    ap.add_argument("--save_trajs_for_ds", type=int, nargs="*", default=[1, 5, 250, 500])
    ap.add_argument("--traj_eval_episodes", type=int, default=1)
    ap.add_argument("--traj_out_subdir", type=str, default="traj_debug")
    ap.add_argument("--plot_trajs", type=int, default=1, choices=[0, 1])

    # --- Debug visualizer arguments ---
    ap.add_argument(
        "--debug_episodes",
        type=int,
        default=0,
        help="Number of episodes to run with LoggingMCTS for debug plots. "
             "0 = disabled. Recommended: 2-3.",
    )
    ap.add_argument(
        "--debug_ds",
        type=int,
        nargs="*",
        default=[],
        help="Dataset sizes for which to run debug plots. "
             "Empty = all sizes. Example: --debug_ds 100 300",
    )
    ap.add_argument(
        "--debug_methods",
        type=str,
        nargs="*",
        default=["calibrated_usad", "morel_usad"],
        help="Which methods to run debug plots for. "
             "Default: calibrated_usad morel_usad",
    )
    ap.add_argument(
        "--debug_log_sims",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, log internal simulation paths (needed for Plot 2). ~10%% overhead.",
    )
    ap.add_argument(
        "--debug_sim_steps",
        type=int,
        nargs="*",
        default=None,
        help="Which real steps to generate Plot 2 for. "
             "None = every 5 steps. Example: --debug_sim_steps 0 5 10 20",
    )
    ap.add_argument(
        "--run_methods",
        type=str,
        nargs="*",
        default=None,
        help="Restrict which methods are evaluated. "
             "None = all. Example: --run_methods calibrated_variance morel_usad",
    )

    # --- Offline RL baseline arguments ---
    ap.add_argument(
        "--baseline_epochs",
        type=int,
        default=200,
        help="Training epochs for offline RL baselines (IQL, TD3+BC, MOReL-full, MOPO-MSE). "
             "Default: 200. Use a smaller value (e.g. 50) for quick validation.",
    )
    ap.add_argument(
        "--baseline_rollout_batch",
        type=int,
        default=1000,
        help="Number of model-rollout starting states per epoch for model-based baselines "
             "(MOReL-full, MOPO-MSE). Capped at len(D_train) automatically. Default: 1000.",
    )
    ap.add_argument(
        "--load_from_cache",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, skip dataset collection + ensemble training + calibration and load "
             "pre-computed structures from the cache directory (produced by a previous run). "
             "Use this to run baselines only on top of an existing Cal-SPI result directory.",
    )

    args = ap.parse_args()
    ensure_dir(args.outdir)

    # ---- PPO train-once / load ----
    if int(args.ppo_train_once) == 1 and (not os.path.exists(args.ppo_global_path)):
        run_ppo_training_to_path(
            train_ppo_script=str(args.train_ppo_script),
            seed=int(args.ppo_train_seed),
            total_timesteps=int(args.ppo_timesteps),
            continuing_task=int(args.continuing_task),
            reward_type=str(args.reward_type),
            reward_mode=str(args.reward_mode),
            step_penalty=float(args.step_penalty),
            dist_penalty=float(args.dist_penalty),
            success_bonus=float(args.success_bonus),
            living_cost=float(args.living_cost),
            out_path=str(args.ppo_global_path),
        )

    print(f"[PPO LOAD] {args.ppo_global_path}", flush=True)
    ppo = PPO.load(args.ppo_global_path, device="cpu")

    ds_list = [int(x) for x in args.dataset_trajs]
    max_ds = int(max(ds_list))

    rows: List[Dict[str, Any]] = []
    traj_plot_store: Dict[Tuple[int, int], Dict[str, List[Dict[str, Any]]]] = {}

    # When loading from cache, pre-populate rows with all existing results so they
    # are preserved in the final CSV (only rows for methods in --run_methods will be
    # re-run and overwritten; all others are kept as-is).
    _active_methods_global = set(args.run_methods) if args.run_methods else None
    if int(args.load_from_cache) == 1:
        ckpt_path = os.path.join(args.outdir, "results_checkpoint.csv")
        if os.path.exists(ckpt_path):
            df_existing = pd.read_csv(ckpt_path)
            # Keep rows for methods that are NOT being re-run
            if _active_methods_global is not None:
                df_keep = df_existing[~df_existing["method"].isin(_active_methods_global)]
            else:
                df_keep = df_existing.iloc[0:0]  # re-running everything — start fresh
            rows = df_keep.to_dict("records")
            print(f"[CACHE] Pre-loaded {len(rows)} rows from {ckpt_path} "
                  f"(keeping methods not in --run_methods)", flush=True)
        else:
            print(f"[CACHE] No existing checkpoint at {ckpt_path} — starting fresh.", flush=True)

    for seed in args.seeds:
        print(f"\n[SEED] {seed}", flush=True)

        env_real = make_env(
            int(args.continuing_task),
            str(args.reward_type),
            seed=int(seed + 123),
            living_cost=float(args.living_cost),
            success_bonus=float(args.success_bonus),
            reward_mode=str(args.reward_mode),
            step_penalty=float(args.step_penalty),
            dist_penalty=float(args.dist_penalty),
        )

        # action bounds for unsafe planner
        low = np.asarray(env_real.action_space.low, dtype=np.float32).ravel()
        high = np.asarray(env_real.action_space.high, dtype=np.float32).ravel()

        traj_seeds = trajectory_seed_list(int(seed), int(max_ds))
        if int(args.shuffle_nesting) == 1:
            traj_seeds = shuffle_traj_seeds_once(traj_seeds, int(seed))
            print("[NESTING] Using SHUFFLED prefix nesting (deterministic per seed).", flush=True)
        else:
            print("[NESTING] Using RAW prefix nesting (ordered).", flush=True)

        calib_traj_ids_global = choose_calib_traj_ids(int(seed), int(max_ds), float(args.calib_frac))

        # ---- PPO baseline eval ONCE per seed (same env, fixed episode seeds) ----
        planner_eval_seed = int(seed + 10_000)
        ppo_capture_eps = set(range(int(args.traj_eval_episodes)))

        ppo_mm, ppo_epi_arr, ppo_trajs = eval_ppo_only(
            env_real,
            ppo,
            int(args.eval_episodes),
            planner_eval_seed,
            int(args.max_steps),
            float(args.gamma),
            capture_trajs=True,
            capture_episode_indices=ppo_capture_eps,
        )
        print(f"[DONE] ppo_only | {ppo_mm}", flush=True)

        # Save PPO trajectories once per seed, then reuse for all selected ds plots
        ppo_traj_list = []
        for ei, traj in ppo_trajs.items():
            npz_path = os.path.join(
                args.outdir,
                str(args.traj_out_subdir),
                f"seed{int(seed)}",
                f"ppo_only_ep{int(ei)}.npz",
            )
            save_traj_npz(npz_path, traj)
            ppo_traj_list.append(traj)

        for ds in ds_list:
            print(f"\n[DATASET] ds={ds} (REGENERATE via traj_seeds[:ds])", flush=True)

            # Load per-(ds,seed) partial checkpoint
            _outdir = str(args.outdir)
            _completed_methods = _pm_load_completed_methods(_outdir)
            if _completed_methods:
                print(f"[CKPT] {len(_completed_methods)} method(s) already done: "
                      f"{list(_completed_methods.keys())}", flush=True)

            capture_this_ds = int(ds) in set(int(x) for x in args.save_trajs_for_ds)
            capture_eps = set(range(int(args.traj_eval_episodes))) if capture_this_ds else None

            if capture_this_ds:
                traj_plot_store[(int(seed), int(ds))] = {}
                traj_plot_store[(int(seed), int(ds))]["ppo_only"] = ppo_traj_list

            if int(args.load_from_cache) == 0:
                # ── Normal path: collect data, train ensemble, calibrate ──────────
                data = collect_dataset_from_traj_seeds(
                    env=env_real,
                    ppo=ppo,
                    traj_seeds=traj_seeds[: int(ds)],
                    max_steps=int(args.max_steps),
                )

                train, calib = split_train_calib_by_traj_ids(
                    data=data,
                    calib_traj_ids_global=calib_traj_ids_global,
                    ds=int(ds),
                )

                # SAVING dataset
                save_dataset_cache(
                    args.outdir, seed, ds, data, train, calib,
                    traj_seeds_used=traj_seeds[: int(ds)],
                    calib_traj_ids_global=calib_traj_ids_global,
                )

                obs_dim = int(train["obs_observation"].shape[1])
                act_dim = int(train["act"].shape[1])

                ens = build_ensemble(obs_dim, act_dim)
                fit_ensemble(ens, train)

                # SAVING ensemble
                save_ensemble_cache(
                    args.outdir, seed, ds, ens,
                    obs_dim=obs_dim,
                    act_dim=act_dim,
                    ensemble_cfg=None,
                )

                goal_thr_real = 0.45
                print(f"[GOAL THRESH REAL] using {goal_thr_real}", flush=True)

                model_env_cfg = ModelEnvConfig(
                    continuing_task=bool(int(args.continuing_task)),
                    reward_type=str(args.reward_type),
                    goal_thresh=float(goal_thr_real),
                    max_steps=int(args.max_steps),
                    truncate_on_time_limit=False,
                    living_cost=float(args.living_cost),
                    success_bonus=float(args.success_bonus),
                    reward_mode=str(args.reward_mode),
                    step_penalty=float(args.step_penalty),
                    dist_penalty=float(args.dist_penalty),
                )
                model_env = ModelPointMazeEnv(ensemble=ens, cfg=model_env_cfg)

                print("[DEBUG] traj_id in calib:", "traj_id" in calib, flush=True)
                cal_usad, eps_usad_emp = fit_calibrator_and_eps(
                    ens=ens,
                    calib=calib,
                    kind="usad",
                    seed=int(seed + 3000),
                    eps_quantile=float(args.eps_quantile),
                    n_bins=int(args.calib_bins),
                    delta=float(args.calib_delta),
                    min_samples=int(args.calib_min_samples),
                    max_steps=int(args.max_steps),
                    stochastic=bool(int(args.calib_stochastic)),
                    n_eps=int(args.calib_n_eps),
                )
                cal_var, eps_var_emp = fit_calibrator_and_eps(
                    ens=ens,
                    calib=calib,
                    kind="epistemic",
                    seed=int(seed + 4000),
                    eps_quantile=float(args.eps_quantile),
                    n_bins=int(args.calib_bins),
                    delta=float(args.calib_delta),
                    min_samples=int(args.calib_min_samples),
                    max_steps=int(args.max_steps),
                    stochastic=bool(int(args.calib_stochastic)),
                    n_eps=int(args.calib_n_eps),
                )

                save_calibrator_cache(args.outdir, seed, ds, name="usad", calibrator=cal_usad, eps=float(eps_usad_emp))
                save_calibrator_cache(args.outdir, seed, ds, name="epistemic", calibrator=cal_var, eps=float(eps_var_emp))

            else:
                # ── Cache path: load pre-computed structures ───────────────────────
                print(f"[CACHE] Loading from cache: seed={seed} ds={ds}", flush=True)
                train, calib = load_dataset_cache(args.outdir, seed, ds)

                obs_dim = int(train["obs_observation"].shape[1])
                act_dim = int(train["act"].shape[1])

                ens = load_ensemble_cache(args.outdir, seed, ds, obs_dim, act_dim)

                goal_thr_real = 0.45
                model_env_cfg = ModelEnvConfig(
                    continuing_task=bool(int(args.continuing_task)),
                    reward_type=str(args.reward_type),
                    goal_thresh=float(goal_thr_real),
                    max_steps=int(args.max_steps),
                    truncate_on_time_limit=False,
                    living_cost=float(args.living_cost),
                    success_bonus=float(args.success_bonus),
                    reward_mode=str(args.reward_mode),
                    step_penalty=float(args.step_penalty),
                    dist_penalty=float(args.dist_penalty),
                )
                model_env = ModelPointMazeEnv(ensemble=ens, cfg=model_env_cfg)

                cal_usad, eps_usad_emp = load_calibrator_cache(args.outdir, seed, ds, name="usad")
                cal_var,  eps_var_emp  = load_calibrator_cache(args.outdir, seed, ds, name="epistemic")

            eps_usad = float(eps_usad_emp) if int(args.eps_from_calibration) == 1 else float(args.eps)
            eps_var = float(eps_var_emp) if int(args.eps_from_calibration) == 1 else float(args.eps)

            base_cfg = OfflineMCTSConfig(
                num_sims=int(args.num_sims),
                max_depth=int(args.max_depth),
                gamma=float(args.gamma),
                gate_inside_tree=True,
                model_sampling="fixed_per_sim",
                conservative_on_low_data=True,

                ppo_std_scale=float(args.ppo_std_scale),
                dpw_k=float(args.dpw_k),
                dpw_alpha=float(args.dpw_alpha),
                n_actions_root=int(args.n_actions_root),
                n_actions_per_node=int(args.n_actions_per_node),
                c_puct=float(args.c_puct),
            )

            # --- PPO baseline row replicated across ds (flat line by construction) ---
            _ppo_row = {
                "seed": int(seed),
                "dataset_trajs": int(ds),
                "method": "ppo_only",
                "tau": float(args.tau),
                "eps_usad": float(eps_usad),
                "eps_var": float(eps_var),
                "reward_type": str(args.reward_type),
                "continuing_task": int(args.continuing_task),
                "living_cost": float(args.living_cost),
                "success_bonus": float(args.success_bonus),
                "reward_mode": str(args.reward_mode),
                "step_penalty": float(args.step_penalty),
                "dist_penalty": float(args.dist_penalty),
                **ppo_mm,
            }
            rows.append(_ppo_row)
            if "ppo_only" not in _completed_methods:
                _pm_save_partial_result(_outdir, _ppo_row)

            # ------------------------------------------------------------------
            # Debug helper: runs LoggingMCTS for a few episodes and saves plots
            # Controlled by --debug_episodes, --debug_ds, --debug_methods, etc.
            # ------------------------------------------------------------------
            def run_debug_method(
                method: str,
                cfg: OfflineMCTSConfig,
                calibrator: Optional[Any],
                use_gate: bool,
                use_ppo: bool,
            ):
                if not _DEBUG_VIZ_AVAILABLE:
                    return
                if int(args.debug_episodes) <= 0:
                    return
                if method not in set(args.debug_methods):
                    return
                if args.debug_ds and int(ds) not in set(int(x) for x in args.debug_ds):
                    return

                print(
                    f"[DEBUG VIZ] running {int(args.debug_episodes)} debug episode(s) "
                    f"for {method} | seed={seed} ds={ds}",
                    flush=True,
                )

                debug_planner = LoggingMCTS(
                    model_env=model_env,
                    ppo_policy=ppo,
                    cfg=cfg,
                    calibrator=calibrator,
                    use_gate=bool(use_gate),
                    use_ppo_prior=bool(use_ppo),
                    use_ppo_leaf_value=bool(use_ppo),
                    rng_seed=int(seed + 999),
                    log_sims=bool(int(args.debug_log_sims)),
                )

                # Use the same eval seed as the main eval loop, one per episode
                base_ep_seed = int(planner_eval_seed)

                for ep_idx in range(int(args.debug_episodes)):
                    ep_seed = base_ep_seed + int(ep_idx)

                    logs = run_debug_episode(
                        env_real,
                        debug_planner,
                        seed=ep_seed,
                        max_steps=int(args.max_steps),
                        reset_logs=True,
                    )

                    if not logs:
                        print(
                            f"[DEBUG VIZ] {method} ep{ep_idx}: no logs collected, skipping",
                            flush=True,
                        )
                        continue

                    # Retrieve the goal position for this episode seed
                    goal_pos = None
                    try:
                        obs_tmp, _ = env_real.reset(seed=ep_seed)
                        goal_pos = np.asarray(
                            obs_tmp.get("desired_goal", [0.0, 0.0]), dtype=np.float32
                        )
                    except Exception:
                        pass

                    # Which real steps to include in Plot 2
                    if args.debug_sim_steps is not None:
                        sim_steps = [int(t) for t in args.debug_sim_steps]
                    else:
                        sim_steps = list(range(0, len(logs), 5))  # every 5 steps by default

                    out_ep_dir = os.path.join(
                        args.outdir, "debug", method,
                        f"seed{int(seed)}_ds{int(ds)}_ep{int(ep_idx)}",
                    )

                    plot_debug_episode(
                        logs,
                        method_name=method,
                        out_dir=out_ep_dir,
                        steps_for_sim_plots=sim_steps,
                        goal_pos=goal_pos,
                        # PointMazeMedium default limits — adjust if your maze differs
                        maze_xlim=(-3.5, 3.5),
                        maze_ylim=(-3.5, 3.5),
                        show_detail_step=(
                            sim_steps[len(sim_steps) // 2] if sim_steps else None
                        ),
                    )

                    print(f"[DEBUG VIZ] {method} ep{ep_idx} plots -> {out_ep_dir}/", flush=True)

            def run_method(method: str, cfg: OfflineMCTSConfig, calibrator: Optional[Any], use_gate: bool, use_ppo: bool):
                if method in _completed_methods:
                    print(f"[CKPT] {method} already done — loading from checkpoint", flush=True)
                    rows.append(_completed_methods[method])
                    return

                ckpt_m = _PMEpisodeCheckpoint(_outdir, method)
                planner = OfflineMCTS_DPW_PPO_SPIBB_Parametric(
                    model_env=model_env,
                    ppo_policy=ppo,
                    cfg=cfg,
                    calibrator=calibrator,
                    use_gate=bool(use_gate),
                    use_ppo_prior=bool(use_ppo),
                    use_ppo_leaf_value=bool(use_ppo),
                    rng_seed=int(seed + 999),
                )

                mm, mm_epi_arr, traj_payload = eval_planner(
                    env_real,
                    planner,
                    int(args.eval_episodes),
                    planner_eval_seed,
                    int(args.max_steps),
                    float(args.gamma),
                    capture_trajs=capture_this_ds,
                    capture_episode_indices=capture_eps,
                    ckpt=ckpt_m,
                )
                mm.update(_delta_metrics(mm_epi_arr["R"] - ppo_epi_arr["R"], "delta_return"))
                mm.update(_delta_metrics(mm_epi_arr["RD"] - ppo_epi_arr["RD"], "delta_discounted"))

                low_data = bool(getattr(calibrator, "meta", {}).get("low_data_regime", False)) if calibrator is not None else False
                tau_val = float(args.tau)
                print(
                    f"[GATE DEBUG] seed={seed} ds={ds} method={method} | "
                    f"tau={tau_val:.4f} eps_usad={eps_usad:.4f} eps_var={eps_var:.4f} | "
                    f"low_data={int(low_data)} coverage={mm['root_coverage_mean']:.4f} "
                    f"fallback={mm['root_fallback_rate']:.4f}",
                    flush=True,
                )

                _row = {
                    "seed": int(seed),
                    "dataset_trajs": int(ds),
                    "method": method,
                    "tau": float(args.tau),
                    "eps_usad": float(eps_usad),
                    "eps_var": float(eps_var),
                    "reward_type": str(args.reward_type),
                    "continuing_task": int(args.continuing_task),
                    "living_cost": float(args.living_cost),
                    "success_bonus": float(args.success_bonus),
                    "reward_mode": str(args.reward_mode),
                    "step_penalty": float(args.step_penalty),
                    "dist_penalty": float(args.dist_penalty),
                    **mm,
                }
                rows.append(_row)
                _pm_save_partial_result(_outdir, _row)
                print(f"[DONE] {method} | {mm}", flush=True)

                if capture_this_ds:
                    method_traj_list = []
                    for ei, traj in traj_payload.items():
                        npz_path = os.path.join(
                            args.outdir,
                            str(args.traj_out_subdir),
                            f"ds{int(ds)}",
                            f"seed{int(seed)}",
                            f"{method}_ep{int(ei)}.npz",
                        )
                        save_traj_npz(npz_path, traj)
                        method_traj_list.append(traj)

                    traj_plot_store[(int(seed), int(ds))][method] = method_traj_list

            def run_unsafe_model_only(method: str, cfg: OfflineMCTSConfig):
                if method in _completed_methods:
                    print(f"[CKPT] {method} already done — loading from checkpoint", flush=True)
                    rows.append(_completed_methods[method])
                    return

                ckpt_m = _PMEpisodeCheckpoint(_outdir, method)
                planner = OfflineMCTS_DPW_Ensemble_UNSAFE(
                    model_env=model_env,
                    cfg=cfg,
                    action_low=low,
                    action_high=high,
                    rng_seed=int(seed + 999),
                )

                mm, mm_epi_arr, traj_payload = eval_planner(
                    env_real,
                    planner,
                    int(args.eval_episodes),
                    planner_eval_seed,
                    int(args.max_steps),
                    float(args.gamma),
                    capture_trajs=capture_this_ds,
                    capture_episode_indices=capture_eps,
                    ckpt=ckpt_m,
                )
                mm.update(_delta_metrics(mm_epi_arr["R"] - ppo_epi_arr["R"], "delta_return"))
                mm.update(_delta_metrics(mm_epi_arr["RD"] - ppo_epi_arr["RD"], "delta_discounted"))

                _row = {
                    "seed": int(seed),
                    "dataset_trajs": int(ds),
                    "method": method,
                    "tau": float(args.tau),
                    "eps_usad": float(eps_usad),
                    "eps_var": float(eps_var),
                    "reward_type": str(args.reward_type),
                    "continuing_task": int(args.continuing_task),
                    "living_cost": float(args.living_cost),
                    "success_bonus": float(args.success_bonus),
                    "reward_mode": str(args.reward_mode),
                    "step_penalty": float(args.step_penalty),
                    "dist_penalty": float(args.dist_penalty),
                    **mm,
                }
                rows.append(_row)
                _pm_save_partial_result(_outdir, _row)
                print(f"[DONE] {method} | {mm}", flush=True)

                if capture_this_ds:
                    method_traj_list = []
                    for ei, traj in traj_payload.items():
                        npz_path = os.path.join(
                            args.outdir,
                            str(args.traj_out_subdir),
                            f"ds{int(ds)}",
                            f"seed{int(seed)}",
                            f"{method}_ep{int(ei)}.npz",
                        )
                        save_traj_npz(npz_path, traj)
                        method_traj_list.append(traj)

                    traj_plot_store[(int(seed), int(ds))][method] = method_traj_list

            def run_baseline(method: str, agent) -> None:
                """Evaluate a pre-trained offline RL baseline on env_real."""
                if method in _completed_methods:
                    print(f"[CKPT] {method} already done — loading from checkpoint", flush=True)
                    rows.append(_completed_methods[method])
                    return

                def act_fn(obs):
                    return agent.act(obs)

                ckpt_m = _PMEpisodeCheckpoint(_outdir, method)
                mm, mm_epi_arr, _ = _eval_policy_episodic(
                    env_real, act_fn,
                    int(args.eval_episodes), planner_eval_seed,
                    int(args.max_steps), float(args.gamma),
                    ckpt=ckpt_m,
                )
                mm.update(_delta_metrics(mm_epi_arr["R"]  - ppo_epi_arr["R"],  "delta_return"))
                mm.update(_delta_metrics(mm_epi_arr["RD"] - ppo_epi_arr["RD"], "delta_discounted"))

                _row = {
                    "seed": int(seed),
                    "dataset_trajs": int(ds),
                    "method": method,
                    "tau": float(args.tau),
                    "eps_usad": float(eps_usad),
                    "eps_var": float(eps_var),
                    "reward_type": str(args.reward_type),
                    "continuing_task": int(args.continuing_task),
                    "living_cost": float(args.living_cost),
                    "success_bonus": float(args.success_bonus),
                    "reward_mode": str(args.reward_mode),
                    "step_penalty": float(args.step_penalty),
                    "dist_penalty": float(args.dist_penalty),
                    **mm,
                }
                rows.append(_row)
                _pm_save_partial_result(_outdir, _row)
                print(f"[DONE] {method} | {mm}", flush=True)

            # Helper: returns True if this method should be run
            _active_methods = set(args.run_methods) if args.run_methods else None
            def _should_run(method: str) -> bool:
                return _active_methods is None or method in _active_methods

            # 1) calibrated_usad  (B(u) <= eps)
            cfg_cu = OfflineMCTSConfig(**vars(base_cfg))
            cfg_cu.gate_mode = "calibrated"
            cfg_cu.uncertainty_type = "usad"
            cfg_cu.eps_threshold = float(eps_usad)
            if _should_run("calibrated_usad"):
                run_method("calibrated_usad", cfg_cu, calibrator=cal_usad, use_gate=True, use_ppo=True)
                run_debug_method("calibrated_usad", cfg_cu, calibrator=cal_usad, use_gate=True, use_ppo=True)

            # 2) calibrated_variance (epistemic)
            cfg_cv = OfflineMCTSConfig(**vars(base_cfg))
            cfg_cv.gate_mode = "calibrated"
            cfg_cv.uncertainty_type = "epistemic"
            cfg_cv.eps_threshold = float(eps_var)
            if _should_run("calibrated_variance"):
                run_method("calibrated_variance", cfg_cv, calibrator=cal_var, use_gate=True, use_ppo=True)
                run_debug_method("calibrated_variance", cfg_cv, calibrator=cal_var, use_gate=True, use_ppo=True)

            # 3) morel_usad  (uncertainty gate u <= tau)
            cfg_morel = OfflineMCTSConfig(**vars(base_cfg))
            cfg_morel.gate_mode = "uncertainty"
            cfg_morel.uncertainty_type = "usad"
            cfg_morel.uncertainty_threshold = float(args.tau)
            if _should_run("morel_usad"):
                run_method("morel_usad", cfg_morel, calibrator=None, use_gate=True, use_ppo=True)
                run_debug_method("morel_usad", cfg_morel, calibrator=None, use_gate=True, use_ppo=True)

            # 4) UNSAFE pure model-only MCTS (no PPO, no gate)
            cfg_unsafe = OfflineMCTSConfig(**vars(base_cfg))
            if _should_run("mcts_only_no_gate"):
                run_unsafe_model_only("mcts_only_no_gate", cfg_unsafe)

            methods_cfg = {
                "mcts_only_no_gate": cfg_unsafe,
                "morel_usad": cfg_morel,
                "calibrated_usad": cfg_cu,
                "calibrated_variance": cfg_cv,
            }
            save_config_cache(
                args.outdir, seed, ds,
                args=args,
                base_cfg=base_cfg,
                model_env_cfg=model_env_cfg,
                methods_cfg=methods_cfg,
            )

            # ── Offline RL Baselines ───────────────────────────────────────────
            # Baselines train on the FULL dataset (D_train ∪ D_calib).
            # The train/calib split is an internal Cal-SPI detail for gate
            # calibration; IQL/TD3+BC/MOReL/MOPO have no calibration step,
            # so they should use all ds trajectories — consistent with how
            # these methods are evaluated in D4RL/original papers.
            # Note: for small ds (e.g. ds=1), D_train may be empty because
            # trajectory 0 was globally pre-assigned to calibration; using
            # the full dataset avoids the degenerate empty-train edge case.
            # ──────────────────────────────────────────────────────────────────
            data_for_baselines = {
                k: np.concatenate([train[k], calib[k]], axis=0)
                for k in train.keys()
                if k in calib
            }
            goal_dim  = int(train["obs_desired_goal"].shape[1])
            state_dim = obs_dim + goal_dim
            n_bl_ep   = int(args.baseline_epochs)
            rb        = min(int(args.baseline_rollout_batch), len(data_for_baselines["act"]))

            # 5) IQL
            # PointMaze is a navigation task with sparse reward → AntMaze-style hyperparameters
            # (Kostrikov 2021 Table 5: alpha=0.9, beta=10.0 for AntMaze vs 0.7/3.0 for locomotion)
            if _should_run("iql"):
                iql_agent = IQLAgent(
                    obs_dim=state_dim, act_dim=act_dim,
                    cfg=IQLConfig(hidden_dim=256, n_hidden=3, batch_size=256,
                                  lr=3e-4, alpha=0.9, beta=10.0,
                                  normalize_states=True, seed=int(seed)),
                )
                if "iql" not in _completed_methods:
                    if not _pm_load_baseline_model(_outdir, "iql", iql_agent):
                        print(f"[BASELINE] iql seed={seed} ds={ds} epochs={n_bl_ep}", flush=True)
                        iql_agent.train(data_for_baselines, n_epochs=n_bl_ep)
                        _pm_save_baseline_model(_outdir, "iql", iql_agent)
                run_baseline("iql", iql_agent)

            # 6) TD3+BC — same hyperparameters across all domains (Fujimoto & Gu 2021 Table 1)
            if _should_run("td3bc"):
                td3_agent = TD3BCAgent(
                    obs_dim=state_dim, act_dim=act_dim,
                    cfg=TD3BCConfig(hidden_dim=256, n_hidden=2, batch_size=256,
                                    lr=3e-4, alpha_bc=2.5,
                                    normalize_states=True, seed=int(seed)),
                )
                if "td3bc" not in _completed_methods:
                    if not _pm_load_baseline_model(_outdir, "td3bc", td3_agent):
                        print(f"[BASELINE] td3bc seed={seed} ds={ds} epochs={n_bl_ep}", flush=True)
                        td3_agent.train(data_for_baselines, n_epochs=n_bl_ep)
                        _pm_save_baseline_model(_outdir, "td3bc", td3_agent)
                run_baseline("td3bc", td3_agent)

            # 7) MOReL-full (reuses pre-trained MSE ensemble — no re-training)
            # tau_percentile=95: data-driven threshold (MOReL paper: heuristic on D_train USAD)
            if _should_run("morel_full"):
                morel_agent = MORelFullAgent(
                    ensemble=ens, obs_dim=state_dim, act_dim=act_dim,
                    cfg=MORelFullConfig(tau=float(args.tau), tau_percentile=95.0,
                                        kappa=50.0, rollout_length=5, rollout_batch=rb,
                                        hidden_dim=256, batch_size=256, seed=int(seed)),
                )
                if "morel_full" not in _completed_methods:
                    if not _pm_load_baseline_model(_outdir, "morel_full", morel_agent):
                        print(f"[BASELINE] morel_full seed={seed} ds={ds} epochs={n_bl_ep}", flush=True)
                        morel_agent.train(data_for_baselines, n_epochs=n_bl_ep)
                        _pm_save_baseline_model(_outdir, "morel_full", morel_agent)
                run_baseline("morel_full", morel_agent)

            # 8) MOPO-MSE
            if _should_run("mopo_mse"):
                mopo_agent = MOPOMSEAgent(
                    ensemble=ens, obs_dim=state_dim, act_dim=act_dim,
                    cfg=MOPOMSEConfig(lam=1.0, rollout_length=5, rollout_batch=rb,
                                      hidden_dim=256, batch_size=256, seed=int(seed)),
                )
                if "mopo_mse" not in _completed_methods:
                    if not _pm_load_baseline_model(_outdir, "mopo_mse", mopo_agent):
                        print(f"[BASELINE] mopo_mse seed={seed} ds={ds} epochs={n_bl_ep}", flush=True)
                        mopo_agent.train(data_for_baselines, n_epochs=n_bl_ep)
                        _pm_save_baseline_model(_outdir, "mopo_mse", mopo_agent)
                run_baseline("mopo_mse", mopo_agent)

            # 9) CQL
            if _should_run("cql"):
                cql_agent = CQLAgent(
                    obs_dim=state_dim, act_dim=act_dim,
                    cfg=CQLConfig(hidden_dim=256, n_hidden=3, batch_size=256, lr=3e-4,
                                  alpha_sac=0.1, alpha_cql=5.0, n_action_samples=10,
                                  normalize_states=True, seed=int(seed)),
                )
                if "cql" not in _completed_methods:
                    if not _pm_load_baseline_model(_outdir, "cql", cql_agent):
                        print(f"[BASELINE] cql seed={seed} ds={ds} epochs={n_bl_ep}", flush=True)
                        cql_agent.train(data_for_baselines, n_epochs=n_bl_ep)
                        _pm_save_baseline_model(_outdir, "cql", cql_agent)
                run_baseline("cql", cql_agent)

            # 10) BEAR
            if _should_run("bear"):
                bear_agent = BEARAgent(
                    obs_dim=state_dim, act_dim=act_dim,
                    cfg=BEARConfig(hidden_dim=256, n_hidden=3, vae_hidden_dim=750, vae_latent_dim=0,
                                   lr=3e-4, lr_vae=3e-4, batch_size=256,
                                   alpha_sac=0.05, delta_mmd=0.05,
                                   kernel="laplacian", kernel_bw=20.0,
                                   n_mmd_samples=4, n_vae_samples=10,
                                   normalize_states=True, seed=int(seed)),
                )
                if "bear" not in _completed_methods:
                    if not _pm_load_baseline_model(_outdir, "bear", bear_agent):
                        print(f"[BASELINE] bear seed={seed} ds={ds} epochs={n_bl_ep}", flush=True)
                        bear_agent.train(data_for_baselines, n_epochs=n_bl_ep)
                        _pm_save_baseline_model(_outdir, "bear", bear_agent)
                run_baseline("bear", bear_agent)

            # 11) Cal-QL
            if _should_run("cal_ql"):
                cal_ql_agent = CalQLAgent(
                    obs_dim=state_dim, act_dim=act_dim,
                    cfg=CalQLConfig(hidden_dim=256, n_hidden=3, batch_size=256, lr=3e-4,
                                    alpha_sac=0.1, alpha_cql=5.0, n_action_samples=10,
                                    normalize_states=True, k_mc=5, lambda_cal=1.0,
                                    seed=int(seed)),
                )
                if "cal_ql" not in _completed_methods:
                    if not _pm_load_baseline_model(_outdir, "cal_ql", cal_ql_agent):
                        print(f"[BASELINE] cal_ql seed={seed} ds={ds} epochs={n_bl_ep}", flush=True)
                        cal_ql_agent.train(data_for_baselines, n_epochs=n_bl_ep)
                        _pm_save_baseline_model(_outdir, "cal_ql", cal_ql_agent)
                run_baseline("cal_ql", cal_ql_agent)

            rows_sane = [{k: _sanitize_for_df(v) for k, v in r.items()} for r in rows]
            df_partial = pd.DataFrame.from_records(rows_sane)
            save_checkpoint(df_partial, args.outdir)

            if capture_this_ds and int(args.plot_trajs) == 1:
                plot_path = os.path.join(
                    args.outdir,
                    str(args.traj_out_subdir),
                    f"traj_plot_ds{int(ds)}_seed{int(seed)}.png",
                )
                plot_trajectories_2d(
                    traj_plot_store[(int(seed), int(ds))],
                    outpath=plot_path,
                    title=f"PointMaze trajectories | ds={int(ds)} seed={int(seed)}",
                )
                print(f"[TRAJ PLOT] saved {plot_path}", flush=True)

        env_real.close()

    # ---- save ----
    rows_sane = [{k: _sanitize_for_df(v) for k, v in r.items()} for r in rows]
    df_partial = pd.DataFrame.from_records(rows_sane)
    save_checkpoint(df_partial, args.outdir)
    out_csv = os.path.join(args.outdir, "results.csv")
    df_partial.to_csv(out_csv, index=False)
    print(f"\n[SAVED] {out_csv}", flush=True)

    # ---- plots ----
    plot_metric(
        df_partial,
        "dataset_trajs",
        "return_mean",
        os.path.join(args.outdir, "plot_mean.png"),
        "Return mean vs dataset size",
    )

    plot_metric(
        df_partial,
        "dataset_trajs",
        "cvar15",
        os.path.join(args.outdir, "plot_cvar15.png"),
        "CVaR(15%) vs dataset size",
    )

    plot_metric(
        df_partial,
        "dataset_trajs",
        "success_rate",
        os.path.join(args.outdir, "plot_success.png"),
        "Success rate vs dataset size",
    )

    print("[DONE] plots saved", flush=True)

if __name__ == "__main__":
    main()


# OUT=results_pointmaze_speeddist_trajs

# for DS in 10 100 1000; do
# python scripts/offline/pointmaze/benchmark_pointmaze_gates_full.py \
#   --outdir "${OUT}/ds${DS}/seed0" \
#   --train_ppo_script scripts/offline/pointmaze/train_ppo_pointmaze_suboptimal.py \
#   --ppo_train_once 1 \
#   --ppo_global_path models/ppo_global_pointmaze_speeddist_40k.zip \
#   --ppo_train_seed 0 \
#   --ppo_timesteps 40000 \
#   --reward_type dense \
#   --reward_mode speed_distance \
#   --step_penalty 1.0 \
#   --dist_penalty 0.2 \
#   --success_bonus 1500.0 \
#   --living_cost 0.0 \
#   --continuing_task 0 \
#   --dataset_trajs "${DS}" \
#   --seeds 0 \
#   --num_sims 100 \
#   --max_depth 10 \
#   --gamma 0.99 \
#   --calib_frac 0.2 \
#   --calib_bins 30 \
#   --calib_delta 0.05 \
#   --calib_min_samples 50 \
#   --eps_from_calibration 1 \
#   --eps_quantile 0.95 \
#   --tau 0.2 \
#   --eval_episodes 40 \
#   --max_steps 300 \
#   --save_trajs_for_ds 10 100 1000 \
#   --traj_eval_episodes 1 \
#   --plot_trajs 1
# done




