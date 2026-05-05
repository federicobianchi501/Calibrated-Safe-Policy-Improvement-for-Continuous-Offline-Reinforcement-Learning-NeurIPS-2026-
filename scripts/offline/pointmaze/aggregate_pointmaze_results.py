# =============================================================================
# Anonymous authors. Affiliations and acknowledgements will be added on
# acceptance of the associated paper. Submitted to NeurIPS 2026 (double-blind).
# =============================================================================
"""
Aggregate PointMaze benchmark results across seeds and produce paper-ready plots.

Assumes per-run outputs like:
<root>/ds{DS}/seed{SEED}/results.csv

This patched version:
1) loads all results.csv under root
2) normalizes key columns
3) builds TWO aggregations:
   - absolute metrics over seeds for each (dataset_trajs, method)
   - paired delta-vs-baseline metrics over seeds for each (dataset_trajs, method)
4) saves:
   - aggregated_results.csv                      # absolute aggregation (backward-compatible name)
   - aggregated_results_absolute.csv            # same as above
   - aggregated_results_delta_vs_ppo.csv        # paired deltas vs baseline
   - plot_return_mean.svg                       # absolute
   - plot_cvar15.svg                            # absolute
   - plot_success_rate.svg                      # absolute
   - plot_delta_return_mean_vs_ppo.svg
   - plot_delta_cvar15_vs_ppo.svg
   - plot_delta_success_rate_vs_ppo.svg
   - optional trajectory plots
   - optional main 4-panel figure

Trajectory plotting mode:
- produces ONE FIGURE PER SAVED EPISODE INDEX
- each figure overlays all methods for the same episode index
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import re
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METRICS_DEFAULT = ["return_mean", "cvar15", "success_rate"]
DEFAULT_XLABEL = r"Number of trajectories in dataset $\mathcal{D}$"

BAND_MODE_CHOICES = ["none", "std", "sem", "ci95"]

USER_METHODS_LAST = [
    "morel_usad",
    "calibrated_variance",
    "calibrated_usad",
]

BASELINE_METHOD = "ppo_only"


def _legend_ncol(n_items: int, max_cols: int = 4) -> int:
    if n_items <= 0:
        return 1
    return min(max_cols, n_items)


def _legend_top_rect(n_items: int, ncol: int) -> float:
    nrows = max(1, math.ceil(n_items / ncol))
    return max(0.72, 0.90 - 0.08 * (nrows - 1))

# Keys MUST match the `method` strings appearing in results.csv.
METHOD_STYLE: Dict[str, Dict[str, object]] = {
    "ppo_only": {
        "label": "Baseline",
        "color": "#E69F00",
        "linestyle": "-",
        "marker": None,
    },
    "mcts_only_no_gate": {
        "label": "MCTS-DPW",
        "color": "#0072B2",
        "linestyle": "--",
        "marker": "v",
    },
    "morel_usad": {
        "label": "SPI-U",
        "color": "#6A3D9A",
        "linestyle": "-",
        "marker": "s",
    },
    "calibrated_variance": {
        "label": "CSPI-V",
        "color": "#D62728",
        "linestyle": "-",
        "marker": "^",
    },
    "calibrated_usad": {
        "label": "CSPI-U",
        "color": "#1B9E77",
        "linestyle": "-",
        "marker": "o",
    },
    "mopo_mse": {
        "label": "MOPO",
        "color": "#56B4E9",
        "linestyle": ":",
        "marker": "D",
    },
    "morel_full": {
        "label": "MOReL",
        "color": "#CC79A7",
        "linestyle": "-.",
        "marker": "P",
    },
    "iql": {
        "label": "IQL",
        "color": "#7F7F7F",
        "linestyle": "--",
        "marker": "X",
    },
    "td3bc": {
        "label": "TD3+BC",
        "color": "#8C564B",
        "linestyle": ":",
        "marker": "*",
    },
    "bear": {
        "label": "BEAR",
        "color": "#F4A582",
        "linestyle": "--",
        "marker": "h",
    },
    "cql": {
        "label": "CQL",
        "color": "#92C5DE",
        "linestyle": "-.",
        "marker": "H",
    },
    "cal_ql": {
        "label": "Cal-QL",
        "color": "#4DAC26",
        "linestyle": ":",
        "marker": "8",
    },
}

TRAJ_METHOD_ORDER = [
    "ppo_only",
    "mcts_only_no_gate",
    "morel_usad",
    "calibrated_usad",
    "calibrated_variance",
]


# ============================================================================
# Trajectory helpers
# ============================================================================

def _find_traj_npz(root: str, ds: int, seed: int) -> Dict[str, List[str]]:
    """
    Expected layout from benchmark:
      PPO:
        <root>/<traj_out_subdir>/seed{SEED}/ppo_only_ep{EI}.npz
      other methods:
        <root>/<traj_out_subdir>/ds{DS}/seed{SEED}/{method}_ep{EI}.npz
    """
    out: Dict[str, List[str]] = {}

    ppo_pattern = os.path.join(root, "**", f"seed{seed}", "ppo_only_ep*.npz")
    ppo_paths = sorted(glob.glob(ppo_pattern, recursive=True))
    if ppo_paths:
        out["ppo_only"] = ppo_paths

    for method in TRAJ_METHOD_ORDER:
        if method == "ppo_only":
            continue
        patt = os.path.join(root, "**", f"ds{ds}", f"seed{seed}", f"{method}_ep*.npz")
        paths = sorted(glob.glob(patt, recursive=True))
        if paths:
            out[method] = paths

    return out


def _load_traj_npz(path: str) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    out = {
        "xy": np.asarray(z["xy"], dtype=np.float32),
        "start": np.asarray(z["start"], dtype=np.float32),
        "goal": np.asarray(z["goal"], dtype=np.float32),
    }
    if "rewards" in z:
        out["rewards"] = np.asarray(z["rewards"], dtype=np.float32)
    if "done_flags" in z:
        out["done_flags"] = np.asarray(z["done_flags"], dtype=np.int32)
    if "n_steps" in z:
        out["n_steps"] = int(np.asarray(z["n_steps"]).reshape(-1)[0])
    if "success" in z:
        out["success"] = int(np.asarray(z["success"]).reshape(-1)[0])
    if "terminated" in z:
        out["terminated"] = int(np.asarray(z["terminated"]).reshape(-1)[0])
    if "return" in z:
        out["return"] = float(np.asarray(z["return"]).reshape(-1)[0])
    if "discounted_return" in z:
        out["discounted_return"] = float(np.asarray(z["discounted_return"]).reshape(-1)[0])
    if "return_wo_lc" in z:
        out["return_wo_lc"] = float(np.asarray(z["return_wo_lc"]).reshape(-1)[0])
    if "time_to_success" in z:
        out["time_to_success"] = float(np.asarray(z["time_to_success"]).reshape(-1)[0])
    out["__source_path"] = path
    return out


def _load_trajs_for_ds_seed(
    root: str,
    ds: int,
    seed: int,
    max_episodes_per_method: Optional[int] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    found = _find_traj_npz(root, ds=ds, seed=seed)
    out: Dict[str, List[Dict[str, Any]]] = {}

    for method, paths in found.items():
        if max_episodes_per_method is not None and int(max_episodes_per_method) > 0:
            paths = paths[: int(max_episodes_per_method)]
        out[method] = [_load_traj_npz(p) for p in paths]

    return out


def plot_trajectories_for_ds_seed(
    root: str,
    ds: int,
    seed: int,
    outdir: str,
    *,
    max_episodes_per_method: int = 1,
    filename: Optional[str] = None,
) -> Optional[str]:
    trajs_by_method = _load_trajs_for_ds_seed(
        root=root,
        ds=int(ds),
        seed=int(seed),
        max_episodes_per_method=int(max_episodes_per_method),
    )

    if len(trajs_by_method) == 0:
        print(f"[WARN] No trajectory npz found for ds={ds}, seed={seed}")
        return None

    os.makedirs(outdir, exist_ok=True)

    if filename is None:
        filename = f"plot_trajs_ds{int(ds)}_seed{int(seed)}.svg"
    outpath = os.path.join(outdir, filename)

    plt.figure(figsize=(7.0, 7.0))
    ax = plt.gca()

    plotted_start = False
    plotted_goal = False

    for method in TRAJ_METHOD_ORDER:
        if method not in trajs_by_method:
            continue

        style = METHOD_STYLE.get(method, {})
        label = style.get("label", method)
        color = style.get("color", None)

        for idx, traj in enumerate(trajs_by_method[method]):
            xy = np.asarray(traj["xy"], dtype=np.float32)
            start = np.asarray(traj["start"], dtype=np.float32)
            goal = np.asarray(traj["goal"], dtype=np.float32)

            ax.plot(
                xy[:, 0],
                xy[:, 1],
                linewidth=2.0,
                alpha=0.9,
                color=color,
                label=label if idx == 0 else None,
            )

            ax.scatter(
                xy[-1, 0],
                xy[-1, 1],
                s=28,
                color=color,
                alpha=0.9,
            )

            if not plotted_start:
                ax.scatter(
                    start[0],
                    start[1],
                    s=90,
                    color="black",
                    marker="o",
                    label="Start",
                    zorder=5,
                )
                plotted_start = True

            if not plotted_goal:
                goal_radius = 0.45
                goal_region = plt.Circle(
                    (float(goal[0]), float(goal[1])),
                    goal_radius,
                    color="orange",
                    alpha=0.15,
                    ec="orange",
                    lw=1.5,
                    fill=True,
                    zorder=2,
                )
                ax.add_patch(goal_region)

                ax.scatter(
                    goal[0],
                    goal[1],
                    s=170,
                    color="orange",
                    marker="*",
                    label="Target",
                    zorder=6,
                )
                plotted_goal = True

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Trajectories | ds={int(ds)} seed={int(seed)}")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(outpath, format="svg")
    plt.close()

    print(f"[SAVED] {outpath}")
    return outpath


def _extract_episode_idx_from_path(path: str) -> Optional[int]:
    m = re.search(r"_ep(\d+)\.npz$", os.path.basename(path))
    if m is None:
        return None
    return int(m.group(1))


def _find_traj_npz_by_episode(root: str, ds: int, seed: int) -> Dict[str, Dict[int, str]]:
    out: Dict[str, Dict[int, str]] = {}

    ppo_pattern = os.path.join(root, "**", f"seed{seed}", "ppo_only_ep*.npz")
    ppo_paths = sorted(glob.glob(ppo_pattern, recursive=True))
    if ppo_paths:
        out["ppo_only"] = {}
        for p in ppo_paths:
            ep = _extract_episode_idx_from_path(p)
            if ep is not None:
                out["ppo_only"][ep] = p

    for method in TRAJ_METHOD_ORDER:
        if method == "ppo_only":
            continue
        patt = os.path.join(root, "**", f"ds{ds}", f"seed{seed}", f"{method}_ep*.npz")
        paths = sorted(glob.glob(patt, recursive=True))
        if paths:
            out[method] = {}
            for p in paths:
                ep = _extract_episode_idx_from_path(p)
                if ep is not None:
                    out[method][ep] = p

    return out


def plot_trajectories_per_episode_for_ds_seed(
    root: str,
    ds: int,
    seed: int,
    outdir: str,
    *,
    max_episodes_per_method: int = 1,
):
    found = _find_traj_npz_by_episode(root=root, ds=int(ds), seed=int(seed))
    if len(found) == 0:
        print(f"[WARN] No trajectory npz found for ds={ds}, seed={seed}")
        return []

    all_eps = set()
    for _, ep_map in found.items():
        all_eps.update(ep_map.keys())

    all_eps = sorted(all_eps)
    if max_episodes_per_method is not None and int(max_episodes_per_method) > 0:
        all_eps = all_eps[: int(max_episodes_per_method)]

    os.makedirs(outdir, exist_ok=True)
    saved = []

    for ep in all_eps:
        plt.figure(figsize=(7.0, 7.0))
        ax = plt.gca()

        plotted_start = False
        plotted_goal = False
        something_plotted = False

        for method in TRAJ_METHOD_ORDER:
            if method not in found:
                continue
            if ep not in found[method]:
                continue

            traj = _load_traj_npz(found[method][ep])
            style = METHOD_STYLE.get(method, {})
            label = style.get("label", method)
            color = style.get("color", None)

            xy = np.asarray(traj["xy"], dtype=np.float32)
            start = np.asarray(traj["start"], dtype=np.float32)
            goal = np.asarray(traj["goal"], dtype=np.float32)

            ax.plot(
                xy[:, 0],
                xy[:, 1],
                linewidth=2.0,
                alpha=0.95,
                color=color,
                label=label,
            )

            ax.scatter(
                xy[-1, 0],
                xy[-1, 1],
                s=28,
                color=color,
                alpha=0.95,
            )

            if not plotted_start:
                ax.scatter(
                    start[0],
                    start[1],
                    s=90,
                    color="black",
                    marker="o",
                    label="Start",
                    zorder=5,
                )
                plotted_start = True

            if not plotted_goal:
                goal_radius = 0.45
                goal_region = plt.Circle(
                    (float(goal[0]), float(goal[1])),
                    goal_radius,
                    facecolor="orange",
                    edgecolor="orange",
                    alpha=0.12,
                    lw=1.0,
                    zorder=2,
                )
                ax.add_patch(goal_region)

                ax.scatter(
                    goal[0],
                    goal[1],
                    s=170,
                    color="orange",
                    marker="*",
                    label="Target",
                    zorder=6,
                )
                plotted_goal = True

            something_plotted = True

        if not something_plotted:
            plt.close()
            continue

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Trajectories | ds={int(ds)} seed={int(seed)} ep={int(ep)}")
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(frameon=False, fontsize=9)
        plt.tight_layout()

        outpath = os.path.join(outdir, f"plot_trajs_ds{int(ds)}_seed{int(seed)}_ep{int(ep)}.svg")
        plt.savefig(outpath, format="svg")
        plt.close()

        print(f"[SAVED] {outpath}")
        saved.append(outpath)

    return saved


# ============================================================================
# Results loading + normalization
# ============================================================================

def _find_results_csv(root: str) -> List[str]:
    pattern = os.path.join(root, "ds*", "seed*", "results.csv")
    return sorted(glob.glob(pattern))


def _safe_read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["__source_path"] = path
    return df


def _infer_ds_seed_from_path(path: str) -> Tuple[Optional[int], Optional[int]]:
    parts = path.replace("\\", "/").split("/")
    ds = None
    seed = None
    for p in parts:
        if p.startswith("ds"):
            try:
                ds = int(p.replace("ds", ""))
            except Exception:
                pass
        if p.startswith("seed"):
            try:
                seed = int(p.replace("seed", ""))
            except Exception:
                pass
    return ds, seed


def _looks_numeric_series(s: pd.Series) -> bool:
    if s.dtype.kind in ("i", "u", "f"):
        return True
    try:
        pd.to_numeric(s.dropna().iloc[:50], errors="raise")
        return True
    except Exception:
        return False


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "method" not in out.columns:
        for alt in ["algo", "name", "label"]:
            if alt in out.columns:
                out["method"] = out[alt].astype(str)
                break
    if "method" not in out.columns:
        raise ValueError("Missing 'method' column in results.csv")

    if "dataset_trajs" not in out.columns:
        for alt in ["ds", "dataset", "n_trajs", "num_trajs"]:
            if alt in out.columns:
                out["dataset_trajs"] = pd.to_numeric(out[alt], errors="coerce")
                break
    if "dataset_trajs" not in out.columns or out["dataset_trajs"].isna().all():
        ds, _ = _infer_ds_seed_from_path(str(out["__source_path"].iloc[0]))
        if ds is None:
            raise ValueError(
                "Cannot infer dataset_trajs; add it to results.csv or ensure path uses ds{N}."
            )
        out["dataset_trajs"] = int(ds)

    if "seed" not in out.columns:
        _, s = _infer_ds_seed_from_path(str(out["__source_path"].iloc[0]))
        if s is None:
            raise ValueError("Cannot infer seed; add it to results.csv or ensure path uses seed{N}.")
        out["seed"] = int(s)

    out["dataset_trajs"] = pd.to_numeric(out["dataset_trajs"], errors="coerce").astype(int)
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce").astype(int)
    out["method"] = out["method"].astype(str)

    id_cols = {"__source_path", "method", "dataset_trajs", "seed"}
    for c in out.columns:
        if c in id_cols:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


# ============================================================================
# Aggregation helpers
# ============================================================================

def _aggregate_absolute(df_all: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """
    Aggregate absolute metrics over seeds for each (dataset_trajs, method).
    If duplicates exist within a seed, average them first.
    """
    present_metrics = [m for m in metrics if m in df_all.columns]
    keep_cols = ["dataset_trajs", "seed", "method"] + present_metrics
    df = df_all[keep_cols].copy()

    df = df.dropna(subset=["dataset_trajs", "seed", "method"])

    if len(present_metrics) == 0:
        out = (
            df[["dataset_trajs", "method", "seed"]]
            .drop_duplicates()
            .groupby(["dataset_trajs", "method"], as_index=False)["seed"]
            .nunique()
            .rename(columns={"seed": "n_seeds"})
        )
        return out.sort_values(["dataset_trajs", "method"]).reset_index(drop=True)

    df_seed = (
        df.groupby(["dataset_trajs", "method", "seed"], as_index=False)[present_metrics]
          .mean()
    )

    agg_mean = (
        df_seed.groupby(["dataset_trajs", "method"], as_index=False)[present_metrics]
              .mean()
    )
    agg_std = (
        df_seed.groupby(["dataset_trajs", "method"], as_index=False)[present_metrics]
              .std(ddof=1)
    )
    n_seeds = (
        df_seed.groupby(["dataset_trajs", "method"], as_index=False)["seed"]
              .nunique()
              .rename(columns={"seed": "n_seeds"})
    )

    out = agg_mean.copy()
    for m in present_metrics:
        out[f"{m}__std"] = agg_std[m].values

    out = out.merge(n_seeds, on=["dataset_trajs", "method"], how="left")
    out = out.sort_values(["dataset_trajs", "method"]).reset_index(drop=True)
    return out


def _aggregate_delta_vs_baseline(
    df_all: pd.DataFrame,
    metrics: List[str],
    baseline_method: str = "ppo_only",
) -> pd.DataFrame:
    """
    Build paired deltas seed-by-seed:
        delta_metric = metric(method, ds, seed) - metric(baseline, ds, seed)

    Then aggregate deltas across seeds for each (dataset_trajs, method).
    """
    present_metrics = [m for m in metrics if m in df_all.columns]
    keep_cols = ["dataset_trajs", "seed", "method"] + present_metrics
    df = df_all[keep_cols].copy()

    df = df.dropna(subset=["dataset_trajs", "seed", "method"])

    if len(present_metrics) == 0:
        return pd.DataFrame(columns=["dataset_trajs", "method", "n_seeds"])

    df_seed = (
        df.groupby(["dataset_trajs", "method", "seed"], as_index=False)[present_metrics]
          .mean()
    )

    base = df_seed[df_seed["method"] == baseline_method].copy()
    if base.empty:
        raise ValueError(f"Baseline method '{baseline_method}' not found in results.")

    base = base[["dataset_trajs", "seed"] + present_metrics]
    base = base.rename(columns={m: f"{m}__baseline" for m in present_metrics})

    paired = df_seed.merge(base, on=["dataset_trajs", "seed"], how="inner")
    paired = paired[paired["method"] != baseline_method].copy()

    delta_metrics = []
    for m in present_metrics:
        dm = f"delta_{m}"
        paired[dm] = paired[m] - paired[f"{m}__baseline"]
        delta_metrics.append(dm)

    agg_mean = (
        paired.groupby(["dataset_trajs", "method"], as_index=False)[delta_metrics]
              .mean()
    )
    agg_std = (
        paired.groupby(["dataset_trajs", "method"], as_index=False)[delta_metrics]
              .std(ddof=1)
    )
    n_seeds = (
        paired.groupby(["dataset_trajs", "method"], as_index=False)["seed"]
              .nunique()
              .rename(columns={"seed": "n_seeds"})
    )

    out = agg_mean.copy()
    for m in delta_metrics:
        out[f"{m}__std"] = agg_std[m].values

    out = out.merge(n_seeds, on=["dataset_trajs", "method"], how="left")
    out = out.sort_values(["dataset_trajs", "method"]).reset_index(drop=True)
    return out


def _pick_metric(agg_cols, preferred: str, fallback: str) -> Optional[str]:
    if preferred in agg_cols:
        return preferred
    if fallback in agg_cols:
        return fallback
    return None


def _compute_yerr(sub: pd.DataFrame, metric: str, band_mode: str = "none") -> Optional[np.ndarray]:
    """
    band_mode:
      - none : no band
      - std  : standard deviation
      - sem  : standard error
      - ci95 : 95% confidence interval ~= 1.96 * sem
    """
    if band_mode == "none":
        return None

    std_col = f"{metric}__std"
    if std_col not in sub.columns:
        return None

    std = sub[std_col].values.astype(float)

    if band_mode == "std":
        return std

    if "n_seeds" not in sub.columns:
        return None

    n = np.maximum(sub["n_seeds"].values.astype(float), 1.0)
    sem = std / np.sqrt(n)

    if band_mode == "sem":
        return sem
    if band_mode == "ci95":
        return 1.96 * sem

    raise ValueError(f"Unknown band_mode: {band_mode}")


# ============================================================================
# Plot helpers
# ============================================================================

def _ordered_methods(agg: pd.DataFrame, include_baseline: bool = True) -> List[str]:
    present = set(agg["method"].unique().tolist())

    baseline = [BASELINE_METHOD] if include_baseline and BASELINE_METHOD in present else []

    styled_middle = [
        m for m in METHOD_STYLE.keys()
        if m in present and m != BASELINE_METHOD and m not in USER_METHODS_LAST
    ]

    other_middle = sorted([
        m for m in present
        if m not in METHOD_STYLE and m != BASELINE_METHOD and m not in USER_METHODS_LAST
    ])

    user_last = [m for m in USER_METHODS_LAST if m in present]

    return baseline + styled_middle + other_middle + user_last


def _add_y_headroom(ax, top_frac: float = 0.10, bottom_frac: float = 0.03) -> None:
    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin
    if not np.isfinite(ymin) or not np.isfinite(ymax) or yr <= 0:
        return
    ax.set_ylim(ymin - bottom_frac * yr, ymax + top_frac * yr)

def _plot_metric(
    agg: pd.DataFrame,
    metric: str,
    outdir: str,
    ylabel: str,
    filename_stub: str,
    *,
    band_mode: str = "none",
    zero_line: bool = False,
    title: Optional[str] = None,
    include_baseline: bool = True,
    xlabel: str = DEFAULT_XLABEL,
) -> None:
    methods = _ordered_methods(agg, include_baseline=include_baseline)
    ds_values = sorted(agg["dataset_trajs"].unique().tolist())

    plt.figure(figsize=(7.5, 4.5))

    for method in methods:
        sub = agg[agg["method"] == method].sort_values("dataset_trajs")
        if metric not in sub.columns:
            continue

        x = sub["dataset_trajs"].values.astype(float)
        y = sub[metric].values.astype(float)
        yerr = _compute_yerr(sub, metric, band_mode=band_mode)

        style = METHOD_STYLE.get(method, {})
        label = style.get("label", method)
        color = style.get("color", None)
        linestyle = style.get("linestyle", "-")
        marker = style.get("marker", "o")

        plt.plot(
            x,
            y,
            linewidth=1.8,
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
        )

        if yerr is not None:
            mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)
            if np.any(mask):
                plt.fill_between(
                    x[mask],
                    (y - yerr)[mask],
                    (y + yerr)[mask],
                    alpha=0.12,
                    color=color,
                )

    if zero_line:
        plt.axhline(0.0, linewidth=1.2, linestyle="--", color="black", alpha=0.7)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(ds_values, ds_values)
    plt.grid(True, alpha=0.25)

    ax = plt.gca()
    if metric in {"cvar15", "delta_cvar15"}:
        _add_y_headroom(ax, top_frac=0.16, bottom_frac=0.03)
    else:
        _add_y_headroom(ax, top_frac=0.08, bottom_frac=0.03)


    fig = plt.gcf()
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()

    if len(handles) > 0:
        ncol = _legend_ncol(len(labels), max_cols=4)
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=ncol,
            frameon=False,
            fontsize=8,
            columnspacing=1.2,
            handletextpad=0.6,
        )
        top_rect = 0.90
    else:
        top_rect = 0.95

    plt.tight_layout(rect=[0, 0, 1, top_rect])

    svg_path = os.path.join(outdir, f"{filename_stub}.svg")
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {svg_path}")


def _plot_panel(
    ax,
    agg: pd.DataFrame,
    metric: Optional[str],
    ylabel: str,
    title: str,
    *,
    band_mode: str,
    include_baseline: bool,
    zero_line: bool = False,
) -> None:
    if metric is None or metric not in agg.columns:
        ax.axis("off")
        ax.text(0.5, 0.5, f"Missing metric:\n{metric}", ha="center", va="center")
        return

    methods = _ordered_methods(agg, include_baseline=include_baseline)

    for method in methods:
        sub = agg[agg["method"] == method].sort_values("dataset_trajs")
        if metric not in sub.columns:
            continue

        x = sub["dataset_trajs"].values.astype(float)
        y = sub[metric].values.astype(float)
        yerr = _compute_yerr(sub, metric, band_mode=band_mode)

        style = METHOD_STYLE.get(method, {})
        label = style.get("label", method)
        color = style.get("color", None)
        linestyle = style.get("linestyle", "-")
        marker = style.get("marker", "o")

        ax.plot(
            x, y,
            linewidth=1.8,
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
        )

        if yerr is not None:
            mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)
            if np.any(mask):
                ax.fill_between(
                    x[mask],
                    (y - yerr)[mask],
                    (y + yerr)[mask],
                    alpha=0.12,
                    color=color,
                )

    if zero_line:
        ax.axhline(0.0, linewidth=1.2, linestyle="--", color="black", alpha=0.7)

    ds_values = sorted(agg["dataset_trajs"].unique().tolist())
    ax.set_xticks(ds_values)
    ax.set_xticklabels([str(int(v)) for v in ds_values])
    if metric in {"cvar15", "delta_cvar15"}:
        _add_y_headroom(ax, top_frac=0.16, bottom_frac=0.03)
    else:
        _add_y_headroom(ax, top_frac=0.08, bottom_frac=0.03)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)


def plot_main_figure_4panel(
    agg_abs: pd.DataFrame,
    agg_delta: pd.DataFrame,
    outdir: str,
    *,
    tail_mode: str = "cvar",
    xlabel: Optional[str] = None,
    filename: str = "figure_main_4panel.svg",
    abs_band_mode: str = "none",
    delta_band_mode: str = "ci95",
) -> None:
    """
    4-panel figure:
      A) Success@H (absolute)
      B) Median steps-to-success (absolute, if available)
      C) Mean improvement vs baseline (delta)
      D) Tail safety (delta CVaR or delta P(Δ<0), if available)
    """
    if xlabel is None:
        xlabel = DEFAULT_XLABEL

    os.makedirs(outdir, exist_ok=True)

    cols_abs = set(agg_abs.columns)
    cols_delta = set(agg_delta.columns)

    m_success = _pick_metric(cols_abs, "success_rate", "success_rate")
    m_tts_med = _pick_metric(cols_abs, "time_to_success_median", "time_to_success_median")
    m_dmean = _pick_metric(cols_delta, "delta_budget_return_mean", "delta_return_mean")

    if tail_mode.lower() == "cvar":
        m_dtail = _pick_metric(cols_delta, "delta_budget_return_cvar15", "delta_cvar15")
        tail_ylabel = r"CVaR$_{0.15}(\Delta)$"
    elif tail_mode.lower() == "pneg":
        m_dtail = _pick_metric(cols_delta, "delta_budget_return_p_neg", "delta_return_p_neg")
        tail_ylabel = r"$\mathbb{P}(\Delta < 0)$"
    else:
        raise ValueError("tail_mode must be 'cvar' or 'pneg'")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True)
    axA, axB = axes[0, 0], axes[0, 1]
    axC, axD = axes[1, 0], axes[1, 1]

    _plot_panel(
        axA,
        agg_abs,
        metric=m_success,
        ylabel="Success rate",
        title="A — Success@H",
        band_mode=abs_band_mode,
        include_baseline=True,
        zero_line=False,
    )

    _plot_panel(
        axB,
        agg_abs,
        metric=m_tts_med,
        ylabel="Median steps-to-success",
        title="B — Median steps-to-success",
        band_mode=abs_band_mode,
        include_baseline=True,
        zero_line=False,
    )

    _plot_panel(
        axC,
        agg_delta,
        metric=m_dmean,
        ylabel=r"$\Delta$ mean return",
        title="C — Mean improvement vs baseline",
        band_mode=delta_band_mode,
        include_baseline=False,
        zero_line=True,
    )

    _plot_panel(
        axD,
        agg_delta,
        metric=m_dtail,
        ylabel=tail_ylabel,
        title="D — Tail safety",
        band_mode=delta_band_mode,
        include_baseline=False,
        zero_line=True,
    )

    axC.set_xlabel(xlabel)
    axD.set_xlabel(xlabel)

    handles, labels = axA.get_legend_handles_labels()
    if len(handles) > 0:
        ncol = _legend_ncol(len(labels), max_cols=4)
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.015),
            ncol=ncol,
            frameon=False,
            fontsize=9,
            columnspacing=1.3,
            handletextpad=0.6,
        )
        top_rect = 0.89
    else:
        top_rect = 0.95

    fig.tight_layout(rect=[0, 0, 1, top_rect])
    out_path = os.path.join(outdir, filename)
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {out_path}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Root folder containing ds*/seed*/results.csv")
    ap.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Where to write aggregated csv + plots (default: root)",
    )
    ap.add_argument(
        "--metrics",
        type=str,
        default=",".join(METRICS_DEFAULT),
        help="Comma-separated metrics to aggregate. Use ALL to aggregate all numeric columns.",
    )
    ap.add_argument(
        "--baseline_method",
        type=str,
        default="ppo_only",
        help="Method used as baseline for paired delta plots.",
    )
    ap.add_argument(
        "--abs_band_mode",
        type=str,
        default="none",
        choices=BAND_MODE_CHOICES,
        help="Band mode for absolute plots: none/std/sem/ci95",
    )
    ap.add_argument(
        "--delta_band_mode",
        type=str,
        default="ci95",
        choices=BAND_MODE_CHOICES,
        help="Band mode for delta-vs-baseline plots: none/std/sem/ci95",
    )
    ap.add_argument(
        "--make_main_figure",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, generate the main 4-panel figure.",
    )
    ap.add_argument(
        "--tail_mode",
        type=str,
        default="cvar",
        choices=["cvar", "pneg"],
        help="Panel D metric for the 4-panel figure.",
    )
    ap.add_argument(
        "--plot_trajs",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, also plot saved trajectory npz files.",
    )
    ap.add_argument(
        "--traj_ds",
        type=int,
        nargs="*",
        default=[],
        help="Dataset sizes for which to plot trajectories, e.g. 10 100 1000",
    )
    ap.add_argument(
        "--traj_seeds",
        type=int,
        nargs="*",
        default=[0],
        help="Seeds for which to plot trajectories.",
    )
    ap.add_argument(
        "--traj_max_episodes_per_method",
        type=int,
        default=1,
        help="Number of episode indices to plot per (ds, seed).",
    )
    ap.add_argument(
        "--traj_outdir",
        type=str,
        default=None,
        help="Where to save trajectory plots (default: <outdir>/traj_plots)",
    )

    args = ap.parse_args()

    root = os.path.abspath(args.root)
    outdir = os.path.abspath(args.outdir) if args.outdir else root
    os.makedirs(outdir, exist_ok=True)

    paths = _find_results_csv(root)
    if len(paths) == 0:
        raise SystemExit(f"No results.csv found under: {root} (expected {root}/ds*/seed*/results.csv)")

    dfs: List[pd.DataFrame] = []
    for p in paths:
        try:
            df = _safe_read_csv(p)
            df = _normalize_df(df)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] skipping {p}: {e}")

    if len(dfs) == 0:
        raise SystemExit("No valid results.csv loaded.")

    df_all = pd.concat(dfs, ignore_index=True)

    metrics_arg = args.metrics.strip()
    if metrics_arg.upper() == "ALL":
        id_cols = {"dataset_trajs", "seed", "method", "__source_path"}
        metrics = []
        for c in df_all.columns:
            if c in id_cols:
                continue
            if _looks_numeric_series(df_all[c]):
                metrics.append(c)
        metrics = sorted(metrics)
    else:
        metrics = [m.strip() for m in metrics_arg.split(",") if m.strip()]

    # Absolute aggregation over methods/seeds
    agg_abs = _aggregate_absolute(df_all, metrics=metrics)

    # Delta aggregation only for core metrics usually used in comparisons
    delta_metrics_candidates = ["return_mean", "cvar15", "success_rate"]
    delta_metrics = [m for m in delta_metrics_candidates if m in df_all.columns]

    agg_delta = _aggregate_delta_vs_baseline(
        df_all,
        metrics=delta_metrics,
        baseline_method=str(args.baseline_method),
    )

    # Save CSVs
    agg_csv_legacy = os.path.join(outdir, "aggregated_results.csv")
    agg_abs_csv = os.path.join(outdir, "aggregated_results_absolute.csv")
    agg_delta_csv = os.path.join(
        outdir,
        f"aggregated_results_delta_vs_{str(args.baseline_method)}.csv",
    )

    agg_abs.to_csv(agg_csv_legacy, index=False)
    agg_abs.to_csv(agg_abs_csv, index=False)
    agg_delta.to_csv(agg_delta_csv, index=False)

    print(f"[SAVED] {agg_csv_legacy}")
    print(f"[SAVED] {agg_abs_csv}")
    print(f"[SAVED] {agg_delta_csv}")

    # Main 4-panel figure
    if int(args.make_main_figure) == 1:
        plot_main_figure_4panel(
            agg_abs=agg_abs,
            agg_delta=agg_delta,
            outdir=outdir,
            tail_mode=str(args.tail_mode),
            xlabel=DEFAULT_XLABEL,
            filename=f"figure_main_4panel_{args.tail_mode}.svg",
            abs_band_mode=str(args.abs_band_mode),
            delta_band_mode=str(args.delta_band_mode),
        )

    # ---------------------------------------------------------------------
    # Absolute plots (backward-compatible names)
    # ---------------------------------------------------------------------
    if "return_mean" in agg_abs.columns:
        _plot_metric(
            agg_abs,
            metric="return_mean",
            outdir=outdir,
            ylabel=r"$\bar{\rho}(\pi, M^*)$",
            filename_stub="plot_return_mean",
            band_mode=str(args.abs_band_mode),
            zero_line=False,
            title="Mean return",
            include_baseline=True,
            xlabel=DEFAULT_XLABEL,
        )

    if "cvar15" in agg_abs.columns:
        _plot_metric(
            agg_abs,
            metric="cvar15",
            outdir=outdir,
            ylabel=r"CVaR$_{0.15}\,\bar{\rho}(\pi, M^*)$",
            filename_stub="plot_cvar15",
            band_mode=str(args.abs_band_mode),
            zero_line=False,
            title="CVaR15",
            include_baseline=True,
            xlabel=DEFAULT_XLABEL,
        )

    if "success_rate" in agg_abs.columns:
        _plot_metric(
            agg_abs,
            metric="success_rate",
            outdir=outdir,
            ylabel="Success rate",
            filename_stub="plot_success_rate",
            band_mode=str(args.abs_band_mode),
            zero_line=False,
            title="Success rate",
            include_baseline=True,
            xlabel=DEFAULT_XLABEL,
        )

    # ---------------------------------------------------------------------
    # Delta-vs-baseline plots (paired seed-by-seed)
    # ---------------------------------------------------------------------
    if "delta_return_mean" in agg_delta.columns:
        _plot_metric(
            agg_delta,
            metric="delta_return_mean",
            outdir=outdir,
            ylabel=rf"$\Delta$ mean return vs {args.baseline_method}",
            filename_stub=f"plot_delta_return_mean_vs_{args.baseline_method}",
            band_mode=str(args.delta_band_mode),
            zero_line=True,
            title=f"Mean return improvement vs {args.baseline_method}",
            include_baseline=False,
            xlabel=DEFAULT_XLABEL,
        )

    if "delta_cvar15" in agg_delta.columns:
        _plot_metric(
            agg_delta,
            metric="delta_cvar15",
            outdir=outdir,
            ylabel=rf"$\Delta$ CVaR$_{{0.15}}$ vs {args.baseline_method}",
            filename_stub=f"plot_delta_cvar15_vs_{args.baseline_method}",
            band_mode=str(args.delta_band_mode),
            zero_line=True,
            title=f"CVaR15 improvement vs {args.baseline_method}",
            include_baseline=False,
            xlabel=DEFAULT_XLABEL,
        )

    if "delta_success_rate" in agg_delta.columns:
        _plot_metric(
            agg_delta,
            metric="delta_success_rate",
            outdir=outdir,
            ylabel=rf"$\Delta$ success rate vs {args.baseline_method}",
            filename_stub=f"plot_delta_success_rate_vs_{args.baseline_method}",
            band_mode=str(args.delta_band_mode),
            zero_line=True,
            title=f"Success-rate improvement vs {args.baseline_method}",
            include_baseline=False,
            xlabel=DEFAULT_XLABEL,
        )

    # ---- optional trajectory plots ----
    if int(args.plot_trajs) == 1:
        traj_outdir = (
            os.path.abspath(args.traj_outdir)
            if args.traj_outdir
            else os.path.join(outdir, "traj_plots")
        )

        ds_for_trajs = (
            [int(x) for x in args.traj_ds]
            if len(args.traj_ds) > 0
            else sorted(df_all["dataset_trajs"].unique().tolist())
        )
        seeds_for_trajs = [int(x) for x in args.traj_seeds]

        for ds in ds_for_trajs:
            for seed in seeds_for_trajs:
                plot_trajectories_per_episode_for_ds_seed(
                    root=root,
                    ds=int(ds),
                    seed=int(seed),
                    outdir=traj_outdir,
                    max_episodes_per_method=int(args.traj_max_episodes_per_method),
                )

    print("\n[SUMMARY absolute]")
    if "n_seeds" in agg_abs.columns:
        print(agg_abs.groupby("method")["n_seeds"].max().sort_values(ascending=False).to_string())

    print("\n[SUMMARY delta]")
    if len(agg_delta) > 0 and "n_seeds" in agg_delta.columns:
        print(agg_delta.groupby("method")["n_seeds"].max().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()