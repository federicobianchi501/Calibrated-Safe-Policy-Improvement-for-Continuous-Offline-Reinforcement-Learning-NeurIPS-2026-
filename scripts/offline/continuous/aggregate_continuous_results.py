#!/usr/bin/env python3
# =============================================================================
# Anonymous authors. Affiliations and acknowledgements will be added on
# acceptance of the associated paper. Submitted to NeurIPS 2026 (double-blind).
# =============================================================================
"""
aggregate_continuous_results.py — Aggregate Cal-SPI benchmark results for Walker2d / Hopper.

Expected layout (produced by benchmark_continuous.py via run_benchmark_*.sh):
    <root>/ds{DS}/seed{SEED}/results.csv

The script:
  1. Loads all results.csv under root
  2. Aggregates mean ± STD over seeds for each (ds, method)
  3. Saves:
     - aggregated_results.csv       (absolute metrics, mean+std over seeds)
     - aggregated_delta_vs_ppo.csv  (delta vs ppo_only baseline, seed-paired)
     - plot_return_mean.svg
     - plot_cvar15.svg
     - plot_success_rate.svg
     - plot_xvel_mean.svg

Usage:
    python aggregate_continuous_results.py --root results_continuous_walker2d
    python aggregate_continuous_results.py --root results_continuous_hopper \\
        --outdir results_continuous_hopper/aggregated
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MATPLOTLIB = True
except ImportError:
    _MATPLOTLIB = False

BASELINE_METHOD = "ppo_only"
BAND_MODE_CHOICES = ["none", "std", "sem", "ci95"]
DEFAULT_XLABEL = r"Number of trajectories in dataset $\mathcal{D}$"

# Methods always plotted last (Cal-SPI variants, most prominent in legend)
USER_METHODS_LAST = [
    "morel_usad",
    "calibrated_variance",
    "calibrated_usad",
]


def _legend_ncol(n_items: int, max_cols: int = 4) -> int:
    return min(max_cols, max(1, n_items))


# ── Metrics to aggregate ──────────────────────────────────────────────────────
METRICS = [
    "return_mean",
    "return_std",
    "return_median",
    "return_min",
    "cvar_15",
    "discounted_return_mean",
    "ep_len_mean",
    "success_rate",
    "xvel_mean",
    "root_fallback_rate",
    "root_coverage_mean",
]

# ── Plot config ───────────────────────────────────────────────────────────────
# Colors/markers are kept in sync with aggregate_pointmaze_results.py
METHOD_STYLE: Dict[str, Dict] = {
    "ppo_only": {
        "label": "Baseline",
        "color": "#E69F00",
        "linestyle": "-",
        "marker": None,
        "zorder": 3,
    },
    "mcts_only_no_gate": {
        "label": "MCTS-DPW",
        "color": "#0072B2",
        "linestyle": "--",
        "marker": "v",
        "zorder": 2,
    },
    "morel_usad": {
        "label": "SPI-U",
        "color": "#6A3D9A",
        "linestyle": "-",
        "marker": "s",
        "zorder": 2,
    },
    "calibrated_variance": {
        "label": "CSPI-V",
        "color": "#D62728",
        "linestyle": "-",
        "marker": "^",
        "zorder": 4,
    },
    "calibrated_usad": {
        "label": "CSPI-U",
        "color": "#1B9E77",
        "linestyle": "-",
        "marker": "o",
        "zorder": 5,
    },
    "mopo_mse": {
        "label": "MOPO",
        "color": "#56B4E9",
        "linestyle": ":",
        "marker": "D",
        "zorder": 2,
    },
    "morel_full": {
        "label": "MOReL",
        "color": "#CC79A7",
        "linestyle": "-.",
        "marker": "P",
        "zorder": 2,
    },
    "iql": {
        "label": "IQL",
        "color": "#7F7F7F",
        "linestyle": "--",
        "marker": "X",
        "zorder": 2,
    },
    "td3bc": {
        "label": "TD3+BC",
        "color": "#8C564B",
        "linestyle": ":",
        "marker": "*",
        "zorder": 2,
    },
    "bear": {
        "label": "BEAR",
        "color": "#F4A582",
        "linestyle": "--",
        "marker": "h",
        "zorder": 2,
    },
    "cql": {
        "label": "CQL",
        "color": "#92C5DE",
        "linestyle": "-.",
        "marker": "H",
        "zorder": 2,
    },
    "cal_ql": {
        "label": "Cal-QL",
        "color": "#4DAC26",
        "linestyle": ":",
        "marker": "8",
        "zorder": 2,
    },
}

METHOD_ORDER = [
    "ppo_only",
    "mcts_only_no_gate",
    "morel_usad",
    "calibrated_variance",
    "calibrated_usad",
    "mopo_mse",
    "morel_full",
    "iql",
    "td3bc",
    "bear",
    "cql",
    "cal_ql",
]


# ── CSV loading ───────────────────────────────────────────────────────────────

def _infer_ds_seed(path: str) -> Tuple[Optional[int], Optional[int]]:
    """Infer ds and seed from path like '.../ds50/seed3/results.csv'."""
    parts = path.replace("\\", "/").split("/")
    ds, seed = None, None
    for p in parts:
        if re.fullmatch(r"ds\d+", p):
            ds = int(p[2:])
        if re.fullmatch(r"seed\d+", p):
            seed = int(p[4:])
    return ds, seed


def _load_one(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ds, seed = _infer_ds_seed(path)

    # Normalise column names
    if "dataset_trajs" not in df.columns:
        for alt in ("ds", "dataset", "n_trajs"):
            if alt in df.columns:
                df["dataset_trajs"] = pd.to_numeric(df[alt], errors="coerce").astype(int)
                break
    if "dataset_trajs" not in df.columns:
        if ds is not None:
            df["dataset_trajs"] = ds
        else:
            raise ValueError(f"Cannot infer dataset_trajs from {path}")

    if "seed" not in df.columns:
        if seed is not None:
            df["seed"] = seed
        else:
            raise ValueError(f"Cannot infer seed from {path}")

    # cvar_15 alias
    if "cvar_15" not in df.columns and "cvar15" in df.columns:
        df["cvar_15"] = df["cvar15"]

    df["dataset_trajs"] = pd.to_numeric(df["dataset_trajs"], errors="coerce").astype(int)
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype(int)
    df["__source"] = path
    return df


def load_all(root: str) -> pd.DataFrame:
    pattern = os.path.join(root, "**", "results.csv")
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        raise FileNotFoundError(
            f"No results.csv found under {root}. "
            "Run benchmark first: bash run_benchmark_walker2d.sh --quick"
        )
    print(f"[AGG] Found {len(paths)} results.csv files under {root}")
    dfs = [_load_one(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    for col in METRICS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_absolute(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Mean ± STD over seeds for each (dataset_trajs, method).
    If multiple rows per (ds, seed, method) exist, average them first.
    """
    present = [m for m in METRICS if m in df_all.columns]
    df = df_all[["dataset_trajs", "seed", "method"] + present].copy()
    df = df.dropna(subset=["dataset_trajs", "seed", "method"])

    # Per-seed average (handles duplicates)
    df_s = df.groupby(["dataset_trajs", "method", "seed"], as_index=False)[present].mean()

    mean = df_s.groupby(["dataset_trajs", "method"], as_index=False)[present].mean()
    std  = df_s.groupby(["dataset_trajs", "method"], as_index=False)[present].std(ddof=1)
    n    = (df_s.groupby(["dataset_trajs", "method"], as_index=False)["seed"]
                .nunique().rename(columns={"seed": "n_seeds"}))

    out = mean.copy()
    for m in present:
        out[f"{m}__std"] = std[m].values
    out = out.merge(n, on=["dataset_trajs", "method"], how="left")
    return out.sort_values(["dataset_trajs", "method"]).reset_index(drop=True)


def aggregate_delta(
    df_all: pd.DataFrame,
    baseline: str = "ppo_only",
) -> pd.DataFrame:
    """
    Seed-paired delta vs baseline: Δmetric = metric(method) - metric(baseline).
    """
    present = [m for m in METRICS if m in df_all.columns]
    df = df_all[["dataset_trajs", "seed", "method"] + present].copy()
    df = df.dropna(subset=["dataset_trajs", "seed", "method"])

    df_s = df.groupby(["dataset_trajs", "method", "seed"], as_index=False)[present].mean()

    base = df_s[df_s["method"] == baseline][["dataset_trajs", "seed"] + present].copy()
    if base.empty:
        print(f"[WARN] Baseline '{baseline}' not found — skipping delta aggregation.")
        return pd.DataFrame()
    base = base.rename(columns={m: f"{m}__base" for m in present})

    paired = df_s[df_s["method"] != baseline].merge(
        base, on=["dataset_trajs", "seed"], how="inner"
    )
    delta_cols = []
    for m in present:
        d = f"delta_{m}"
        paired[d] = paired[m] - paired[f"{m}__base"]
        delta_cols.append(d)

    mean = paired.groupby(["dataset_trajs", "method"], as_index=False)[delta_cols].mean()
    std  = paired.groupby(["dataset_trajs", "method"], as_index=False)[delta_cols].std(ddof=1)
    n    = (paired.groupby(["dataset_trajs", "method"], as_index=False)["seed"]
                  .nunique().rename(columns={"seed": "n_seeds"}))

    out = mean.copy()
    for d in delta_cols:
        out[f"{d}__std"] = std[d].values
    out = out.merge(n, on=["dataset_trajs", "method"], how="left")
    return out.sort_values(["dataset_trajs", "method"]).reset_index(drop=True)


# ── Console table ─────────────────────────────────────────────────────────────

def print_table(agg: pd.DataFrame, metrics: List[str], title: str = "") -> None:
    if title:
        print(f"\n{'─'*60}")
        print(f"  {title}")
        print(f"{'─'*60}")

    cols = ["dataset_trajs", "method", "n_seeds"]
    for m in metrics:
        if m in agg.columns:
            cols.append(m)
        if f"{m}__std" in agg.columns:
            cols.append(f"{m}__std")

    cols = [c for c in cols if c in agg.columns]
    df_show = agg[cols].copy()
    float_cols = [c for c in df_show.columns if df_show[c].dtype.kind == "f"]
    df_show[float_cols] = df_show[float_cols].round(4)
    print(df_show.to_string(index=False))


# ── Plotting helpers (mirrors aggregate_pointmaze_results.py style) ───────────

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


def _compute_yerr(
    sub: pd.DataFrame, metric: str, band_mode: str = "none"
) -> Optional[np.ndarray]:
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
    raise ValueError(f"Unknown band_mode: {band_mode!r}")


def _add_y_headroom(ax, top_frac: float = 0.10, bottom_frac: float = 0.03) -> None:
    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin
    if not np.isfinite(ymin) or not np.isfinite(ymax) or yr <= 0:
        return
    ax.set_ylim(ymin - bottom_frac * yr, ymax + top_frac * yr)


def _auto_zoom_ymin(agg: pd.DataFrame, metric: str, baseline: str = BASELINE_METHOD,
                    competitive_frac: float = 0.70) -> Optional[float]:
    """
    Compute ymin as the minimum mean of "competitive" methods (those that
    achieve >= competitive_frac * baseline mean across any ds value), minus
    a small margin. This excludes collapsed methods from setting the floor.
    """
    base_rows = agg[agg["method"] == baseline][metric].dropna()
    if base_rows.empty:
        return None
    base_mean = float(base_rows.mean())
    threshold = competitive_frac * base_mean
    # Per-method global mean
    method_means = agg.groupby("method")[metric].mean()
    competitive = method_means[method_means >= threshold]
    if competitive.empty:
        return max(0.0, base_mean * 0.5)
    ymin_raw = float(competitive.min())
    margin = max(5.0, 0.03 * base_mean)
    return max(0.0, ymin_raw - margin)


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
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
) -> None:
    if not _MATPLOTLIB:
        print(f"[WARN] matplotlib not available — skipping {filename_stub}")
        return
    if metric not in agg.columns:
        print(f"[WARN] Metric '{metric}' not in aggregated data — skipping plot.")
        return

    methods = _ordered_methods(agg, include_baseline=include_baseline)
    ds_values = sorted(agg["dataset_trajs"].unique().tolist())

    plt.figure(figsize=(7.5, 4.5))

    for method in methods:
        sub = agg[agg["method"] == method].sort_values("dataset_trajs")
        if metric not in sub.columns or sub.empty:
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
            x, y,
            linewidth=1.8,
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            zorder=style.get("zorder", 2),
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
    if metric in {"cvar_15", "delta_cvar_15"}:
        _add_y_headroom(ax, top_frac=0.16, bottom_frac=0.03)
    else:
        _add_y_headroom(ax, top_frac=0.08, bottom_frac=0.03)
    cur_ymin, cur_ymax = ax.get_ylim()
    if ymin is not None:
        ax.set_ylim(ymin, cur_ymax)
    if ymax is not None:
        ax.set_ylim(ax.get_ylim()[0], ymax)

    fig = plt.gcf()
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ncol = _legend_ncol(len(labels), max_cols=4)
        fig.legend(
            handles, labels,
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
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
) -> None:
    if metric is None or metric not in agg.columns:
        ax.axis("off")
        ax.text(0.5, 0.5, f"Missing metric:\n{metric}", ha="center", va="center",
                transform=ax.transAxes)
        return

    methods = _ordered_methods(agg, include_baseline=include_baseline)

    for method in methods:
        sub = agg[agg["method"] == method].sort_values("dataset_trajs")
        if metric not in sub.columns or sub.empty:
            continue

        x = sub["dataset_trajs"].values.astype(float)
        y = sub[metric].values.astype(float)
        yerr = _compute_yerr(sub, metric, band_mode=band_mode)

        style = METHOD_STYLE.get(method, {})
        ax.plot(
            x, y,
            linewidth=1.8,
            label=style.get("label", method),
            color=style.get("color", None),
            linestyle=style.get("linestyle", "-"),
            marker=style.get("marker", "o"),
            zorder=style.get("zorder", 2),
        )

        if yerr is not None:
            mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)
            if np.any(mask):
                ax.fill_between(
                    x[mask],
                    (y - yerr)[mask],
                    (y + yerr)[mask],
                    alpha=0.12,
                    color=style.get("color", None),
                )

    if zero_line:
        ax.axhline(0.0, linewidth=1.2, linestyle="--", color="black", alpha=0.7)

    ds_values = sorted(agg["dataset_trajs"].unique().tolist())
    ax.set_xticks(ds_values)
    ax.set_xticklabels([str(int(v)) for v in ds_values])
    if metric in {"cvar_15", "delta_cvar_15"}:
        _add_y_headroom(ax, top_frac=0.16, bottom_frac=0.03)
    else:
        _add_y_headroom(ax, top_frac=0.08, bottom_frac=0.03)
    cur_ymin, cur_ymax = ax.get_ylim()
    if ymin is not None:
        ax.set_ylim(ymin, cur_ymax)
    if ymax is not None:
        ax.set_ylim(ax.get_ylim()[0], ymax)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)


def plot_main_figure_4panel(
    agg_abs: pd.DataFrame,
    agg_delta: pd.DataFrame,
    outdir: str,
    *,
    xlabel: str = DEFAULT_XLABEL,
    filename: str = "figure_main_4panel.svg",
    abs_band_mode: str = "none",
    delta_band_mode: str = "ci95",
    ymin_return: Optional[float] = None,
) -> None:
    """
    4-panel figure (mirrors pointmaze style):
      A) Mean return (absolute)
      B) Mean x-velocity (absolute, domain-specific)
      C) Δ mean return vs baseline (delta, seed-paired)
      D) Δ CVaR₁₅ vs baseline (delta, seed-paired)
    """
    os.makedirs(outdir, exist_ok=True)

    cols_abs = set(agg_abs.columns)
    cols_delta = set(agg_delta.columns)

    m_return  = "return_mean"  if "return_mean"  in cols_abs   else None
    m_xvel    = "xvel_mean"    if "xvel_mean"    in cols_abs   else None
    m_dreturn = "delta_return_mean" if "delta_return_mean" in cols_delta else None
    m_dcvar   = "delta_cvar_15"    if "delta_cvar_15"    in cols_delta else None

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=False)
    axA, axB = axes[0, 0], axes[0, 1]
    axC, axD = axes[1, 0], axes[1, 1]

    _plot_panel(axA, agg_abs,   metric=m_return,
                ylabel=r"Mean return $\bar\rho$",
                title="A — Return (absolute)",
                band_mode=abs_band_mode, include_baseline=True,
                ymin=ymin_return)

    _plot_panel(axB, agg_abs,   metric=m_xvel,
                ylabel="Mean x-velocity",
                title="B — x-velocity (absolute)",
                band_mode=abs_band_mode, include_baseline=True)

    _plot_panel(axC, agg_delta, metric=m_dreturn,
                ylabel=r"$\Delta$ mean return",
                title="C — Return improvement vs baseline",
                band_mode=delta_band_mode, include_baseline=False, zero_line=True)

    _plot_panel(axD, agg_delta, metric=m_dcvar,
                ylabel=r"$\Delta$ CVaR$_{0.15}$",
                title="D — Tail-risk improvement vs baseline",
                band_mode=delta_band_mode, include_baseline=False, zero_line=True)

    for ax in (axC, axD):
        ax.set_xlabel(xlabel)

    handles, labels = axA.get_legend_handles_labels()
    if handles:
        ncol = _legend_ncol(len(labels), max_cols=4)
        fig.legend(
            handles, labels,
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate Cal-SPI continuous benchmark results")
    ap.add_argument("--root", type=str, required=True,
                    help="Root results directory (e.g. results_continuous_walker2d)")
    ap.add_argument("--outdir", type=str, default=None,
                    help="Output directory (default: <root>/aggregated)")
    ap.add_argument("--baseline_method", type=str, default="ppo_only",
                    help="Baseline method for delta plots")
    ap.add_argument("--abs_band_mode", type=str, default="none",
                    choices=BAND_MODE_CHOICES,
                    help="Band mode for absolute plots: none/std/sem/ci95")
    ap.add_argument("--delta_band_mode", type=str, default="none",
                    choices=BAND_MODE_CHOICES,
                    help="Band mode for delta plots: none/std/sem/ci95")
    ap.add_argument("--make_main_figure", type=int, default=1, choices=[0, 1],
                    help="If 1, generate the main 4-panel figure")
    ap.add_argument("--no_plots", action="store_true",
                    help="Skip all plot generation")
    ap.add_argument("--return_ymin", type=float, default=None,
                    help="Fix y-axis lower bound for absolute return plots "
                         "(e.g. 400). Default: auto-zoom at 50%% of baseline mean.")
    args = ap.parse_args()

    outdir = args.outdir or os.path.join(args.root, "aggregated")
    os.makedirs(outdir, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    df_all = load_all(args.root)
    print(f"[AGG] Loaded {len(df_all)} rows, methods: {sorted(df_all['method'].unique())}")
    print(f"[AGG] Dataset sizes: {sorted(df_all['dataset_trajs'].unique())}")
    print(f"[AGG] Seeds: {sorted(df_all['seed'].unique())}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    agg_abs = aggregate_absolute(df_all)
    agg_delta = aggregate_delta(df_all, baseline=args.baseline_method)

    # Add percentage-delta columns: delta_<metric>_pct = delta / baseline_mean * 100
    if not agg_delta.empty:
        base_means = (agg_abs[agg_abs["method"] == args.baseline_method]
                      .groupby("dataset_trajs")[list(
                          m for m in METRICS if m in agg_abs.columns
                      )].mean())
        for m in METRICS:
            dcol = f"delta_{m}"
            pcol = f"delta_{m}_pct"
            if dcol in agg_delta.columns and m in base_means.columns:
                pct = agg_delta[["dataset_trajs", dcol]].copy()
                pct = pct.join(base_means[[m]].rename(columns={m: f"{m}__base"}),
                               on="dataset_trajs")
                agg_delta[pcol] = pct[dcol] / pct[f"{m}__base"].abs() * 100.0

    agg_csv = os.path.join(outdir, "aggregated_results.csv")
    agg_abs_csv = os.path.join(outdir, "aggregated_results_absolute.csv")
    agg_abs.to_csv(agg_csv, index=False)
    agg_abs.to_csv(agg_abs_csv, index=False)
    print(f"[AGG] Saved → {agg_csv}")
    print(f"[AGG] Saved → {agg_abs_csv}")

    if not agg_delta.empty:
        delta_csv = os.path.join(outdir, f"aggregated_delta_vs_{args.baseline_method}.csv")
        agg_delta.to_csv(delta_csv, index=False)
        print(f"[AGG] Saved → {delta_csv}")

    # ── Console summary ───────────────────────────────────────────────────────
    print_table(agg_abs, ["return_mean", "success_rate", "xvel_mean"],
                title="Aggregated results (absolute)")
    if not agg_delta.empty:
        print_table(agg_delta, ["delta_return_mean", "delta_success_rate"],
                    title=f"Delta vs {args.baseline_method} (seed-paired)")

    if args.no_plots:
        print(f"\n[AGG] Done (no plots). All CSVs in: {outdir}")
        return

    # ── Compute zoom ymin (auto or explicit) ─────────────────────────────────
    zoom_ymin = args.return_ymin
    if zoom_ymin is None:
        zoom_ymin = _auto_zoom_ymin(agg_abs, "return_mean",
                                    baseline=args.baseline_method)
    if zoom_ymin is not None:
        print(f"[AGG] Auto zoom ymin for return plots: {zoom_ymin:.1f}")

    # ── 4-panel main figure ───────────────────────────────────────────────────
    if int(args.make_main_figure) == 1:
        plot_main_figure_4panel(
            agg_abs=agg_abs,
            agg_delta=agg_delta,
            outdir=outdir,
            xlabel=DEFAULT_XLABEL,
            filename="figure_main_4panel.svg",
            abs_band_mode=args.abs_band_mode,
            delta_band_mode=args.delta_band_mode,
        )
        # Zoom version: y-axis clipped to show top-method differences
        plot_main_figure_4panel(
            agg_abs=agg_abs,
            agg_delta=agg_delta,
            outdir=outdir,
            xlabel=DEFAULT_XLABEL,
            filename="figure_main_4panel_zoom.svg",
            abs_band_mode=args.abs_band_mode,
            delta_band_mode=args.delta_band_mode,
            ymin_return=zoom_ymin,
        )

    # ── Absolute plots ────────────────────────────────────────────────────────
    if "return_mean" in agg_abs.columns:
        _plot_metric(agg_abs, "return_mean", outdir,
                     ylabel=r"$\bar{\rho}(\pi, M^*)$",
                     filename_stub="plot_return_mean",
                     band_mode=args.abs_band_mode,
                     title="Mean return",
                     include_baseline=True)
        # Zoom version
        _plot_metric(agg_abs, "return_mean", outdir,
                     ylabel=r"$\bar{\rho}(\pi, M^*)$",
                     filename_stub="plot_return_mean_zoom",
                     band_mode=args.abs_band_mode,
                     title="Mean return (zoomed — competitive methods)",
                     include_baseline=True,
                     ymin=zoom_ymin)

    if "cvar_15" in agg_abs.columns:
        _plot_metric(agg_abs, "cvar_15", outdir,
                     ylabel=r"CVaR$_{0.15}\,\bar{\rho}(\pi, M^*)$",
                     filename_stub="plot_cvar15",
                     band_mode=args.abs_band_mode,
                     title="CVaR₁₅",
                     include_baseline=True)

    if "success_rate" in agg_abs.columns:
        _plot_metric(agg_abs, "success_rate", outdir,
                     ylabel="Survival rate (not fallen)",
                     filename_stub="plot_success_rate",
                     band_mode=args.abs_band_mode,
                     title="Survival rate",
                     include_baseline=True)

    if "xvel_mean" in agg_abs.columns:
        _plot_metric(agg_abs, "xvel_mean", outdir,
                     ylabel="Mean x-velocity",
                     filename_stub="plot_xvel_mean",
                     band_mode=args.abs_band_mode,
                     title="Mean x-velocity",
                     include_baseline=True)

    # ── Delta-vs-baseline plots ───────────────────────────────────────────────
    if not agg_delta.empty:
        bsl = args.baseline_method

        if "delta_return_mean" in agg_delta.columns:
            _plot_metric(agg_delta, "delta_return_mean", outdir,
                         ylabel=rf"$\Delta$ mean return vs {bsl}",
                         filename_stub=f"plot_delta_return_mean_vs_{bsl}",
                         band_mode=args.delta_band_mode,
                         zero_line=True,
                         title=f"Return improvement vs {bsl}",
                         include_baseline=False)

        if "delta_cvar_15" in agg_delta.columns:
            _plot_metric(agg_delta, "delta_cvar_15", outdir,
                         ylabel=rf"$\Delta$ CVaR$_{{0.15}}$ vs {bsl}",
                         filename_stub=f"plot_delta_cvar15_vs_{bsl}",
                         band_mode=args.delta_band_mode,
                         zero_line=True,
                         title=f"CVaR₁₅ improvement vs {bsl}",
                         include_baseline=False)

        if "delta_success_rate" in agg_delta.columns:
            _plot_metric(agg_delta, "delta_success_rate", outdir,
                         ylabel=rf"$\Delta$ survival rate vs {bsl}",
                         filename_stub=f"plot_delta_success_rate_vs_{bsl}",
                         band_mode=args.delta_band_mode,
                         zero_line=True,
                         title=f"Survival-rate improvement vs {bsl}",
                         include_baseline=False)

        if "delta_xvel_mean" in agg_delta.columns:
            _plot_metric(agg_delta, "delta_xvel_mean", outdir,
                         ylabel=rf"$\Delta$ x-velocity vs {bsl}",
                         filename_stub=f"plot_delta_xvel_mean_vs_{bsl}",
                         band_mode=args.delta_band_mode,
                         zero_line=True,
                         title=f"x-velocity improvement vs {bsl}",
                         include_baseline=False)

        # ── Percentage-delta plots (Δ / baseline × 100%) ─────────────────────
        if "delta_return_mean_pct" in agg_delta.columns:
            _plot_metric(agg_delta, "delta_return_mean_pct", outdir,
                         ylabel=r"$\Delta$ mean return (\% of baseline)",
                         filename_stub=f"plot_delta_return_pct_vs_{bsl}",
                         band_mode=args.delta_band_mode,
                         zero_line=True,
                         title="Return improvement (% of baseline)",
                         include_baseline=False)

        if "delta_cvar_15_pct" in agg_delta.columns:
            _plot_metric(agg_delta, "delta_cvar_15_pct", outdir,
                         ylabel=r"$\Delta$ CVaR$_{0.15}$ (\% of baseline)",
                         filename_stub=f"plot_delta_cvar15_pct_vs_{bsl}",
                         band_mode=args.delta_band_mode,
                         zero_line=True,
                         title="CVaR₁₅ improvement (% of baseline)",
                         include_baseline=False)

    print(f"\n[AGG] Done. All outputs in: {outdir}")


if __name__ == "__main__":
    main()
