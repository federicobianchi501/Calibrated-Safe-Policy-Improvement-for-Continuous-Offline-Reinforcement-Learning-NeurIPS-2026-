# =============================================================================
# Calibrated Safe Policy Improvement for Continuous Offline Reinforcement Learning
# Federico Bianchi, Alessandro Farinelli, Alberto Castellini
# NeurIPS 2026
# =============================================================================
"""Episode-level summary statistics: mean/std, percentiles, CVaR, success and fallback rates."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import numpy as np


def cvar(x: np.ndarray, alpha: float = 0.15) -> float:
    """Lower-tail CVaR: mean of the ceil(alpha * n) smallest values (at least one)."""
    v = np.sort(np.asarray(x, dtype=np.float64).ravel())
    if v.size == 0:
        return float("nan")
    k = max(1, int(np.ceil(alpha * v.size)))
    return float(np.mean(v[:k]))


def percentile(x: np.ndarray, p: float) -> float:
    """NumPy percentile of a flattened array; NaN for empty input."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, p))


@dataclass
class SummaryStats:
    """Aggregated metrics over a set of evaluation episodes.

    `fallback_mean_pct` is in percent; `lot_decision_mean` and `lot_env_mean` are means of
    the per-episode step counts passed to `summarize_episodes`, measured in decision steps
    and environment steps. Optional fields are NaN when not provided.
    """

    return_mean: float
    return_std: float
    return_p50: float
    return_p90: float
    cvar15: float
    success_rate: float
    final_dist_mean: float = float("nan")
    final_dist_std: float = float("nan")
    min_dist_mean: float = float("nan")
    min_dist_std: float = float("nan")
    fallback_mean_pct: float = float("nan")
    lot_decision_mean: float = float("nan")
    lot_env_mean: float = float("nan")

    def to_dict(self) -> Dict[str, Any]:
        """Return the fields as a plain dict."""
        return asdict(self)


def summarize_episodes(
    returns: np.ndarray,
    success: Optional[np.ndarray] = None,
    final_dist: Optional[np.ndarray] = None,
    min_dist: Optional[np.ndarray] = None,
    fallback_rate: Optional[np.ndarray] = None,
    lot_decision_steps: Optional[np.ndarray] = None,
    lot_env_steps: Optional[np.ndarray] = None,
) -> SummaryStats:
    """Summarize per-episode arrays into a SummaryStats.

    `success` defaults to all zeros; `fallback_rate` is a per-episode fraction in [0, 1]
    and is reported as a percentage. CVaR uses alpha = 0.15.
    """
    rets = np.asarray(returns, dtype=np.float64).ravel()

    if success is None:
        succ = np.zeros_like(rets)
    else:
        succ = np.asarray(success, dtype=np.float64).ravel()

    def _meanstd(x: Optional[np.ndarray]):
        if x is None:
            return float("nan"), float("nan")
        x = np.asarray(x, dtype=np.float64).ravel()
        return float(np.mean(x)) if x.size else float("nan"), float(np.std(x)) if x.size else float("nan")

    fd_m, fd_s = _meanstd(final_dist)
    md_m, md_s = _meanstd(min_dist)

    fb = np.asarray(fallback_rate, dtype=np.float64).ravel() if fallback_rate is not None else None
    fb_mean_pct = float(np.mean(fb) * 100.0) if (fb is not None and fb.size) else float("nan")

    lot_dec = np.asarray(lot_decision_steps, dtype=np.float64).ravel() if lot_decision_steps is not None else None
    lot_env = np.asarray(lot_env_steps, dtype=np.float64).ravel() if lot_env_steps is not None else None

    return SummaryStats(
        return_mean=float(np.mean(rets)) if rets.size else float("nan"),
        return_std=float(np.std(rets)) if rets.size else float("nan"),
        return_p50=percentile(rets, 50),
        return_p90=percentile(rets, 90),
        cvar15=cvar(rets, 0.15),
        success_rate=float(np.mean(succ)) if succ.size else float("nan"),
        final_dist_mean=fd_m,
        final_dist_std=fd_s,
        min_dist_mean=md_m,
        min_dist_std=md_s,
        fallback_mean_pct=fb_mean_pct,
        lot_decision_mean=float(np.mean(lot_dec)) if (lot_dec is not None and lot_dec.size) else float("nan"),
        lot_env_mean=float(np.mean(lot_env)) if (lot_env is not None and lot_env.size) else float("nan"),
    )
