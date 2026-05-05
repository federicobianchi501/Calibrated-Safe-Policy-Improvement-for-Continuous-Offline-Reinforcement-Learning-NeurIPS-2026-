# =============================================================================
# Anonymous authors. Affiliations and acknowledgements will be added on
# acceptance of the associated paper. Submitted to NeurIPS 2026 (double-blind).
# =============================================================================
# safe_rl/offline/uncertainty_calibration.py
# ------------------------------------------------------------
# Calibrates ensemble uncertainty -> error upper bound.
# Fit on hold-out transitions:
#   u_i = uncertainty(s_i, a_i)
#   e_i = || s2_hat - s2_true ||  (on observation space)
# Produces a piecewise-constant bound function bound(u).
# ------------------------------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional, Dict, Any, List, Tuple
import numpy as np


@dataclass
class UncertaintyCalibratorConfig:
    n_bins: int = 30
    delta: float = 0.05             # coverage level: bound covers (1-delta) mass
    binning: str = "quantile"       # "quantile" (recommended) or "uniform"
    min_bin_size: int = 50          # merge small bins if needed
    monotone: bool = True           # enforce non-decreasing bounds
    min_samples: int = 50           # minimum samples required for "normal" calibration
    mode: str = "quantile"          # "quantile" (empirical (1-delta)-quantile) or
                                    # "wilks_strict" (Wilks order-statistic upper
                                    # tolerance limit with Bonferroni confidence 1-alpha/B)
    alpha: float = 0.05             # Wilks-strict: overall confidence (split Bonferroni over B bins)

class UncertaintyCalibrator:
    """
    Maps uncertainty u -> calibrated error bound eps_hat(u).
    Piecewise constant using bins.
    """

    def __init__(
        self,
        cfg: UncertaintyCalibratorConfig,
        bin_edges: np.ndarray,
        bin_bounds: np.ndarray,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.cfg = cfg
        self.bin_edges = np.asarray(bin_edges, dtype=np.float64).ravel()
        self.bin_bounds = np.asarray(bin_bounds, dtype=np.float64).ravel()
        self.meta = dict(meta or {})

        if self.bin_edges.ndim != 1 or self.bin_bounds.ndim != 1:
            raise ValueError("bin_edges and bin_bounds must be 1D arrays.")
        if len(self.bin_edges) < 2:
            raise ValueError("bin_edges must have at least 2 elements.")
        if len(self.bin_bounds) != (len(self.bin_edges) - 1):
            raise ValueError("bin_bounds length must be len(bin_edges)-1.")

    @staticmethod
    def _safe_quantile(x: np.ndarray, q: float) -> float:
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size == 0:
            return float("nan")
        return float(np.quantile(x, q))

    @staticmethod
    def _wilks_strict_bound(e: np.ndarray, delta: float, alpha_b: float) -> float:
        """
        One-sided Wilks (1941, Eq. 6) upper tolerance limit.

        Given n i.i.d. errors e_1..e_n with order statistics
        e_(1) <= ... <= e_(n), the k-th largest value e_(n-k+1) satisfies
            P( F(e_(n-k+1)) >= 1 - delta ) = 1 - I_{1-delta}(n-k+1, k),
        where F is the true error CDF and I is the regularized incomplete
        beta function. We return e_(n-k+1) for the LARGEST admissible k
        such that this probability is >= 1 - alpha_b; this gives the
        tightest bound that is still a valid (1-delta, 1-alpha_b)
        upper tolerance limit.

        Special cases:
          - k=1: e_max, always admissible iff n >= ceil(log(alpha_b)/log(1-delta)).
          - No admissible k: returns +inf (triggers conservative collapse).
          - scipy not available: falls back to k=1 (e_max).
        """
        e = np.asarray(e, dtype=np.float64).ravel()
        e = e[np.isfinite(e)]
        n = int(e.size)
        if n == 0:
            return float("nan")
        es = np.sort(e)
        try:
            from scipy.stats import beta as _beta
        except ImportError:
            import math
            n_min = math.log(max(alpha_b, 1e-300)) / math.log(1.0 - delta)
            if n >= int(math.ceil(n_min)):
                return float(es[-1])
            return float("inf")

        target = 1.0 - alpha_b
        best_k = 0
        for k in range(1, n + 1):
            cov_prob = float(_beta.sf(1.0 - delta, n - k + 1, k))
            if cov_prob >= target:
                best_k = k
            else:
                break  # cov_prob decreases monotonically in k
        if best_k == 0:
            return float("inf")
        return float(es[n - best_k])

    @classmethod
    def _per_bin_bound(
        cls,
        eb: np.ndarray,
        cfg: "UncertaintyCalibratorConfig",
        n_bins_eff: int,
    ) -> float:
        """
        Dispatch: compute B_b for one bin according to cfg.mode.

        - "quantile":     empirical (1-delta)-quantile of errors (point estimate)
        - "wilks_strict": Wilks order-statistic upper tolerance limit with
                          per-bin Bonferroni confidence 1 - alpha/n_bins_eff
        """
        mode = getattr(cfg, "mode", "quantile")
        if mode == "wilks_strict":
            alpha_b = float(getattr(cfg, "alpha", 0.05)) / max(1, int(n_bins_eff))
            return cls._wilks_strict_bound(eb, float(cfg.delta), alpha_b)
        # default / backward-compat: empirical quantile
        q = 1.0 - float(cfg.delta)
        return cls._safe_quantile(eb, q)

    @classmethod
    def fit(
        cls,
        u: np.ndarray,
        e: np.ndarray,
        cfg: Optional[UncertaintyCalibratorConfig] = None,
        meta: Optional[Dict[str, Any]] = None,
        bin_edges: Optional[np.ndarray] = None,
    ) -> "UncertaintyCalibrator":
        """
        Fit calibrator from arrays u (uncertainty) and e (error).

        If `bin_edges` is provided, the quantile-binning step is skipped
        and the fit uses the externally-provided edges. This enables the
        triple-split calibration pipeline of the main paper (Section
        "Dataset split and independence across the calibration pipeline"),
        where the bin edges are fitted on `D_bin` and the per-bin Wilks
        upper tolerance limits are then constructed on the independent
        sample `D_wilks`.
        """
        if cfg is None:
            cfg = UncertaintyCalibratorConfig()

        u = np.asarray(u, dtype=np.float64).ravel()
        e = np.asarray(e, dtype=np.float64).ravel()

        if u.shape[0] != e.shape[0]:
            raise ValueError(f"u and e must have same length, got {u.shape[0]} and {e.shape[0]}")

        # drop NaNs/Infs
        mask = np.isfinite(u) & np.isfinite(e)
        u = u[mask]
        e = e[mask]

        min_needed = max(10, cfg.min_samples)

        if u.size < min_needed:
            # Low-data regime: single-bin conservative calibration (implicit fallback behavior)
            q = 1.0 - float(cfg.delta)
            bound = cls._per_bin_bound(e, cfg, n_bins_eff=1)

            umin = float(np.min(u))
            umax = float(np.max(u))
            if umax <= umin:
                umax = umin + 1e-12

            bin_edges = np.array([umin, umax + 1e-12], dtype=np.float64)
            bin_bounds = np.array([bound], dtype=np.float64)

            meta_out = dict(meta or {})
            meta_out.update({
                "n_samples": int(u.size),
                "delta": float(cfg.delta),
                "quantile": float(q),
                "binning": str(cfg.binning),
                "mode": str(getattr(cfg, "mode", "quantile")),
                "n_bins_requested": int(cfg.n_bins),
                "n_bins_effective": 1,
                "low_data_regime": True,
                "min_needed": int(min_needed),
            })
            return cls(cfg=cfg, bin_edges=bin_edges, bin_bounds=bin_bounds, meta=meta_out)
        
        # Proceed with normal fitting
        if bin_edges is not None:
            # Externally-provided edges (triple-split pipeline).
            edges = np.asarray(bin_edges, dtype=np.float64).ravel()
            if edges.ndim != 1 or len(edges) < 2:
                raise ValueError("bin_edges must be a 1D array with >= 2 entries")
            n_bins = int(len(edges) - 1)
        else:
            n_bins = int(cfg.n_bins)
            n_bins = max(2, n_bins)

            # Build bin edges
            if cfg.binning == "quantile":
                qs = np.linspace(0.0, 1.0, n_bins + 1)
                edges = np.quantile(u, qs)
            elif cfg.binning == "uniform":
                umin, umax = float(np.min(u)), float(np.max(u))
                if umax <= umin:
                    umax = umin + 1e-12
                edges = np.linspace(umin, umax, n_bins + 1)
            else:
                raise ValueError("cfg.binning must be 'quantile' or 'uniform'")

        # ensure strictly increasing edges (fix duplicates due to quantiles)
        edges = np.asarray(edges, dtype=np.float64)
        # jitter duplicates minimally
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-12

        # Assign samples to bins
        # bin index in [0, n_bins-1]
        bin_idx = np.clip(np.digitize(u, edges[1:-1], right=False), 0, n_bins - 1)

        # ── Pass 1: decide merged bin structure (left-to-right, hit min_bin_size) ──
        # We collect the error arrays per merged bin *without* yet computing the
        # per-bin bound, because the Wilks-strict bound needs to know the total
        # number of effective bins (for Bonferroni alpha/B).
        merged_errors = []
        edges_list = [edges[0]]
        acc_e = []
        acc_count = 0

        for b in range(n_bins):
            eb = e[bin_idx == b]
            acc_e.append(eb)
            acc_count += int(eb.size)
            is_last = (b == n_bins - 1)
            if acc_count >= int(cfg.min_bin_size) or is_last:
                eb_all = (
                    np.concatenate(acc_e, axis=0)
                    if len(acc_e) > 0 else np.array([], dtype=np.float64)
                )
                merged_errors.append(eb_all)
                edges_list.append(edges[b + 1])
                acc_e = []
                acc_count = 0

        n_bins_eff = len(merged_errors)

        # ── Pass 2: compute per-bin bound with the chosen mode ──
        bounds_list = [
            cls._per_bin_bound(eb_all, cfg, n_bins_eff=n_bins_eff)
            for eb_all in merged_errors
        ]

        new_edges = np.asarray(edges_list, dtype=np.float64)
        new_bounds = np.asarray(bounds_list, dtype=np.float64)

        # enforce monotonicity (non-decreasing bounds) if requested
        if bool(cfg.monotone) and new_bounds.size > 0:
            # replace NaNs with previous/0
            for i in range(new_bounds.size):
                if not np.isfinite(new_bounds[i]):
                    new_bounds[i] = new_bounds[i - 1] if i > 0 else 0.0
            for i in range(1, new_bounds.size):
                if new_bounds[i] < new_bounds[i - 1]:
                    new_bounds[i] = new_bounds[i - 1]

        meta_out = dict(meta or {})
        meta_out.update(
            {
                "n_samples": int(u.size),
                "delta": float(cfg.delta),
                "quantile": float(1.0 - float(cfg.delta)),
                "binning": str(cfg.binning),
                "mode": str(getattr(cfg, "mode", "quantile")),
                "alpha": float(getattr(cfg, "alpha", 0.05)),
                "n_bins_requested": int(cfg.n_bins),
                "n_bins_effective": int(new_bounds.size),
            }
        )

        return cls(cfg=cfg, bin_edges=new_edges, bin_bounds=new_bounds, meta=meta_out)

    @classmethod
    def fit_with_selection(
        cls,
        u_wilks: np.ndarray,
        e_wilks: np.ndarray,
        u_thr: np.ndarray,
        candidate_bins: List[int],
        bin_edges_per_B: Dict[int, np.ndarray],
        cfg_template: "UncertaintyCalibratorConfig",
        q_eps: float = 0.95,
        n_bootstrap: int = 200,
        random_seed: int = 0,
    ) -> Tuple["UncertaintyCalibrator", float, dict]:
        """
        Fit calibrator over a finite candidate set of bin counts and select B_dagger
        via the one-standard-error rule on the held-out threshold split D_thr.

        Pipeline (paper Section "Bin selection via candidate sweep"):
          For each B in candidate_bins:
            1. Use bin_edges_per_B[B] (precomputed on D_bin) as fixed edges.
            2. Fit per-bin Wilks upper tolerance limits on D_wilks with
               Bonferroni-corrected confidence
                 alpha_eff(B) = cfg_template.alpha / (|candidate_bins| * B),
               accounting for the post-hoc selection of B from a finite set.
            3. Compute eps_B = quantile_{q_eps}({B^(B)(u_n) : n in D_thr}).
            4. Mark B admissible iff
                 (a) B >= B_min := ceil(1/(1-q_eps))                  (discriminability)
                 (b) min_b n_b^(B) >= ceil(log(alpha_eff)/log(1-delta))  (Wilks per-bin)

          If the admissible set is empty -> collapse: returns a single-bin
          calibrator with bound = +inf and eps = -inf, so ModelSafe is always
          False and downstream falls back to the baseline policy.

          Otherwise:
            B_best   = argmin over admissible candidates of eps_B
            se_hat   = nonparametric bootstrap std of eps over D_thr at B_best
            B_dagger = min{B in admissible : eps_B <= eps_min + se_hat}    (one-SE rule)

        Returns (calibrator_at_B_dagger, eps_dagger, diagnostics).
        diagnostics is a dict with per-candidate info, the selected B_dagger,
        eps, eps_min, eps_se, collapsed flag, and selection_rule label.
        """
        if cfg_template is None:
            cfg_template = UncertaintyCalibratorConfig()
        u_wilks = np.asarray(u_wilks, dtype=np.float64).ravel()
        e_wilks = np.asarray(e_wilks, dtype=np.float64).ravel()
        u_thr   = np.asarray(u_thr,   dtype=np.float64).ravel()

        candidate_bins = [int(B) for B in candidate_bins]
        M = max(1, len(candidate_bins))
        delta = float(cfg_template.delta)
        alpha = float(getattr(cfg_template, "alpha", 0.05))
        B_min = int(math.ceil(1.0 / max(1e-12, 1.0 - float(q_eps))))

        per_candidate = []
        cand_records: List[dict] = []

        for B in candidate_bins:
            # alpha_eff: Bonferroni over candidate set AND over bins.
            # Required because B is chosen post-hoc from candidate_bins; without
            # alpha/(M*B) the per-bin coverage guarantee no longer holds jointly
            # across candidates.
            alpha_eff = alpha / (M * max(1, B))
            edges = np.asarray(bin_edges_per_B[B], dtype=np.float64).ravel()

            # bin-edges-conditioned fit on D_wilks (Wilks-strict mode with alpha_eff)
            cfg_B = replace(
                cfg_template,
                n_bins=int(B),
                alpha=float(alpha_eff),
                mode="wilks_strict",
                # do not merge bins below this point: with fixed edges from D_bin,
                # bin merging would change the n_bins_eff and break the alpha_eff bookkeeping
                min_bin_size=1,
            )
            cal_B = cls.fit(
                u_wilks, e_wilks, cfg=cfg_B, bin_edges=edges,
            )

            # epsilon_B on D_thr
            if u_thr.size > 0:
                bound_thr = cal_B.bound_batch(u_thr)
                eps_B = float(np.quantile(bound_thr, float(q_eps))) if np.all(np.isfinite(bound_thr)) else float("inf")
            else:
                bound_thr = np.array([], dtype=np.float64)
                eps_B = float("inf")

            # admissibility
            # bin_idx: assign each u_wilks to its bin
            if len(edges) >= 2:
                bin_idx = np.clip(np.digitize(u_wilks, edges[1:-1], right=False), 0, max(0, len(edges) - 2))
                bin_counts = np.array([int((bin_idx == b).sum()) for b in range(int(B))], dtype=np.int64)
            else:
                bin_counts = np.array([0], dtype=np.int64)
            min_bin_count = int(bin_counts.min()) if bin_counts.size > 0 else 0
            # Wilks per-bin requirement: n_b >= ceil(log(alpha_eff)/log(1-delta))
            try:
                req = int(math.ceil(math.log(alpha_eff) / math.log(1.0 - delta)))
            except Exception:
                req = 10**9
            adm_bins = (B >= B_min)
            adm_wilks = (min_bin_count >= req)
            admissible = bool(adm_bins and adm_wilks)

            rec = dict(
                B=int(B),
                n_bins=int(B),
                n_bins_eff=int(len(cal_B.bin_bounds)),
                bin_counts=bin_counts.tolist(),
                min_bin_count=int(min_bin_count),
                required_min_bin_count=int(req),
                alpha_eff=float(alpha_eff),
                epsilon_B=float(eps_B),
                admissible=bool(admissible),
                admissible_B_min=bool(adm_bins),
                admissible_wilks=bool(adm_wilks),
                selected=False,
                collapsed=False,
            )
            per_candidate.append(rec)
            cand_records.append(dict(record=rec, calibrator=cal_B, bound_thr=bound_thr))

        # Build admissible subset
        admissible_subset = [c for c in cand_records if c["record"]["admissible"]]

        diagnostics = dict(
            per_candidate=per_candidate,
            B_min=int(B_min),
            num_candidate_bins=int(M),
            num_admissible_bins=int(len(admissible_subset)),
            selection_rule="one_standard_error",
            n_bootstrap=int(n_bootstrap),
        )

        if not admissible_subset:
            # Conservative collapse: build a single-bin calibrator that returns +inf
            u_min = float(np.min(u_wilks)) if u_wilks.size else 0.0
            u_max = float(np.max(u_wilks)) if u_wilks.size else 1.0
            if u_max <= u_min:
                u_max = u_min + 1e-12
            col_edges = np.array([u_min, u_max + 1e-12], dtype=np.float64)
            col_bounds = np.array([float("inf")], dtype=np.float64)
            col_meta = dict(
                collapsed=True,
                candidate_bins=list(candidate_bins),
                B_dagger=None,
                B_best=None,
                epsilon=float("-inf"),
                epsilon_min=float("inf"),
                epsilon_se=0.0,
                B_min=int(B_min),
                num_candidate_bins=int(M),
                num_admissible_bins=0,
                selection_rule="one_standard_error",
                reason="empty_admissible_set",
            )
            calibrator = cls(
                cfg=cfg_template, bin_edges=col_edges, bin_bounds=col_bounds, meta=col_meta,
            )
            diagnostics.update(
                collapsed=True, B_dagger=None, B_best=None,
                epsilon=float("-inf"), epsilon_min=float("inf"), epsilon_se=0.0,
                min_bin_count_selected=0, required_min_bin_count_selected=0,
            )
            return calibrator, float("-inf"), diagnostics

        # Find B_best = argmin epsilon_B over admissible
        eps_admissible = [c["record"]["epsilon_B"] for c in admissible_subset]
        best_idx = int(np.argmin(eps_admissible))
        cand_best = admissible_subset[best_idx]
        eps_min = float(cand_best["record"]["epsilon_B"])
        bound_thr_best = cand_best["bound_thr"]

        # Bootstrap SE of the q_eps-quantile of B_best(u) on D_thr
        if n_bootstrap > 0 and bound_thr_best.size > 0:
            rng = np.random.default_rng(int(random_seed))
            N_thr = bound_thr_best.size
            boot_eps = np.empty(int(n_bootstrap), dtype=np.float64)
            for i in range(int(n_bootstrap)):
                idx = rng.integers(0, N_thr, size=N_thr)
                boot_eps[i] = float(np.quantile(bound_thr_best[idx], float(q_eps)))
            eps_se = float(np.std(boot_eps, ddof=1)) if n_bootstrap > 1 else 0.0
        else:
            eps_se = 0.0

        # one-SE rule: smallest B with epsilon_B <= eps_min + eps_se
        threshold = eps_min + eps_se
        admissible_sorted = sorted(admissible_subset, key=lambda c: c["record"]["B"])
        chosen = None
        for c in admissible_sorted:
            if c["record"]["epsilon_B"] <= threshold + 1e-12:
                chosen = c
                break
        if chosen is None:
            chosen = cand_best  # safety fallback

        B_dagger = int(chosen["record"]["B"])
        cal_dagger = chosen["calibrator"]
        eps_dagger = float(chosen["record"]["epsilon_B"])

        # Mark selected in per_candidate diagnostics
        for rec in per_candidate:
            rec["selected"] = (rec["B"] == B_dagger)

        diagnostics.update(
            collapsed=False,
            B_dagger=B_dagger,
            B_best=int(cand_best["record"]["B"]),
            epsilon=eps_dagger,
            epsilon_min=eps_min,
            epsilon_se=eps_se,
            min_bin_count_selected=int(chosen["record"]["min_bin_count"]),
            required_min_bin_count_selected=int(chosen["record"]["required_min_bin_count"]),
        )

        # Augment calibrator meta so it carries the diagnostics
        try:
            cal_dagger.meta.update(dict(
                collapsed=False,
                candidate_bins=list(candidate_bins),
                B_dagger=B_dagger,
                B_best=int(cand_best["record"]["B"]),
                epsilon=eps_dagger,
                epsilon_min=eps_min,
                epsilon_se=eps_se,
                B_min=int(B_min),
                num_candidate_bins=int(M),
                num_admissible_bins=int(len(admissible_subset)),
                selection_rule="one_standard_error",
            ))
        except Exception:
            pass

        return cal_dagger, eps_dagger, diagnostics

    def bound(self, u: float) -> float:
        """
        Return calibrated error bound eps_hat(u).
        """
        uu = float(u)
        # clamp to range
        if uu <= float(self.bin_edges[0]):
            return float(self.bin_bounds[0])
        if uu >= float(self.bin_edges[-1]):
            return float(self.bin_bounds[-1])

        # find bin
        # idx in [0, nb-1]
        idx = int(np.clip(np.searchsorted(self.bin_edges, uu, side="right") - 1, 0, len(self.bin_bounds) - 1))
        return float(self.bin_bounds[idx])

    def bound_batch(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64).ravel()
        out = np.empty_like(u, dtype=np.float64)
        for i in range(u.size):
            out[i] = self.bound(float(u[i]))
        return out

    def save(self, path: str) -> None:
        payload = {
            "cfg": self.cfg,
            "bin_edges": self.bin_edges,
            "bin_bounds": self.bin_bounds,
            "meta": self.meta,
        }
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str) -> "UncertaintyCalibrator":
        data = np.load(path, allow_pickle=True)
        cfg = data["cfg"].item()
        if not isinstance(cfg, UncertaintyCalibratorConfig):
            # backward compatibility: cfg as dict
            cfg = UncertaintyCalibratorConfig(**dict(cfg))
        bin_edges = data["bin_edges"]
        bin_bounds = data["bin_bounds"]
        meta = data["meta"].item() if "meta" in data else {}
        return cls(cfg=cfg, bin_edges=bin_edges, bin_bounds=bin_bounds, meta=meta)
