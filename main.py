"""
Analytic Hierarchy Process (AHP)
=================================
Goal: Rank countries by business environment attractiveness.

Two inputs (hard-coded in __main__):
  1. criteria_pcm  – (n_c x n_c) pairwise comparison matrix of criteria
  2. performance   – (n_a x n_c) raw performance table (all criteria → max)

From the performance table the script automatically derives n_c pairwise
comparison matrices for alternatives (one per criterion) using the ratio
  a_ij = score_i / score_j   (correct for → max direction)

Algorithm
---------
  0. Validate all PCMs
  1. Criteria weights  (RGMM + consistency check / auto-improvement)
  2. Alternative local weights per criterion (derived PCMs)
  3. Consolidated local-weights table
  4. Global weights via distributive synthesis  →  ranking
"""

from __future__ import annotations
import math
from typing import Optional
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
#  Random Consistency Index (Saaty)
# ──────────────────────────────────────────────────────────────────────────────
_MRCI_TABLE: dict[int, float] = {
    1: 0.00, 2: 0.00, 3: 0.52, 4: 0.89, 5: 1.11,
    6: 1.25, 7: 1.35, 8: 1.40, 9: 1.45, 10: 1.49,
    11: 1.52, 12: 1.54, 13: 1.56, 14: 1.58, 15: 1.59,
}

_CR_HARD_THRESHOLDS: dict[int, float] = {3: 0.05, 4: 0.08}
_CR_DEFAULT_THRESHOLD = 0.10


def _mrci(n: int) -> float:
    """Random consistency index for an n×n matrix (linear extrapolation for n>15)."""
    return _MRCI_TABLE.get(n, 1.59 + (n - 15) * 0.005)


def _cr_threshold(n: int) -> float:
    """Acceptable CR threshold for matrix size n."""
    return _CR_HARD_THRESHOLDS.get(n, _CR_DEFAULT_THRESHOLD)


# ──────────────────────────────────────────────────────────────────────────────
#  Core AHP math
# ──────────────────────────────────────────────────────────────────────────────

def rgmm_weights(A: np.ndarray) -> np.ndarray:
    """
    Row Geometric Mean Method (RGMM).
    Unnormalised: v_i = (prod_j a_ij)^(1/n)
    Normalised:   w_i = v_i / sum(v)
    """
    n = A.shape[0]
    v = np.prod(A, axis=1) ** (1.0 / n)
    return v / v.sum()


def lambda_max(A: np.ndarray, w: np.ndarray) -> float:
    """
    Approximate lambda_max via column sums:
      sigma_i = sum_j a_ji ,  lambda_max = sum_i sigma_i * w_i
    """
    return float(A.sum(axis=0) @ w)


def consistency_index(lmax: float, n: int) -> float:
    """CI = (lambda_max - n) / (n - 1).  Returns 0 for n <= 1."""
    return 0.0 if n <= 1 else (lmax - n) / (n - 1)


def consistency_ratio(ci: float, n: int) -> float:
    """CR = CI / MRCI.  Returns 0 when MRCI = 0 (n <= 2)."""
    ri = _mrci(n)
    return 0.0 if ri == 0.0 else ci / ri


def analyse_pcm(A: np.ndarray, label: str = "") -> dict:
    """
    Full consistency analysis of a PCM.
    Returns a dict with: n, weights, lambda_max, CI, MRCI, CR,
                         threshold, is_consistent, label.
    """
    n = A.shape[0]
    w = rgmm_weights(A)
    lmax = lambda_max(A, w)
    ci = consistency_index(lmax, n)
    ri = _mrci(n)
    cr = consistency_ratio(ci, n)
    thr = _cr_threshold(n)
    return {
        "label": label,
        "n": n,
        "weights": w,
        "lambda_max": lmax,
        "CI": ci,
        "MRCI": ri,
        "CR": cr,
        "threshold": thr,
        "is_consistent": cr <= thr,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  PCM derivation from raw performance scores
# ──────────────────────────────────────────────────────────────────────────────

def pcm_from_scores(scores: np.ndarray) -> np.ndarray:
    """
    Build an (n×n) PCM from a 1-D array of raw scores (all → max).
      a_ij = scores[i] / scores[j]

    Edge case: zero or negative scores are replaced with a tiny epsilon
    to prevent division-by-zero while preserving relative ordering.
    """
    eps = 1e-9
    s = np.where(scores <= 0, eps, scores.astype(float))
    return s[:, None] / s[None, :]   # outer division → (n, n)


def alt_pcms_from_performance(performance: np.ndarray) -> list[np.ndarray]:
    """
    Derive one PCM per criterion from an (n_alt × n_crit) performance table.
    Returns a list of n_crit square PCMs, each of shape (n_alt × n_alt).
    """
    return [pcm_from_scores(performance[:, j]) for j in range(performance.shape[1])]


# ──────────────────────────────────────────────────────────────────────────────
#  Automatic consistency improvement
# ──────────────────────────────────────────────────────────────────────────────

def _build_geom_step(Ak: np.ndarray, w: np.ndarray, alpha: float) -> np.ndarray:
    """One iteration: weighted geometric mean.  a*_ij = a_ij^alpha * (w_i/w_j)^(1-alpha)"""
    n = Ak.shape[0]
    A_new = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            ratio = w[i] / w[j] if w[j] > 0 else 1.0
            A_new[i, j] = (Ak[i, j] ** alpha) * (ratio ** (1.0 - alpha))
            A_new[j, i] = 1.0 / A_new[i, j]
    return A_new


def _build_arith_step(Ak: np.ndarray, w: np.ndarray, alpha: float) -> np.ndarray:
    """One iteration: weighted arithmetic mean.  a*_ij = alpha*a_ij + (1-alpha)*(w_i/w_j)"""
    n = Ak.shape[0]
    A_new = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            ratio = w[i] / w[j] if w[j] > 0 else 1.0
            A_new[i, j] = alpha * Ak[i, j] + (1.0 - alpha) * ratio
            A_new[j, i] = 1.0 / A_new[i, j]
    return A_new


def improve_consistency(
    A: np.ndarray,
    method: str = "geom",
    alpha: float = 0.8,
    max_iter: int = 200,
    thr: Optional[float] = None,
) -> tuple[np.ndarray, list[float], int]:
    """
    Iteratively reduce CR of PCM A.

    Parameters
    ----------
    A        : original PCM
    method   : 'geom'  – weighted geometric mean
               'arith' – weighted arithmetic mean
    alpha    : blending weight in (0, 1)
    max_iter : safety cap on iterations
    thr      : target CR (default: size-dependent standard threshold)

    Returns
    -------
    (improved_PCM, cr_history_list, n_iterations_used)
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be strictly in (0, 1)")
    n = A.shape[0]
    threshold = thr if thr is not None else _cr_threshold(n)
    step_fn = _build_geom_step if method == "geom" else _build_arith_step

    Ak = A.copy()
    cr_history: list[float] = []

    def _cr(M: np.ndarray) -> float:
        w_loc = rgmm_weights(M)
        return consistency_ratio(consistency_index(lambda_max(M, w_loc), n), n)

    for k in range(max_iter):
        w = rgmm_weights(Ak)
        cr = consistency_ratio(consistency_index(lambda_max(Ak, w), n), n)
        cr_history.append(cr)
        if cr <= threshold:
            return Ak, cr_history, k
        Ak = step_fn(Ak, w, alpha)

    # Record CR of the matrix actually returned (post-final-step) so the
    # last entry of cr_history always describes the returned matrix.
    cr_history.append(_cr(Ak))
    return Ak, cr_history, max_iter


def efficiency_metrics(A_orig: np.ndarray, A_mod: np.ndarray) -> dict:
    """
    Modification quality (Saaty criteria):
      delta = max |a^(k)_ij - a^(0)_ij|          should be < 0.2
      sigma = (1/n) * sqrt(sum(a^(k)_ij-a^(0)_ij)^2)  should be < 0.1
    """
    n = A_orig.shape[0]
    diff = A_mod - A_orig
    delta = float(np.abs(diff).max())
    sigma = float(np.sqrt((diff ** 2).sum()) / n)
    return {
        "delta": delta, "delta_ok": delta < 0.2,
        "sigma": sigma, "sigma_ok": sigma < 0.1,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Validation
# ──────────────────────────────────────────────────────────────────────────────

def validate_pcm(A: np.ndarray, label: str = "") -> list[str]:
    """
    Verify a PCM for: square shape, positive elements,
    unit diagonal, reciprocal symmetry (a_ij * a_ji ~= 1).
    Returns a list of warning strings (empty list means OK).
    """
    issues: list[str] = []
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        issues.append(f"[{label}] Not square: shape {A.shape}")
        return issues

    n = A.shape[0]

    if np.any(A <= 0):
        issues.append(f"[{label}] Contains non-positive element(s)")

    for i in range(n):
        if not math.isclose(A[i, i], 1.0, rel_tol=1e-6):
            issues.append(f"[{label}] Diagonal a[{i},{i}] = {A[i,i]:.6f} != 1")

    for i in range(n):
        for j in range(i + 1, n):
            prod = A[i, j] * A[j, i]
            if not math.isclose(prod, 1.0, rel_tol=1e-4):
                issues.append(
                    f"[{label}] Reciprocity violation: "
                    f"a[{i},{j}]={A[i,j]:.4f}, a[{j},{i}]={A[j,i]:.4f}, "
                    f"product={prod:.6f}"
                )
    return issues


def saaty_scale_audit(
    A: np.ndarray,
    label: str = "",
    scale_max: float = 9.0,
) -> list[str]:
    """
    Warn when a PCM entry exceeds Saaty's 1-9 judgment scale.
    Off-scale entries (typical for ratio-derived PCMs over disparate raw scores)
    let a single criterion's data swamp the global synthesis, bypassing the
    compression the 1-9 scale provides.
    Returns a list of warning strings (empty list means within scale).
    """
    issues: list[str] = []
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return issues
    max_entry = float(A.max())
    if max_entry > scale_max:
        i, j = (int(x) for x in np.unravel_index(int(A.argmax()), A.shape))
        issues.append(
            f"[{label}] Saaty-scale violation: a[{i},{j}] = {max_entry:.2f} > {scale_max:.0f} "
            f"(ratio {max_entry:.1f}:1 -- single-criterion dominance risk)"
        )
    return issues


# ──────────────────────────────────────────────────────────────────────────────
#  Synthesis
# ──────────────────────────────────────────────────────────────────────────────

def distributive_synthesis(
    criteria_weights: np.ndarray,   # shape (n_c,)
    alt_local_weights: np.ndarray,  # shape (n_a, n_c)
) -> np.ndarray:                    # returns shape (n_a, )
    """Global priorities: w_i^glob = sum_j w_j^C * r_ij"""
    return alt_local_weights @ criteria_weights


# ──────────────────────────────────────────────────────────────────────────────
#  Pretty-print helpers
# ──────────────────────────────────────────────────────────────────────────────
_SEP = "=" * 68


def _print_analysis(info: dict) -> None:
    print(f"\n  {'-'*52}")
    print(f"  {info['label']}")
    print(f"  {'-'*52}")
    print(f"  n         = {info['n']}")
    print(f"  lambda_max= {info['lambda_max']:.4f}")
    print(f"  CI        = {info['CI']:.4f}")
    print(f"  MRCI      = {info['MRCI']:.4f}")
    print(f"  CR        = {info['CR'] * 100:.2f}%  (threshold {info['threshold'] * 100:.0f}%)")
    status = "CONSISTENT [OK]" if info["is_consistent"] else "INCONSISTENT [!]"
    print(f"  Status    : {status}")
    print(f"  Weights   :")
    for i, wi in enumerate(info["weights"]):
        print(f"    [{i + 1}] {wi:.4f}")


def _maybe_improve(
    A: np.ndarray,
    label: str,
    alpha: float,
    user_thr: float,
) -> np.ndarray:
    """
    Check CR; if it exceeds user_thr, run both improvement methods,
    print comparison, and return the best weights (lower CR wins).
    """
    info = analyse_pcm(A, label)
    if info["CR"] <= user_thr:
        return info["weights"]

    print(
        f"\n  CR = {info['CR']*100:.2f}% > {user_thr*100:.0f}%"
        f"  ->  automatic consistency improvement"
    )

    best_cr: float = float("inf")
    best_method: str = "geom"          # initialised; overwritten in the loop
    best_w: np.ndarray = info["weights"]

    for method in ("geom", "arith"):
        A_imp, history, iters = improve_consistency(
            A, method=method, alpha=alpha, thr=user_thr
        )
        info_imp = analyse_pcm(A_imp, f"{label} ({method})")
        em = efficiency_metrics(A, A_imp)
        mname = "Geometric " if method == "geom" else "Arithmetic"
        print(
            f"  [{mname}]  iter={iters:3d}  "
            f"CR={info_imp['CR']*100:.2f}%  "
            f"delta={em['delta']:.4f}({'OK' if em['delta_ok'] else '!'})  "
            f"sigma={em['sigma']:.4f}({'OK' if em['sigma_ok'] else '!'})  "
            f"path: {[f'{x*100:.1f}%' for x in history]}"
        )
        if info_imp["CR"] < best_cr:
            best_cr = info_imp["CR"]
            best_w = info_imp["weights"]
            best_method = method

    print(f"  -> Selected: {'geometric' if best_method == 'geom' else 'arithmetic'} method")
    return best_w


# ──────────────────────────────────────────────────────────────────────────────
#  Main AHP runner
# ──────────────────────────────────────────────────────────────────────────────

def run_ahp(
    criteria_pcm: np.ndarray,
    performance: np.ndarray,
    criteria_names: list[str],
    alt_names: list[str],
    alpha: float = 0.8,
    user_cr_threshold: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Execute the full AHP pipeline.

    Parameters
    ----------
    criteria_pcm      : (n_c x n_c) expert pairwise comparison matrix for criteria
    performance       : (n_a x n_c) raw scores; all columns must be -> max
                        (invert min-criteria before passing: use 1/value)
    criteria_names    : list of n_c criterion labels
    alt_names         : list of n_a alternative labels
    alpha             : blending weight for auto-improvement, in (0, 1)
    user_cr_threshold : decision-maker's acceptable CR level (e.g. 0.15)

    Returns
    -------
    (global_weights, ranking_indices)
      global_weights  – (n_a,) array of global priority scores
      ranking_indices – (n_a,) indices sorted descending by priority
    """
    n_c = len(criteria_names)
    n_a = len(alt_names)

    if criteria_pcm.shape != (n_c, n_c):
        raise ValueError(f"criteria_pcm must be {n_c}x{n_c}, got {criteria_pcm.shape}")
    if performance.shape != (n_a, n_c):
        raise ValueError(f"performance must be {n_a}x{n_c}, got {performance.shape}")

    # Derive all alternative PCMs from the performance table
    alt_pcms = alt_pcms_from_performance(performance)

    print(_SEP)
    print("  ANALYTIC HIERARCHY PROCESS  (AHP)")
    print("  Goal: Rank countries by business environment attractiveness")
    print(_SEP)

    # ── 0. Validate ───────────────────────────────────────────────────────────
    print("\n[0] PCM VALIDATION")
    all_ok = True
    checks = [("Criteria PCM", criteria_pcm)] + [
        (f"Alt PCM - {criteria_names[j]}", alt_pcms[j]) for j in range(n_c)
    ]
    for lbl, mat in checks:
        issues = validate_pcm(mat, lbl)
        if issues:
            all_ok = False
            for msg in issues:
                print(f"  [!]  {msg}")
    if all_ok:
        print("  All PCMs passed structural validation [OK]")

    # Saaty 1-9 scale audit (derived alternative PCMs are most at risk of
    # off-scale entries when raw scores span orders of magnitude).
    print("\n  Saaty 1-9 scale audit:")
    scale_warnings = 0
    for lbl, mat in checks:
        for msg in saaty_scale_audit(mat, lbl):
            print(f"  [!]  {msg}")
            scale_warnings += 1
    if scale_warnings == 0:
        print("    All PCMs within 1-9 scale [OK]")
    else:
        print(f"    {scale_warnings} criterion(criteria) outside 1-9 scale "
              f"-> that column's data dominates the synthesis.")

    # ── 1. Criteria weights ───────────────────────────────────────────────────
    print(f"\n{_SEP}")
    print("[1] CRITERIA PAIRWISE COMPARISON MATRIX")
    print(_SEP)

    info_c = analyse_pcm(criteria_pcm, "Criteria PCM")
    _print_analysis(info_c)
    criteria_weights = _maybe_improve(
        criteria_pcm, "Criteria PCM", alpha, user_cr_threshold
    )

    # ── 2. Alternative local weights ──────────────────────────────────────────
    print(f"\n{_SEP}")
    print("[2] ALTERNATIVE PCMs  (ratio method: a_ij = score_i / score_j)")
    print(_SEP)

    alt_local_weights = np.zeros((n_a, n_c))
    for j in range(n_c):
        label = f"Alt PCM - {criteria_names[j]}"
        info_alt = analyse_pcm(alt_pcms[j], label)
        _print_analysis(info_alt)
        alt_local_weights[:, j] = _maybe_improve(
            alt_pcms[j], label, alpha, user_cr_threshold
        )

    # ── 3. Consolidated table ─────────────────────────────────────────────────
    print(f"\n{_SEP}")
    print("[3] CONSOLIDATED LOCAL WEIGHTS TABLE")
    print(_SEP)

    col = 12
    print(
        f"\n  {'Alternative':<22}"
        + "".join(f"{'C' + str(j + 1):>{col}}" for j in range(n_c))
    )
    print(
        f"  {'Criteria weights':<22}"
        + "".join(f"{criteria_weights[j]:>{col}.4f}" for j in range(n_c))
    )
    print(f"  {'-' * (22 + col * n_c)}")
    for i, aname in enumerate(alt_names):
        print(
            f"  {aname:<22}"
            + "".join(f"{alt_local_weights[i, j]:>{col}.4f}" for j in range(n_c))
        )

    # ── 4. Global weights & ranking ───────────────────────────────────────────
    print(f"\n{_SEP}")
    print("[4] GLOBAL WEIGHTS & RANKING  (distributive synthesis)")
    print(_SEP)

    global_weights = distributive_synthesis(criteria_weights, alt_local_weights)
    ranking = np.argsort(global_weights)[::-1]
    ranking_indices: list[int] = ranking.tolist()

    roman = [
        "I", "II", "III", "IV", "V", "VI", "VII", "VIII",
        "IX", "X", "XI", "XII", "XIII", "XIV", "XV",
    ]

    print(f"\n  {'Rank':<6} {'Alternative':<24} {'Global weight':>14}")
    print(f"  {'-' * 46}")
    for pos, alt_idx in enumerate(ranking_indices):
        rank_label = roman[pos] if pos < len(roman) else str(pos + 1)
        print(f"  {rank_label:<6} {alt_names[alt_idx]:<24} {global_weights[alt_idx]:>14.4f}")

    total = float(global_weights.sum())
    total_ok = abs(total - 1.0) < 1e-6
    print(f"\n  Sum of global weights: {total:.6f}  {'~= 1 [OK]' if total_ok else '!= 1 [!]'}")

    print(f"\n{_SEP}")
    best_idx = ranking_indices[0]
    print(
        f"  CONCLUSION: Top-ranked alternative -> {alt_names[best_idx]}"
        f"  (w = {global_weights[best_idx]:.4f})"
    )
    print(_SEP)

    return global_weights, ranking


# ──────────────────────────────────────────────────────────────────────────────
#  Entry point  –  ALL DATA IS DEFINED HERE
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    CRITERIA_NAMES: list[str] = [
        "C1: Ease of doing business",
        "C2: Absence of corruption",
        "C3: Favourable taxation",
        "C4: Market access",
        "C5: Infrastructure quality",
        "C6: Legal protection",
        "C7: Economic stability",
    ]

    ALT_NAMES: list[str] = [
        "Ukraine",
        "USA",
        "United Kingdom",
        "France",
        "Germany",
        "Russia",
        "China",
        "Japan",
    ]

    # ── INPUT 1: Expert pairwise comparison matrix for criteria (7×7) ─────────
    # Rows / columns correspond to C1 … C7 in order.
    CRITERIA_PCM = np.array([
        [1,    1/3,  3,    1/5,  1/3,  1/4,  1/2],
        [3,    1,    5,    1/3,  2,    1/2,  2  ],
        [1/3,  1/5,  1,    1/7,  1/5,  1/6,  1/4],
        [5,    3,    7,    1,    3,    2,    4  ],
        [3,    1/2,  5,    1/3,  1,    1/3,  2  ],
        [4,    2,    6,    1/2,  3,    1,    3  ],
        [2,    1/2,  4,    1/4,  1/2,  1/3,  1  ],
    ], dtype=float)

    # ── INPUT 2: Raw performance table (8 alternatives × 7 criteria) ──────────
    # All columns must be expressed as -> max.
    #   raw score maps to a higher priority through the ratio a_ij = si / sj
    PERFORMANCE = np.array([
        [70.2,   36,   1/18.0,  0.191,   2.7,  0.49,  1/6.5 ],  # Ukraine
        [84.0,   64,   1/21.0,  28.751,  3.8,  0.70,  1/2.9 ],  # USA
        [83.5,   70,   1/25.0,  3.686,   3.7,  0.79,  1/3.3 ],  # United Kingdom
        [76.8,   66,   1/25.0,  3.160,   3.9,  0.74,  1/2.0 ],  # France
        [79.7,   77,   1/29.9,  4.686,   4.1,  0.83,  1/2.2 ],  # Germany
        [78.2,   22,   1/25.0,  2.174,   2.6,  0.42,  1/9.5 ],  # Russia
        [77.9,   43,   1/25.0,  18.744,  3.7,  0.47,  1/0.2 ],  # China
        [78.0,   71,   1/30.6,  4.028,   4.0,  0.78,  1/2.7 ],  # Japan
    ], dtype=float)

    # ── Execute ───────────────────────────────────────────────────────────────
    result_weights, result_ranking = run_ahp(
        criteria_pcm=CRITERIA_PCM,
        performance=PERFORMANCE,
        criteria_names=CRITERIA_NAMES,
        alt_names=ALT_NAMES,
        alpha=0.8,
        user_cr_threshold=0.15,   # decision-maker's threshold: 15 %
    )
