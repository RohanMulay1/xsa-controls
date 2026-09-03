"""
xsac.stats - paired tests, bootstrap, reliability and power.

Everything here reports an ``n`` and an interval. The spec's anti-pattern table
forbids reporting a mean without them, and that rule is enforced by returning
them in the same dict rather than leaving it to the caller to remember.

Two lessons from the CRPA campaign are baked in:

* An agreement statistic computed over a degenerate partition is uninformative
  and must come back undefined rather than as a reassuring 1.0.
* A threshold sitting orders of magnitude above the signal is a no-op filter.
  ``threshold_sanity`` makes that computable before an experiment runs, not
  after it has produced a table of ones.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:  # scipy is in requirements, but the module degrades rather than dies.
    from scipy import stats as _sps
except Exception:  # pragma: no cover
    _sps = None


def _finite(xs: Sequence[float]) -> List[float]:
    return [float(x) for x in xs if x is not None and math.isfinite(float(x))]


def mean_ci(values: Sequence[float], confidence: float = 0.95,
            n_boot: int = 10000, seed: int = 0) -> Dict[str, float]:
    """Bootstrap mean with a percentile interval. Always reports n."""
    xs = _finite(values)
    n = len(xs)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0,
                "ci_low": float("nan"), "ci_high": float("nan")}
    if n == 1:
        return {"mean": xs[0], "std": 0.0, "n": 1,
                "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    arr = np.asarray(xs, dtype=float)
    boot = rng.choice(arr, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.quantile(boot, [(1 - confidence) / 2, 1 - (1 - confidence) / 2])
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)),
            "n": n, "ci_low": float(lo), "ci_high": float(hi)}


def paired_test(treatment: Sequence[float], control: Sequence[float],
                confidence: float = 0.95, seed: int = 0) -> Dict[str, object]:
    """One-sample t-test on the paired differences, plus Wilcoxon.

    The unpaired std across seeds is returned alongside, because the contrast
    between it and the paired std is itself a result: it shows how much noise
    the pairing removes, and XSA's paper reports neither.
    """
    if len(treatment) != len(control):
        raise ValueError("paired test needs equal lengths, got {} and {}"
                         .format(len(treatment), len(control)))
    pairs = [(float(a), float(b)) for a, b in zip(treatment, control)
             if math.isfinite(float(a)) and math.isfinite(float(b))]
    n = len(pairs)
    out: Dict[str, object] = {"n": n}
    if n == 0:
        return {**out, "mean_delta": float("nan"), "t": float("nan"),
                "p": float("nan"), "cohen_dz": float("nan"),
                "wilcoxon_p": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "sd_paired": float("nan"),
                "sd_unpaired_treatment": float("nan"),
                "sd_unpaired_control": float("nan")}

    deltas = [a - b for a, b in pairs]
    arr = np.asarray(deltas, dtype=float)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 and sd > 0 else float("nan")

    # Zero-variance limit. If every seed moved by the same non-zero amount,
    # the t statistic diverges and its p-value goes to 0. That is the correct
    # limit and it is the strongest evidence the design can produce, so
    # returning NaN here would report the best possible result as undefined.
    # It is flagged, because with small n an exactly constant difference more
    # often means the arms are not actually independent than that the effect
    # is infinitely certain.
    degenerate = bool(n > 1 and sd == 0.0)
    if degenerate:
        t = math.inf if mean > 0 else (-math.inf if mean < 0 else 0.0)
    else:
        t = mean / se if se and math.isfinite(se) and se > 0 else float("nan")

    p = float("nan")
    wil = float("nan")
    if degenerate:
        p = 0.0 if mean != 0 else 1.0
    elif _sps is not None and n > 1:
        if math.isfinite(t):
            p = float(2 * _sps.t.sf(abs(t), df=n - 1))
        if any(d != 0 for d in deltas) and n >= 6:
            try:
                wil = float(_sps.wilcoxon(arr).pvalue)
            except Exception:
                wil = float("nan")

    ci = mean_ci(deltas, confidence=confidence, seed=seed)
    treat = np.asarray([a for a, _ in pairs], dtype=float)
    ctrl = np.asarray([b for _, b in pairs], dtype=float)
    return {**out,
            "mean_delta": mean,
            "sd_paired": sd,
            "se_paired": se,
            "t": t,
            "p": p,
            "wilcoxon_p": wil,
            "cohen_dz": (mean / sd) if sd > 0 else (
                math.inf if degenerate and mean > 0 else
                (-math.inf if degenerate and mean < 0 else float("nan"))),
            "zero_variance": degenerate,
            "degenerate_note": (
                "every pair moved by an identical amount; t and d_z diverge. "
                "Check the arms are genuinely independent before reporting "
                "this as a result." if degenerate else ""),
            "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"],
            # The noise floor the pairing removes. Report it next to the test.
            "sd_unpaired_treatment": float(treat.std(ddof=1)) if n > 1 else 0.0,
            "sd_unpaired_control": float(ctrl.std(ddof=1)) if n > 1 else 0.0,
            "deltas": deltas}


def holm_bonferroni(pvalues: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni step-down correction.

    The primary endpoint is pre-registered and is not part of this family; the
    caller passes only the secondary arms. Mixing them would spend the primary
    test's alpha on comparisons we declared secondary in advance.
    """
    items = [(k, v) for k, v in pvalues.items()
             if v is not None and math.isfinite(float(v))]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, float] = {k: float("nan") for k in pvalues}
    running = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * float(p))
        running = max(running, adj)   # step-down monotonicity
        out[k] = running
    return out


def minimum_detectable_effect(sigma_paired: float, n_seeds: int,
                              multiplier: float = 2.9) -> float:
    """MDE at 80% power, alpha 0.05, per the spec's Day-3 gate formula."""
    if n_seeds <= 0:
        raise ValueError("n_seeds must be positive")
    return multiplier * float(sigma_paired) / math.sqrt(n_seeds)


def go_no_go(mde: float) -> Dict[str, object]:
    """The pre-registered Day-3 decision rule. Returns the branch, in writing.

    Encoded as code rather than prose precisely because the spec warns that
    discovering on Day 8 that the design had no power is the most likely way
    the project fails.
    """
    if not math.isfinite(mde):
        return {"branch": "undetermined", "action": "MDE is not finite",
                "proceed": False}
    if mde < 0.002:
        return {"branch": "proceed",
                "action": "Proceed as specced. We can resolve PR #264's "
                          "measured effect size of 0.00076 nats.",
                "proceed": True}
    if mde < 0.008:
        return {"branch": "reduce",
                "action": "Drop the secondary arms. Run 3 arms x 12 seeds at "
                          "CFG_S. Keep the scale check and A1.",
                "proceed": True}
    return {"branch": "kill",
            "action": "KILL the training leg today. The design cannot decide "
                      "anything. Spend the remaining budget on Track A and "
                      "write the frozen-model paper.",
            "proceed": False}


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation. NaN when either input has no variance."""
    xs, ys = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    if xs.size < 3:
        return float("nan")
    if _sps is not None:
        r = _sps.spearmanr(xs, ys).correlation
        return float(r) if r is not None else float("nan")
    rx = np.argsort(np.argsort(xs)).astype(float)
    ry = np.argsort(np.argsort(ys)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def pearson(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    xs, ys = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    if xs.size < 3 or xs.std() == 0 or ys.std() == 0:
        return float("nan"), float("nan")
    if _sps is not None:
        res = _sps.pearsonr(xs, ys)
        return float(res[0]), float(res[1])
    return float(np.corrcoef(xs, ys)[0, 1]), float("nan")


def top_k_agreement(rank_a: Sequence[int], rank_b: Sequence[int],
                    k: int) -> float:
    sa, sb = set(rank_a[:k]), set(rank_b[:k])
    return len(sa & sb) / float(k) if k else float("nan")


def cluster_bootstrap_ci(values: Sequence[float], clusters: Sequence[object],
                         confidence: float = 0.95, n_boot: int = 5000,
                         seed: int = 0) -> Dict[str, float]:
    """Bootstrap over clusters, not rows.

    The CRPA analysis had 72 rows carrying only 24 independent values, so a
    row-level interval was far too narrow. Resampling whole clusters is the
    fix, and it is what the OVC bug-4 regression test checks.
    """
    groups: Dict[object, List[float]] = {}
    for v, c in zip(values, clusters):
        if math.isfinite(float(v)):
            groups.setdefault(c, []).append(float(v))
    keys = list(groups)
    if len(keys) < 2:
        return {"mean": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n_clusters": len(keys),
                "n_rows": sum(len(g) for g in groups.values())}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        pool: List[float] = []
        for i in pick:
            pool.extend(groups[keys[i]])
        means.append(float(np.mean(pool)))
    lo, hi = np.quantile(means, [(1 - confidence) / 2,
                                 1 - (1 - confidence) / 2])
    allv = [v for g in groups.values() for v in g]
    return {"mean": float(np.mean(allv)), "ci_low": float(lo),
            "ci_high": float(hi), "n_clusters": len(keys),
            "n_rows": len(allv)}


def split_half_reliability(half_a: Sequence[float],
                           half_b: Sequence[float]) -> Dict[str, float]:
    """Spearman between two disjoint-half estimates of the same quantity.

    This is Check 0's core number. If it is low, every downstream correlation
    attenuates toward zero by construction and a near-zero result means
    nothing.
    """
    r = spearman(half_a, half_b)
    return {"r": r, "n": int(min(len(half_a), len(half_b))),
            "resolvable": bool(math.isfinite(r) and r >= 0.3)}


def disattenuate(rho_observed: float, r_delta: float,
                 r_stat: float) -> float:
    """Correct a correlation for the unreliability of both measures.

    Returns NaN when either reliability is non-positive, because the
    correction is undefined there and a large number produced by dividing by a
    near-zero reliability is an artifact, not a finding.
    """
    if not all(math.isfinite(x) for x in (rho_observed, r_delta, r_stat)):
        return float("nan")
    if r_delta <= 0 or r_stat <= 0:
        return float("nan")
    return float(rho_observed / math.sqrt(r_delta * r_stat))


def threshold_sanity(threshold: float, signal: Sequence[float],
                     warn_ratio: float = 10.0) -> Dict[str, object]:
    """Is this threshold doing anything at all?

    ``eps = 0.03`` against a mean signal of 2e-6 is a ratio of 1.5e4: the
    filter passed everything and the whole gating mechanism was a no-op for an
    entire campaign. Compute this on the calibration split before the
    experiment runs and treat a large ratio as a stop condition.
    """
    xs = [abs(float(s)) for s in signal
          if s is not None and math.isfinite(float(s))]
    if not xs:
        return {"ratio": float("nan"), "mean_abs_signal": float("nan"),
                "threshold": float(threshold), "is_noop": True,
                "note": "no finite signal values"}
    mean_abs = float(np.mean(xs))
    ratio = float(threshold) / mean_abs if mean_abs > 0 else float("inf")
    noop = ratio > warn_ratio
    return {"ratio": ratio, "mean_abs_signal": mean_abs,
            "threshold": float(threshold), "is_noop": bool(noop),
            "n": len(xs),
            "note": ("threshold is {:.3g}x the mean signal; it is not doing "
                     "what you think".format(ratio) if noop
                     else "threshold is within {:.0f}x of the signal"
                          .format(warn_ratio))}


def pinned_at_extremum(value: float, minimum: float, maximum: float,
                       tol: float = 1e-4) -> Dict[str, object]:
    """Flag an auxiliary term sitting at its own analytic bound.

    A constant that equals ln(4) in every condition is an inactive mechanism,
    not a balanced one, and reporting it as a finding publishes a constant.
    """
    span = abs(maximum - minimum)
    scale = span if span > 0 else 1.0
    at_min = abs(value - minimum) <= tol * scale
    at_max = abs(value - maximum) <= tol * scale
    pinned = bool(at_min or at_max)
    return {"value": float(value), "analytic_min": float(minimum),
            "analytic_max": float(maximum), "at_min": bool(at_min),
            "at_max": bool(at_max), "pinned": pinned,
            "note": ("value sits at its analytic {}; this is not a result"
                     .format("minimum" if at_min else "maximum")
                     if pinned else "not pinned")}


def replicate_agreement(replicates: List[List[float]], eps: float,
                        top_k: int) -> Dict[str, object]:
    """Agreement between replicate estimates at one sample budget.

    Classification agreement is returned as NaN, never 1.0, when eps put every
    candidate in one class. Agreement over a one-class partition is undefined
    rather than perfect, and the 1.0 it otherwise prints reads as a stable
    estimator while measuring nothing.
    """
    n_rep = len(replicates)
    if n_rep < 2:
        return {"mean_spearman": float("nan"), "std_spearman": float("nan"),
                "mean_top_k_agreement": float("nan"),
                "mean_classification_agreement": float("nan"),
                "classification_degenerate_pairs": 0,
                "classification_is_degenerate": True,
                "n_replicates": n_rep, "n_comparisons": 0}

    spearmans, topks, classes = [], [], []
    degenerate = 0
    for a, b in itertools.combinations(range(n_rep), 2):
        da, db = replicates[a], replicates[b]
        spearmans.append(spearman(da, db))
        ra = sorted(range(len(da)), key=lambda i: da[i])
        rb = sorted(range(len(db)), key=lambda i: db[i])
        topks.append(top_k_agreement(ra, rb, top_k))
        la = [d <= eps for d in da]
        lb = [d <= eps for d in db]
        if not ((0 < sum(la) < len(la)) or (0 < sum(lb) < len(lb))):
            degenerate += 1
            continue
        classes.append(float(np.mean([x == y for x, y in zip(la, lb)])))

    finite = [s for s in spearmans if math.isfinite(s)]
    return {"mean_spearman": float(np.mean(finite)) if finite else float("nan"),
            "std_spearman": float(np.std(finite)) if len(finite) > 1 else 0.0,
            "mean_top_k_agreement": float(np.mean(topks)) if topks else float("nan"),
            "mean_classification_agreement":
                float(np.mean(classes)) if classes else float("nan"),
            "classification_degenerate_pairs": degenerate,
            "classification_is_degenerate": degenerate == len(spearmans),
            "n_replicates": n_rep, "n_comparisons": len(spearmans)}
