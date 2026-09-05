"""
xsac.figures - the five publication figures.

Rules this module enforces, each from the spec's Day-9 gate:

* Every figure writes its **source data** as a CSV beside the image, so every
  plot is reproducible from its own file and no number can appear in a figure
  that exists nowhere else.
* Both ``.png`` (300 dpi) and ``.pdf`` (vector, for LaTeX).
* A figure whose inputs are missing is **skipped with a message**. It is never
  drawn from placeholder or synthesised points: an absent experiment must look
  absent.
* No series is distinguished by red versus green alone. Every series carries a
  distinct marker and linestyle as well as a colour, so the figures survive
  greyscale conversion and colour-vision deficiency.
* Every caption states ``n``. Every axis states units.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .stats import minimum_detectable_effect  # noqa: E402

#: Colour-vision-safe categorical slots. Never red/green as the only contrast.
#: Chosen for relative luminance, not only for hue. The previous palette put
#: baseline (#4d4d4d) and diagmask (#4a3aa7) 0.001 apart in luminance: two
#: identical lines in greyscale or print, which spec section 11 forbids. Every
#: pair here is at least 0.125 apart, and scripts/check_figures.py enforces it.
SERIES = {
    "baseline": "#3c3c3c",
    "xsa": "#4473b9",
    "random": "#f59b6e",
    "meanval": "#3cea38",
    "diagmask": "#d266f8",
}
#: Redundant encoding: marker and linestyle carry identity too.
MARKERS = {"baseline": "o", "xsa": "s", "random": "D",
           "meanval": "^", "diagmask": "v"}
LINESTYLES = {"baseline": "-", "xsa": "--", "random": "-.",
              "meanval": ":", "diagmask": (0, (3, 1, 1, 1))}

SURFACE = "#ffffff"
INK = "#111111"
INK_SECONDARY = "#555555"
GRID = "#dddddd"

#: XSA's own reported gap and PR #264's measured gap, for reference lines.
XSA_CLAIMED_DELTA = -0.017
PR264_MEASURED_DELTA = -0.00076
#: The parameter range XSA actually trained at, shaded in Figure 3.
XSA_TESTED_RANGE = (0.7e9, 2.7e9)


class FigureSkipped(Exception):
    """Raised when a figure's inputs are absent. Never draw a placeholder."""


def _style(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, fontsize=9.5, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=9.5, color=INK_SECONDARY)
    ax.set_title(title, fontsize=11, color=INK, pad=12, loc="left")
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8.5, length=0)
    ax.set_facecolor(SURFACE)


def _save(fig, out_dir: Path, stem: str,
          rows: Sequence[Dict[str, Any]]) -> List[Path]:
    """Write png, pdf and the figure's own source CSV."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / "{}.{}".format(stem, ext)
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor=SURFACE)
        paths.append(p)
    plt.close(fig)
    if rows:
        data = out_dir / "{}_data.csv".format(stem)
        fields: List[str] = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
        with data.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        paths.append(data)
    return paths


def read_csv(path: Path) -> List[Dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _num(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# Figure 1 - learned gates per layer and head
# ---------------------------------------------------------------------------

#: Where a factorial figure looks for its data, in priority order: the primary
#: endpoint first, the underpowered pilot only as a fallback. A figure built
#: from the pilot says so in its title, because a reader who cannot tell which
#: budget a panel came from will assume the primary one.
def factorial_source(results: Path, stem: str):
    """Return (rows, label, is_pilot) for the best available factorial data."""
    for name, label, pilot in (
            ("{}_s.csv".format(stem), "CFG_S", False),
            ("{}_m.csv".format(stem), "CFG_M", False),
            ("{}_s_pilot_5e7.csv".format(stem),
             "CFG_S underpowered pilot, 5e7 tokens/run", True)):
        rows = read_csv(Path(results) / name)
        if rows:
            return rows, label, pilot
    return [], "", False


def fig1_gates(results: Path, out_dir: Path) -> List[Path]:
    """If ``random`` learns positive gates comparable to ``xsa``, this panel
    IS the result: a matched arbitrary direction was worth removing too."""
    import json

    rows, source_label, from_pilot = factorial_source(results, "factorial")
    if not rows:
        raise FigureSkipped(
            "no factorial_*.csv in {}. Run: python scripts/run_factorial.py"
            .format(results))

    gated = [r for r in rows if r.get("learned_alpha")
             and r.get("status") in ("completed", "smoke")
             and r.get("arm") in ("xsa", "random", "meanval", "diagmask")]
    if not gated:
        raise FigureSkipped(
            "no runs with learned gates in the {} factorial"
            .format(source_label))

    arms = [a for a in ("xsa", "random", "meanval", "diagmask")
            if any(r["arm"] == a for r in gated)]
    fig, axes = plt.subplots(1, len(arms), figsize=(3.6 * len(arms), 3.6),
                             squeeze=False, sharey=True)
    data_rows: List[Dict[str, Any]] = []
    for ax, arm in zip(axes[0], arms):
        per_layer: Dict[int, List[float]] = {}
        for r in gated:
            if r["arm"] != arm:
                continue
            try:
                table = json.loads(r["learned_alpha"])
            except Exception:
                continue
            for key, vals in table.items():
                li = int(key.split("_")[1])
                per_layer.setdefault(li, []).extend(float(v) for v in vals)
        if not per_layer:
            continue
        layers = sorted(per_layer)
        means = [float(np.mean(per_layer[li])) for li in layers]
        for li in layers:
            for v in per_layer[li]:
                ax.scatter(li, v, s=14, c=SERIES[arm], alpha=0.35,
                           edgecolors="none", zorder=2)
                data_rows.append({"arm": arm, "layer": li, "tanh_alpha": v})
        ax.plot(layers, means, color=SERIES[arm], linewidth=2,
                marker=MARKERS[arm], linestyle=LINESTYLES[arm], zorder=3,
                label="mean")
        ax.axhline(0.0, color=INK_SECONDARY, linewidth=1,
                   linestyle=(0, (4, 3)), zorder=1)
        _style(ax, "layer", "tanh(alpha)  (dimensionless)", arm)
        ax.legend(frameon=False, fontsize=8)
    n = len({(r["arm"], r["seed"]) for r in gated})
    # The budget goes in the title, not a caption someone may not read.
    fig.suptitle("Figure 1  Learned gate per layer and head  (n = {} runs, {})"
                 .format(n, source_label), fontsize=11, x=0.02, ha="left")
    if from_pilot:
        fig.text(0.02, -0.02, "UNDERPOWERED PILOT: 5e7 tokens per run, "
                 "outside the pre-registered [3.5e8, 6e8] band. Not the "
                 "primary endpoint.", fontsize=8.5, color=SERIES["random"])
    return _save(fig, out_dir, "fig1_gates", data_rows)


# ---------------------------------------------------------------------------
# Figure 2 - paired delta loss per arm
# ---------------------------------------------------------------------------

def fig2_paired_delta(results: Path, out_dir: Path) -> List[Path]:
    frames = []
    for size in ("s", "m"):
        rows = read_csv(Path(results) / "paired_tests_{}.csv".format(size))
        if rows:
            frames.append((size.upper(), rows))
    from_pilot = False
    if not frames:
        rows, label, from_pilot = factorial_source(results, "paired_tests")
        if rows:
            frames.append((label, rows))
    if not frames:
        raise FigureSkipped(
            "no paired_tests_*.csv in {}, including the pilot. Run: python "
            "scripts/run_factorial.py".format(results))

    fig, axes = plt.subplots(1, len(frames), figsize=(5.0 * len(frames), 4.0),
                             squeeze=False, sharey=True)
    data_rows: List[Dict[str, Any]] = []
    for ax, (size, rows) in zip(axes[0], frames):
        arms = [r["arm"] for r in rows]
        xs = np.arange(len(arms))
        for i, r in enumerate(rows):
            arm = r["arm"]
            mean = _num(r["mean_delta"])
            lo, hi = _num(r["ci_low"]), _num(r["ci_high"])
            ax.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]],
                        fmt=MARKERS.get(arm, "o"), color=SERIES.get(arm, "#333"),
                        capsize=4, markersize=8, linewidth=1.6, zorder=3)
            data_rows.append({"size": size, **r})
        ax.axhline(0.0, color=INK_SECONDARY, linewidth=1, zorder=1)
        ax.axhline(XSA_CLAIMED_DELTA, color="#888", linewidth=1,
                   linestyle="--", zorder=1)
        ax.annotate("XSA claimed {:.3f}".format(XSA_CLAIMED_DELTA),
                    xy=(0, XSA_CLAIMED_DELTA), fontsize=7.5,
                    color=INK_SECONDARY, va="bottom")
        ax.axhline(PR264_MEASURED_DELTA, color="#888", linewidth=1,
                   linestyle=":", zorder=1)
        ax.annotate("PR #264 measured {:.5f}".format(PR264_MEASURED_DELTA),
                    xy=(0, PR264_MEASURED_DELTA), fontsize=7.5,
                    color=INK_SECONDARY, va="top")
        ax.set_xticks(xs)
        ax.set_xticklabels(arms, fontsize=8.5)
        _style(ax, "arm", "paired delta val loss vs baseline (nats)",
               size if size.startswith("CFG_")
               else "CFG_{}".format(size))
    ns = ",".join(str(r.get("n_seeds", "?")) for _, rows in frames
                  for r in rows[:1])
    fig.suptitle("Figure 2  Paired delta loss, 95% CI  (n = {} seeds)"
                 .format(ns), fontsize=11, x=0.02, ha="left")
    if from_pilot:
        fig.text(0.02, -0.02, "UNDERPOWERED PILOT: 5e7 tokens per run, "
                 "outside the pre-registered [3.5e8, 6e8] band. Not the "
                 "primary endpoint.", fontsize=8.5, color=SERIES["random"])
    return _save(fig, out_dir, "fig2_paired_delta", data_rows)


# ---------------------------------------------------------------------------
# Figure 3 - Check 1 across the scale ladder
# ---------------------------------------------------------------------------

#: Parameter counts for the ladder, used only to place points on a log axis.
MODEL_PARAMS = {
    "gpt2": 124e6, "gpt2-medium": 355e6, "gpt2-large": 774e6,
    "gpt2-xl": 1.5e9,
    "EleutherAI/pythia-160m": 160e6, "EleutherAI/pythia-410m": 410e6,
    "EleutherAI/pythia-1.4b": 1.4e9, "EleutherAI/pythia-2.8b": 2.8e9,
    "EleutherAI/pythia-6.9b": 6.9e9,
    "Qwen/Qwen3-1.7B": 1.7e9, "meta-llama/Llama-3.2-1B": 1.0e9,
}


def fig3_ladder(results: Path, out_dir: Path) -> List[Path]:
    """The figure that answers the scale objection.

    Machina & Mercer (NAACL 2024) report that large Pythia models are
    isotropic. If the confound vanishes above ~1B, the paper's framing has to
    change, and this is the only experiment that says so directly rather than
    by extrapolation.
    """
    rows = read_csv(Path(results) / "ladder.csv")
    if not rows:
        raise FigureSkipped(
            "missing ladder.csv. Run: python scripts/run_frozen.py --ladder")

    by_model: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)

    points = []
    for model, rs in by_model.items():
        params = MODEL_PARAMS.get(model)
        if params is None:
            continue
        cs = float(np.mean([_num(r["cos_self"]) for r in rs]))
        cn = float(np.mean([_num(r["cos_null"]) for r in rs]))
        points.append({"model": model, "params": params, "cos_self": cs,
                       "cos_null": cn, "excess": cs - cn, "n_heads": len(rs),
                       "self_specific_fraction": (cs - cn) / cs if cs else
                       float("nan")})
    if not points:
        raise FigureSkipped(
            "no ladder rows matched a known parameter count; add the model to "
            "MODEL_PARAMS rather than guessing its size")
    points.sort(key=lambda p: p["params"])

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.axvspan(*XSA_TESTED_RANGE, color="#cfd8e8", alpha=0.45, zorder=0)
    ax.annotate("XSA's tested range\n0.7B - 2.7B",
                xy=(math.sqrt(XSA_TESTED_RANGE[0] * XSA_TESTED_RANGE[1]), 0.02),
                fontsize=7.5, color=INK_SECONDARY, ha="center")
    xs = [p["params"] for p in points]
    for key, colour, marker, ls, label in (
            ("cos_self", SERIES["xsa"], "s", "--", "cos(y_i, v_i)  observed"),
            ("cos_null", SERIES["random"], "D", "-.",
             "cos(y_i, v_j)  anisotropy null"),
            ("excess", SERIES["meanval"], "^", "-", "excess (self-specific)")):
        ax.plot(xs, [p[key] for p in points], color=colour, marker=marker,
                linestyle=ls, linewidth=1.8, markersize=7, label=label,
                zorder=3)
    ax.set_xscale("log")
    _style(ax, "parameters (log scale)", "cosine similarity (dimensionless)",
           "Figure 3  Check 1 across the scale ladder")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    total_heads = sum(p["n_heads"] for p in points)
    ax.annotate("n = {} models, {} heads".format(len(points), total_heads),
                xy=(0.99, 0.02), xycoords="axes fraction", ha="right",
                fontsize=8, color=INK_SECONDARY)
    return _save(fig, out_dir, "fig3_ladder", points)


# ---------------------------------------------------------------------------
# Figure 4 - generality across methods
# ---------------------------------------------------------------------------

def fig4_generality(results: Path, out_dir: Path) -> List[Path]:
    rows = read_csv(Path(results) / "generality.csv")
    if not rows:
        raise FigureSkipped(
            "missing generality.csv (A6). This figure requires real data and "
            "renders nothing without it.")
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    methods = [r["method"] for r in rows]
    observed = [_num(r["observed"]) for r in rows]
    null = [_num(r["null"]) for r in rows]
    xs = np.arange(len(methods))
    # These statistics are not commensurate: attention mass is a probability
    # in [0, 1] (0.40) and the activation ratio is an unbounded norm (12.87).
    # On one linear axis the attention-sink bars are flattened to nothing, so
    # the panel silently hides one of the two results it exists to show.
    # A log axis keeps both readable; the y label says the units differ.
    ax.bar(xs - 0.2, observed, width=0.36, color=SERIES["xsa"],
           label="observed statistic", zorder=2)
    ax.bar(xs + 0.2, null, width=0.36, color=SERIES["random"],
           hatch="///", edgecolor=SURFACE, label="null", zorder=2)
    if max(observed + null) / max(min(x for x in observed + null
                                      if x > 0), 1e-9) > 50:
        ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(methods, fontsize=8, rotation=15, ha="right")
    _style(ax, "method",
           "statistic value, log scale (units differ per method)",
           "Figure 4  Check 1 applied to five methods")
    ax.legend(frameon=False, fontsize=8.5)
    ax.annotate("n = {} methods".format(len(rows)), xy=(0.99, 0.95),
                xycoords="axes fraction", ha="right", fontsize=8,
                color=INK_SECONDARY)
    return _save(fig, out_dir, "fig4_generality", rows)


# ---------------------------------------------------------------------------
# Figure 5 - GQA within-group vs across-group
# ---------------------------------------------------------------------------

def fig5_gqa(results: Path, out_dir: Path) -> List[Path]:
    rows = [r for r in read_csv(Path(results) / "gqa.csv")
            if str(r.get("is_gqa", "")).lower() in ("true", "1")]
    if not rows:
        raise FigureSkipped(
            "no GQA models in gqa.csv. Run: python scripts/run_frozen.py --gqa")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    models = [r["model"] for r in rows]
    xs = np.arange(len(models))
    within = [_num(r["within_group_excess"]) for r in rows]
    across = [_num(r["across_group_excess"]) for r in rows]
    ax.bar(xs - 0.2, within, width=0.36, color=SERIES["meanval"],
           label="within KV group", zorder=2)
    ax.bar(xs + 0.2, across, width=0.36, color=SERIES["diagmask"],
           hatch="\\\\\\", edgecolor=SURFACE, label="across groups", zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(models, fontsize=8, rotation=12, ha="right")
    _style(ax, "model", "excess = cos_self - cos_null (dimensionless)",
           "Figure 5  GQA: within-group vs across-group")
    ax.legend(frameon=False, fontsize=8.5)
    ax.annotate("n = {} models".format(len(rows)), xy=(0.99, 0.95),
                xycoords="axes fraction", ha="right", fontsize=8,
                color=INK_SECONDARY)
    return _save(fig, out_dir, "fig5_gqa", rows)


# ---------------------------------------------------------------------------
# Figure 6 - A2: does the motivating statistic predict its own intervention?
# ---------------------------------------------------------------------------

def fig6_a2_scatter(results: Path, out_dir: Path) -> List[Path]:
    """Per-head statistic against the measured effect of removing it.

    A single rho hides whether a relationship is carried by a handful of
    heads or spread across all of them, and it hides the ceiling that
    split-half unreliability puts on how large that rho could possibly be.
    Both are annotated on each panel.
    """
    heads = read_csv(Path(results) / "a2_per_head.csv")
    corr = read_csv(Path(results) / "a2_correlations.csv")
    if not heads:
        raise FigureSkipped(
            "missing a2_per_head.csv. Run: python scripts/run_reliability.py")
    if not corr:
        raise FigureSkipped("missing a2_correlations.csv")

    models: List[str] = []
    for r in heads:
        if r["model"] not in models:
            models.append(r["model"])

    stats = {}
    for r in corr:
        stats[(r["model"], r["statistic"])] = r

    fig, axes = plt.subplots(1, len(models), figsize=(4.8 * len(models), 4.2),
                             squeeze=False)
    fig.subplots_adjust(wspace=0.32)
    rows_out = []
    for panel, (ax, model) in enumerate(zip(axes[0], models)):
        mine = [r for r in heads if r["model"] == model]
        x = [_num(r["cos_self"]) for r in mine]
        y = [_num(r["delta_pooled"]) for r in mine]
        ax.axhline(0.0, color=INK_SECONDARY, linewidth=1, zorder=1)
        ax.scatter(x, y, s=18, facecolor=SERIES["xsa"], edgecolor=SURFACE,
                   linewidth=0.4, alpha=0.85, zorder=3)
        # Least-squares line, drawn only to guide the eye. The reported
        # statistic is Spearman, which this line does not represent.
        if len(x) >= 3 and np.std(x) > 0:
            b, a = np.polyfit(np.asarray(x), np.asarray(y), 1)
            xs = np.linspace(min(x), max(x), 32)
            ax.plot(xs, b * xs + a, color=SERIES["random"], linewidth=1.4,
                    linestyle="--", zorder=4)
        c = stats.get((model, "cos_self"), {})
        note = "\n".join([
            "raw rho        {:+.3f}".format(_num(c.get("rho_raw"))),
            "disattenuated  {:+.3f}".format(_num(c.get("rho_disattenuated"))),
            "r_delta        {:+.3f}".format(_num(c.get("r_delta"))),
            "ceiling         {:.3f}".format(_num(c.get("ceiling"))),
            "n heads          {}".format(len(mine)),
        ])
        # Boxed, so it stays readable where it sits over the point cloud.
        ax.annotate(note, xy=(0.03, 0.97), xycoords="axes fraction",
                    va="top", ha="left", fontsize=7.6, family="monospace",
                    color=INK_SECONDARY,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor=SURFACE,
                              edgecolor=GRID, linewidth=0.7, alpha=0.92))
        # Only the leftmost panel carries the y label: repeating it puts text
        # from one panel directly against the next panel's data.
        ylabel = ("delta loss when the head's self-value is removed (nats)"
                  if panel == 0 else "")
        _style(ax, "cos(y_i, v_i) per head (dimensionless)", ylabel,
               model.split("/")[-1])
        rows_out.extend(mine)

    fig.suptitle("Figure 6  A2: motivating statistic vs measured effect",
                 fontsize=11, y=1.02)
    return _save(fig, out_dir, "fig6_a2_scatter", rows_out)


# ---------------------------------------------------------------------------
# Figure 7 - power: what the design could have resolved
# ---------------------------------------------------------------------------

#: The effect size XSA's own independent replication (PR #264) reports.
PR264_REFERENCE = -0.00076


def fig7_power(results: Path, out_dir: Path) -> List[Path]:
    """Minimum detectable effect against seed count, one curve per budget.

    Turns the power limitation into a quantity rather than an apology: it
    says how many seeds each token budget would have needed before the
    reported effect came within reach.
    """
    curves = []
    for path in sorted(Path(results).glob("paired_tests_*.csv")):
        if "smoke" in path.name:
            continue
        rows = [r for r in read_csv(path) if r.get("sd_paired")]
        if not rows:
            continue
        primary = [r for r in rows if r.get("arm") == "random"] or rows
        sigma = _num(primary[0]["sd_paired"])
        # The paired-test file records the test, not the budget it was run
        # at. Read the budget from the factorial rows that produced it, and
        # use tokens_seen rather than a configured figure: what a curve is
        # labelled with should be what the runs actually consumed.
        tokens = _num(primary[0].get("tokens_per_run"))
        if not np.isfinite(tokens) or not tokens:
            sibling = path.parent / path.name.replace("paired_tests_",
                                                      "factorial_")
            seen = {_num(r.get("tokens_seen"))
                    for r in read_csv(sibling)} if sibling.exists() else set()
            seen = {v for v in seen if np.isfinite(v) and v}
            # Only label a budget the whole file agrees on. Mixed budgets are
            # exactly what the homogeneity guard exists to catch.
            tokens = seen.pop() if len(seen) == 1 else float("nan")
        if not np.isfinite(sigma) or sigma <= 0:
            continue
        curves.append((path.name, sigma, tokens,
                       int(_num(primary[0].get("n_seeds", 0)))))
    if not curves:
        raise FigureSkipped(
            "no paired_tests_*.csv with a paired sd. Run the factorial.")

    seeds = np.arange(2, 65)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    rows_out = []
    for i, (name, sigma, tokens, n_used) in enumerate(sorted(
            curves, key=lambda c: c[1])):
        mde = [minimum_detectable_effect(sigma, int(n)) for n in seeds]
        label = ("{:.3g} tokens/run".format(tokens) if np.isfinite(tokens)
                 and tokens else name)
        arm = ("xsa", "random", "meanval", "diagmask", "baseline")[i % 5]
        ax.plot(seeds, mde, color=SERIES[arm], linewidth=1.8,
                linestyle=LINESTYLES[arm], label=label, zorder=3)
        if n_used:
            ax.plot([n_used], [minimum_detectable_effect(sigma, n_used)],
                    marker=MARKERS[arm], color=SERIES[arm], markersize=8,
                    markeredgecolor=SURFACE, zorder=5)
        for n in seeds:
            rows_out.append({"source": name, "tokens_per_run": tokens,
                             "sd_paired": sigma, "n_seeds": int(n),
                             "mde_nats": minimum_detectable_effect(sigma,
                                                                   int(n))})

    ax.axhline(abs(PR264_REFERENCE), color=INK, linewidth=1.3,
               linestyle=(0, (5, 3)), zorder=4,
               label="effect PR #264 reports ({:.5f})".format(
                   abs(PR264_REFERENCE)))
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32, 64])
    ax.set_xticklabels(["2", "4", "8", "16", "32", "64"])
    ax.minorticks_off()
    _style(ax, "seeds per arm", "minimum detectable effect (nats)",
           "Figure 7  Power: MDE vs seeds, by token budget")
    ax.legend(frameon=False, fontsize=8.2, loc="upper right")
    ax.annotate("markers show the seed count actually run",
                xy=(0.02, 0.04), xycoords="axes fraction", fontsize=8,
                color=INK_SECONDARY)
    return _save(fig, out_dir, "fig7_power", rows_out)


ALL_FIGURES = {
    "fig1_gates": fig1_gates,
    "fig2_paired_delta": fig2_paired_delta,
    "fig3_ladder": fig3_ladder,
    "fig4_generality": fig4_generality,
    "fig5_gqa": fig5_gqa,
    "fig6_a2_scatter": fig6_a2_scatter,
    "fig7_power": fig7_power,
}
