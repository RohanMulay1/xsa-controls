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
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

#: Colour-vision-safe categorical slots. Never red/green as the only contrast.
SERIES = {
    "baseline": "#4d4d4d",
    "xsa": "#2a78d6",
    "random": "#eb6834",
    "meanval": "#1baf7a",
    "diagmask": "#4a3aa7",
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

def fig1_gates(results: Path, out_dir: Path) -> List[Path]:
    """If ``random`` learns positive gates comparable to ``xsa``, this panel
    IS the result: a matched arbitrary direction was worth removing too."""
    import json

    src = Path(results) / "factorial_s.csv"
    rows = read_csv(src)
    if not rows:
        raise FigureSkipped(
            "missing {}. Run: python scripts/run_factorial.py".format(src))

    gated = [r for r in rows if r.get("learned_alpha")
             and r.get("status") in ("completed", "smoke")
             and r.get("arm") in ("xsa", "random", "meanval", "diagmask")]
    if not gated:
        raise FigureSkipped("no runs with learned gates in {}".format(src))

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
    fig.suptitle("Figure 1  Learned gate per layer and head  (n = {} runs)"
                 .format(n), fontsize=11, x=0.02, ha="left")
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
    if not frames:
        raise FigureSkipped(
            "missing paired_tests_*.csv. Run: python scripts/run_factorial.py")

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
               "CFG_{}".format(size))
    ns = ",".join(str(r.get("n_seeds", "?")) for _, rows in frames
                  for r in rows[:1])
    fig.suptitle("Figure 2  Paired delta loss, 95% CI  (n = {} seeds)"
                 .format(ns), fontsize=11, x=0.02, ha="left")
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
    ax.bar(xs - 0.2, observed, width=0.36, color=SERIES["xsa"],
           label="observed statistic", zorder=2)
    ax.bar(xs + 0.2, null, width=0.36, color=SERIES["random"],
           hatch="///", edgecolor=SURFACE, label="null", zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(methods, fontsize=8, rotation=15, ha="right")
    _style(ax, "method", "statistic value (dimensionless)",
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


ALL_FIGURES = {
    "fig1_gates": fig1_gates,
    "fig2_paired_delta": fig2_paired_delta,
    "fig3_ladder": fig3_ladder,
    "fig4_generality": fig4_generality,
    "fig5_gqa": fig5_gqa,
}
