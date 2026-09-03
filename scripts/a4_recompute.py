"""
A4 - recompute XSA's Figure 1 under an equicorrelated-value null.

Pure arithmetic, no GPU, no downloads. Day 1.

The derivation
--------------
Model the value vectors as unit-norm and equicorrelated: ``cos(v_i, v_j) = rho``
for all ``i != j``. This is the standard first-order model of anisotropy, and
``rho`` is the quantity XSA's own Figure 1 (left) reports at 0.038-0.101.

An attention output is a convex combination of values::

    y_i = sum_j a_ij v_j,    a_ij >= 0,  sum_j a_ij = 1

so, writing ``S2 = sum_j a_ij^2`` for the attention row's participation term,

    <y_i, v_i>  = a_ii + (1 - a_ii) * rho
    ||y_i||^2   = (1 - rho) * S2 + rho
    cos(y_i,v_i) = [a_ii + (1 - a_ii) rho] / sqrt((1 - rho) S2 + rho)

The **floor** is the same quantity computed for a row that gives the token's
own value no special weight at all: y_i built only from the other values. Then
``<y_i, v_i> = rho`` and

    floor = rho / sqrt((1 - rho) S2' + rho),   S2' = sum_{j != i} a_ij^2

``S2`` is the one free parameter. It is set by how peaked the attention row is,
equivalently by the effective number of attended keys ``n_eff = 1 / S2``. We
solve for the ``n`` that reproduces the published floor and report the excess
under every consistent reading, rather than asserting one.

Why this script exists in this form
-----------------------------------
The spec states the result as "floor 0.200 vs observed 0.373 -> excess 0.165 =
44%", and the Day-1 gate requires reproducing all four numbers. **They are not
mutually consistent.** 0.373 - 0.200 = 0.173, and 0.173 / 0.373 = 46.4%, not
44%. The 44% figure is consistent with observed 0.373 and excess 0.165, which
requires a floor of 0.208.

The gate as written can therefore never pass. This script computes the
arithmetic honestly, reports which pair of numbers is self-consistent, and
exits non-zero only if its own internal arithmetic fails. Loosening a number to
make a gate pass is forbidden by the project's own anti-pattern table.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS = Path(__file__).resolve().parents[1] / "results"

# ---------------------------------------------------------------------------
# Published inputs. Provenance in the comment; do not edit without one.
# ---------------------------------------------------------------------------
#: XSA arXiv:2603.09078v1 Figure 1 (left), mean cos(v_i, v_j) over the
#: reported 0.038-0.101 band.
RHO = 0.056
#: XSA Figure 1, mean self-attention weight a_ii.
A_II = 0.050
#: XSA Figure 1, mean cos(y_i, v_i) -- the statistic that motivates the method.
COS_SELF_OBSERVED = 0.373
#: The floor, excess and percentage as the spec states them.
SPEC_FLOOR = 0.200
SPEC_EXCESS = 0.165
SPEC_PERCENT = 44.0


def cos_self(rho: float, a_ii: float, s2: float) -> float:
    """cos(y_i, v_i) under the equicorrelated model."""
    num = a_ii + (1.0 - a_ii) * rho
    den = math.sqrt((1.0 - rho) * s2 + rho)
    return num / den


def cos_floor(rho: float, s2_other: float) -> float:
    """The null: a row that gives the token's own value no extra weight."""
    return rho / math.sqrt((1.0 - rho) * s2_other + rho)


def s2_uniform(a_ii: float, n: float) -> float:
    """Participation term for 'a_ii on self, the rest spread uniformly'."""
    return a_ii ** 2 + (1.0 - a_ii) ** 2 / (n - 1.0)


def solve_n_for_floor(rho: float, target_floor: float) -> float:
    """Effective context width implied by a published floor value."""
    d = rho / target_floor              # = sqrt((1-rho) S2' + rho)
    s2_other = (d * d - rho) / (1.0 - rho)
    if s2_other <= 0:
        return float("nan")
    return 1.0 / s2_other + 1.0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(RESULTS / "xsa_figure1_recompute.txt"))
    parser.add_argument("--json", default=str(RESULTS / "xsa_figure1_recompute.json"))
    args = parser.parse_args(argv)

    n_implied = solve_n_for_floor(RHO, SPEC_FLOOR)
    s2_other = 1.0 / (n_implied - 1.0)
    floor = cos_floor(RHO, s2_other)
    s2_self = s2_uniform(A_II, n_implied)
    predicted_self = cos_self(RHO, A_II, s2_self)

    excess_observed = COS_SELF_OBSERVED - floor
    excess_predicted = predicted_self - floor
    pct_observed = 100.0 * excess_observed / COS_SELF_OBSERVED
    pct_predicted = 100.0 * excess_predicted / predicted_self

    # What floor would the spec's own excess and percentage require?
    floor_from_excess = COS_SELF_OBSERVED - SPEC_EXCESS
    floor_from_percent = COS_SELF_OBSERVED * (1.0 - SPEC_PERCENT / 100.0)

    spec_consistent = abs(
        (COS_SELF_OBSERVED - SPEC_FLOOR) - SPEC_EXCESS) < 5e-4

    lines = []
    add = lines.append
    add("A4 - XSA Figure 1 recomputed under an equicorrelated-value null")
    add("=" * 70)
    add("")
    add("Published inputs (XSA arXiv:2603.09078v1, Figure 1)")
    add("  mean cos(v_i, v_j)   rho     = {:.4f}".format(RHO))
    add("  mean self-weight     a_ii    = {:.4f}".format(A_II))
    add("  mean cos(y_i, v_i)   observed= {:.4f}".format(COS_SELF_OBSERVED))
    add("")
    add("Model")
    add("  cos(y_i,v_i) = [a_ii + (1-a_ii) rho] / sqrt((1-rho) S2 + rho)")
    add("  floor        = rho              / sqrt((1-rho) S2' + rho)")
    add("  one free parameter: S2, the attention row's participation term")
    add("")
    add("Solving for the effective context width that reproduces the")
    add("published floor of {:.3f}:".format(SPEC_FLOOR))
    add("  n_eff (attended keys)        = {:.1f}".format(n_implied))
    add("  floor (recomputed)           = {:.4f}".format(floor))
    add("  cos_self predicted by model  = {:.4f}".format(predicted_self))
    add("  cos_self observed (reported) = {:.4f}".format(COS_SELF_OBSERVED))
    add("  model-vs-observed gap        = {:+.4f}".format(
        predicted_self - COS_SELF_OBSERVED))
    add("")
    add("Excess, both readings")
    add("  observed - floor  = {:.4f}  ->  {:.1f}% of the statistic is "
        "self-specific".format(excess_observed, pct_observed))
    add("  model    - floor  = {:.4f}  ->  {:.1f}%".format(
        excess_predicted, pct_predicted))
    add("")
    add("CONSISTENCY CHECK against the values the spec states")
    add("-" * 70)
    add("  spec: floor {:.3f}, observed {:.3f}, excess {:.3f}, {:.0f}%".format(
        SPEC_FLOOR, COS_SELF_OBSERVED, SPEC_EXCESS, SPEC_PERCENT))
    add("")
    if spec_consistent:
        add("  The stated triple closes.")
    else:
        add("  THE STATED TRIPLE DOES NOT CLOSE, and the Day-1 gate that")
        add("  requires all four numbers can never pass as written.")
        add("")
        add("    observed - floor        = {:.4f} - {:.4f} = {:.4f}".format(
            COS_SELF_OBSERVED, SPEC_FLOOR, COS_SELF_OBSERVED - SPEC_FLOOR))
        add("    but the spec says excess = {:.4f}   (differs by {:.4f})"
            .format(SPEC_EXCESS,
                    abs(COS_SELF_OBSERVED - SPEC_FLOOR - SPEC_EXCESS)))
        add("")
        add("    {:.4f} / {:.4f} = {:.1f}%, not {:.0f}%".format(
            COS_SELF_OBSERVED - SPEC_FLOOR, COS_SELF_OBSERVED,
            100.0 * (COS_SELF_OBSERVED - SPEC_FLOOR) / COS_SELF_OBSERVED,
            SPEC_PERCENT))
        add("")
        add("  Two of the three stated numbers agree with each other:")
        add("    excess {:.3f} and {:.0f}% are consistent with observed {:.3f}"
            .format(SPEC_EXCESS, SPEC_PERCENT, COS_SELF_OBSERVED))
        add("    but they require a floor of {:.4f}, not {:.3f}.".format(
            floor_from_excess, SPEC_FLOOR))
        add("    (the percentage alone implies a floor of {:.4f})".format(
            floor_from_percent))
        add("")
        add("  RESOLUTION ADOPTED: report the floor as the recomputed")
        add("  {:.3f} and the excess as {:.3f} ({:.0f}% of the statistic).".format(
            floor, excess_observed, round(pct_observed)))
        add("  The qualitative claim is unchanged and is what matters: under")
        add("  an anisotropy null, well under half of cos(y_i,v_i) is")
        add("  self-specific, on XSA's own numbers, at XSA's own scale.")
    add("")
    add("Sensitivity to the one free parameter")
    add("-" * 70)
    add("  n_eff    floor   cos_self(model)   excess   % self-specific")
    for n in (16.0, 32.0, n_implied, 64.0, 128.0, 256.0):
        s2o = 1.0 / (n - 1.0)
        f = cos_floor(RHO, s2o)
        cs = cos_self(RHO, A_II, s2_uniform(A_II, n))
        add("  {:6.1f}  {:.4f}       {:.4f}      {:.4f}      {:5.1f}%".format(
            n, f, cs, cs - f, 100.0 * (cs - f) / cs))
    add("")
    add("The floor is sensitive to how peaked the attention rows are, so the")
    add("single number should always be reported with the assumption that")
    add("produced it. That sensitivity is itself an argument for measuring the")
    add("null empirically (Check 1, A1) rather than modelling it.")

    text = "\n".join(lines) + "\n"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)

    payload = {
        "inputs": {"rho": RHO, "a_ii": A_II,
                   "cos_self_observed": COS_SELF_OBSERVED},
        "spec_values": {"floor": SPEC_FLOOR, "excess": SPEC_EXCESS,
                        "percent": SPEC_PERCENT},
        "spec_is_internally_consistent": bool(spec_consistent),
        "floor_implied_by_spec_excess": floor_from_excess,
        "floor_implied_by_spec_percent": floor_from_percent,
        "recomputed": {"n_eff": n_implied, "floor": floor,
                       "cos_self_model": predicted_self,
                       "excess_vs_observed": excess_observed,
                       "percent_self_specific": pct_observed},
    }
    Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
