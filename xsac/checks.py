"""
Sanity checks for attention surgery.

This is the module the paper ships. If you have never read the paper, this
docstring is enough to use it.

The pattern the checks address
------------------------------
A recurring move in transformer design:

1. Measure a statistic on internal representations.
2. Observe it is large.
3. Propose an architectural change that removes it.
4. Report that the change helps.

Steps 2 and 4 each need a control and almost nobody runs either. Step 4 also
needs a prior question answered: is the quantity you are measuring resolvable
at all? These three functions are those three questions.

    check_resolvability(...)   Check 0. Is my measurement above its own noise
                               floor, and does it converge as I add samples?
                               Run this FIRST. If it fails, every correlation
                               you compute afterwards attenuates toward zero by
                               construction and a null result means nothing.

    check_null(...)            Check 1. Cosine similarities on transformer
                               internals are inflated by anisotropy. An
                               attention output y_i is a convex combination of
                               value vectors, so it resembles every v_j, not
                               only its own. Compare against cos(y_i, v_j),
                               j != i. It is essentially never reported.

    check_matched(...)         Check 2. If removing direction d helps, removing
                               a matched arbitrary direction should not. If it
                               does, you have a regulariser, not an insight.

Typical use, in order::

    from xsac.checks import check_resolvability, check_null, check_matched

    # Check 0 first. Nothing downstream is interpretable until this passes.
    r = check_resolvability(delta_half_a, delta_half_b,
                            budgets={2: reps2, 8: reps8, 32: reps32})
    if not r.passed:
        print(r.summary())        # report the reliability failure as a result

    # Check 1: how much of the statistic is self-specific?
    n = check_null(cos_self=0.5406, cos_null=0.3798)
    print(n.summary())            # excess 0.1608, 29.7% self-specific

    # Check 2: does a matched arbitrary direction recover the gain?
    m = check_matched(treatment=xsa_losses, control=random_losses,
                      baseline=baseline_losses)
    print(m.summary())

Every result object carries ``n``, an interval where one is defined, and a
``passed`` flag whose meaning is documented on the class. None of them raise on
a negative result: a failed check is a finding, not an error.

Citation note
-------------
The null in ``check_null`` is standard practice in embedding geometry
(Ethayarajh 2019); we do not claim to have invented it. The contribution is its
absence inside attention-architecture motivation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from xsac.stats import (
    disattenuate,
    holm_bonferroni,
    cluster_bootstrap_ci,
    mean_ci,
    paired_test,
    replicate_agreement,
    split_half_reliability,
)

__all__ = [
    "check_resolvability", "check_null", "check_matched",
    "ResolvabilityResult", "NullResult", "MatchedResult",
    "RESOLVABILITY_RULE",
]


#: The pre-registered decision rule for Check 0. Stated as data so that a
#: paper can quote it and a test can assert it was applied unchanged.
RESOLVABILITY_RULE = (
    (0.6, "reliable",
     "Delta is reliable. Report rho and rho_corrected. A near-zero rho is a "
     "real decoupling."),
    (0.3, "attenuated",
     "Report both, lead with rho_corrected, and state the attenuation "
     "explicitly."),
    (0.0, "unresolvable",
     "Delta is not resolvable at this budget. No decoupling may be claimed. "
     "Report the reliability failure itself as the Check 0 result, and drop "
     "the correlation claim from the abstract."),
)


def _verdict(r_delta: float) -> tuple:
    if not math.isfinite(r_delta):
        return "unresolvable", RESOLVABILITY_RULE[-1][2]
    for threshold, name, action in RESOLVABILITY_RULE:
        if r_delta >= threshold:
            return name, action
    return RESOLVABILITY_RULE[-1][1], RESOLVABILITY_RULE[-1][2]


@dataclass
class ResolvabilityResult:
    """Check 0. ``passed`` means the effect is reliable enough to correlate.

    ``passed`` is True only in the "reliable" band (r_delta >= 0.6). In the
    "attenuated" band the measurement is usable but must be reported with the
    correction, so ``passed`` is False and ``verdict`` says why.
    """

    r_delta: float
    r_stat: float
    verdict: str
    action: str
    n: int
    budget_curve: Dict[int, float] = field(default_factory=dict)
    converges: Optional[bool] = None
    rho_observed: float = float("nan")
    rho_corrected: float = float("nan")

    @property
    def passed(self) -> bool:
        return self.verdict == "reliable"

    def summary(self) -> str:
        lines = [
            "Check 0 (resolvability): {}".format(self.verdict.upper()),
            "  split-half reliability of the effect   r_delta = {:.3f}".format(
                self.r_delta),
            "  split-half reliability of the statistic r_stat = {:.3f}".format(
                self.r_stat),
            "  n candidates = {}".format(self.n),
        ]
        if self.budget_curve:
            pts = ", ".join("{}:{:+.3f}".format(b, v)
                            for b, v in sorted(self.budget_curve.items()))
            lines.append("  budget sweep (rank agreement)  {}".format(pts))
            if self.converges is False:
                lines.append("  the curve is FLAT: more budget does not help, "
                             "so the estimate never converges")
        if math.isfinite(self.rho_observed):
            lines.append("  rho observed = {:+.3f}   rho corrected = {}".format(
                self.rho_observed,
                "{:+.3f}".format(self.rho_corrected)
                if math.isfinite(self.rho_corrected) else "undefined"))
        lines.append("  -> {}".format(self.action))
        return "\n".join(lines)


@dataclass
class NullResult:
    """Check 1. ``passed`` means the statistic survives its anisotropy null.

    ``passed`` is True when the excess is a majority of the raw statistic, i.e.
    most of what was measured is genuinely self-specific rather than explained
    by the null. A False result is not an error; on GPT-2 only 30% of
    cos(y_i, v_i) is self-specific, and that is the finding.
    """

    cos_self: float
    cos_null: float
    excess: float
    self_specific_fraction: float
    n: int = 0
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    label: str = ""
    #: What the two quantities are called. Defaults describe the XSA case, but
    #: Check 1 applies to any statistic with a null, and printing
    #: "cos(y_i, v_i)" beside an attention mass or an activation magnitude
    #: would mislabel a correct number.
    stat_name: str = "cos(y_i, v_i)"
    null_name: str = "cos(y_i, v_j)"

    @property
    def passed(self) -> bool:
        return (math.isfinite(self.self_specific_fraction)
                and self.self_specific_fraction >= 0.5)

    def summary(self) -> str:
        head = "Check 1 (anisotropy null){}: {}".format(
            " [{}]".format(self.label) if self.label else "",
            "statistic survives" if self.passed
            else "MOST OF THE STATISTIC IS THE NULL")
        width = max(len(self.stat_name), len(self.null_name), 14)
        lines = [head,
                 "  observed {:<{w}} = {:.4f}".format(
                     self.stat_name, self.cos_self, w=width),
                 "  null     {:<{w}} = {:.4f}".format(
                     self.null_name, self.cos_null, w=width),
                 "  excess   {:<{w}} = {:.4f}".format("", self.excess,
                                                      w=width)]
        if math.isfinite(self.ci_low):
            lines.append("  excess 95% CI          = [{:.4f}, {:.4f}]  n = {}"
                         .format(self.ci_low, self.ci_high, self.n))
        lines.append("  self-specific fraction = {:.1%} of the raw statistic"
                     .format(self.self_specific_fraction))
        return "\n".join(lines)


@dataclass
class MatchedResult:
    """Check 2. ``passed`` means the mechanism story survived its control.

    ``passed`` is True when the treatment beats the matched control by a
    statistically resolvable margin. False means a matched arbitrary
    intervention recovered the gain, so the effect is not specific to the
    direction the method's story is about.
    """

    treatment_vs_baseline: Dict[str, object]
    control_vs_baseline: Dict[str, object]
    treatment_vs_control: Dict[str, object]
    alpha: float = 0.05
    treatment_name: str = "treatment"
    control_name: str = "control"

    @property
    def passed(self) -> bool:
        p = self.treatment_vs_control.get("p", float("nan"))
        mean = self.treatment_vs_control.get("mean_delta", float("nan"))
        return bool(math.isfinite(p) and p < self.alpha
                    and math.isfinite(mean) and mean < 0)

    def summary(self) -> str:
        def fmt(d, label):
            return ("  {:<28s} mean {:+.5f}  95% CI [{:+.5f}, {:+.5f}]  "
                    "t = {:+.2f}  p = {:.4g}  n = {}".format(
                        label, d.get("mean_delta", float("nan")),
                        d.get("ci_low", float("nan")),
                        d.get("ci_high", float("nan")),
                        d.get("t", float("nan")), d.get("p", float("nan")),
                        d.get("n", 0)))

        verdict = ("the effect is specific to the treatment direction"
                   if self.passed else
                   "A MATCHED ARBITRARY INTERVENTION RECOVERS THE EFFECT: this "
                   "is a regulariser, not a mechanism")
        lines = ["Check 2 (matched intervention): {}".format(verdict),
                 fmt(self.treatment_vs_baseline,
                     "{} vs baseline".format(self.treatment_name)),
                 fmt(self.control_vs_baseline,
                     "{} vs baseline".format(self.control_name)),
                 fmt(self.treatment_vs_control,
                     "{} vs {}".format(self.treatment_name, self.control_name))]
        sdt = self.treatment_vs_baseline.get("sd_unpaired_treatment")
        sdp = self.treatment_vs_baseline.get("sd_paired")
        if isinstance(sdt, float) and isinstance(sdp, float) and sdp > 0:
            lines.append("  unpaired sd {:.5f} vs paired sd {:.5f}: pairing "
                         "removes {:.1f}x of the noise".format(
                             sdt, sdp, sdt / sdp))
        return "\n".join(lines)


def check_resolvability(effect_half_a: Sequence[float],
                        effect_half_b: Sequence[float],
                        statistic_half_a: Optional[Sequence[float]] = None,
                        statistic_half_b: Optional[Sequence[float]] = None,
                        budgets: Optional[Dict[int, List[List[float]]]] = None,
                        rho_observed: float = float("nan"),
                        eps: float = 0.0,
                        top_k: int = 8) -> ResolvabilityResult:
    """Check 0. Is the measured effect resolvable at this sample budget?

    Run this before any correlation. If the effect cannot be measured
    reliably, a near-zero correlation with anything is guaranteed and carries
    no information about the model.

    Parameters
    ----------
    effect_half_a, effect_half_b:
        Per-candidate measurements of the same effect (for example per-head
        delta loss) computed independently on two **disjoint** halves of the
        evaluation data. Their rank correlation is ``r_delta``.
    statistic_half_a, statistic_half_b:
        The same split-half treatment applied to the motivating statistic.
        Optional; without them ``r_stat`` is NaN and no disattenuated
        correlation is reported.
    budgets:
        Optional ``{n_documents: [replicate_estimates, ...]}``. Each value is a
        list of replicate measurement vectors at that budget. A flat curve
        across budgets means the estimate never converges.
    rho_observed:
        The raw correlation between effect and statistic, if already computed.
        Used only to report the disattenuated value.

    Returns
    -------
    ResolvabilityResult
    """
    rel_delta = split_half_reliability(effect_half_a, effect_half_b)
    r_delta = rel_delta["r"]
    r_stat = float("nan")
    if statistic_half_a is not None and statistic_half_b is not None:
        r_stat = split_half_reliability(statistic_half_a, statistic_half_b)["r"]

    curve: Dict[int, float] = {}
    if budgets:
        for budget, reps in sorted(budgets.items()):
            agr = replicate_agreement(reps, eps=eps, top_k=top_k)
            curve[int(budget)] = float(agr["mean_spearman"])

    converges: Optional[bool] = None
    if len(curve) >= 3:
        vals = [curve[b] for b in sorted(curve)]
        finite = [v for v in vals if math.isfinite(v)]
        # "Converges" means the agreement actually climbs with budget. A flat
        # or non-monotone curve that stays near zero is the CRPA signature:
        # replicate Spearman between -0.11 and +0.12, unchanged from budget 2
        # to budget 32.
        converges = bool(finite and (max(finite) - min(finite) > 0.2)
                         and finite[-1] > 0.3)

    verdict, action = _verdict(r_delta)
    return ResolvabilityResult(
        r_delta=r_delta, r_stat=r_stat, verdict=verdict, action=action,
        n=rel_delta["n"], budget_curve=curve, converges=converges,
        rho_observed=float(rho_observed),
        rho_corrected=disattenuate(rho_observed, r_delta, r_stat))


def check_null(cos_self, cos_null, label: str = "",
               paired: bool = True, seed: int = 0,
               stat_name: str = "cos(y_i, v_i)",
               null_name: str = "cos(y_i, v_j)",
               clusters=None) -> NullResult:
    """Check 1. How much of a similarity statistic survives its null?

    Accepts either two scalars (already-averaged values, as reported in a
    paper) or two equal-length sequences of per-candidate measurements. With
    sequences the excess gets a bootstrap interval and an ``n``.

    ``cos_self`` is the motivating statistic, for example the mean
    ``cos(y_i, v_i)``. ``cos_null`` is the anisotropy baseline, for example
    ``cos(y_i, v_j)`` with ``j != i`` sampled within the same sequence. The
    within-sequence part matters: sampling across sequences changes what the
    null controls for.
    """
    scalar = isinstance(cos_self, (int, float)) and isinstance(
        cos_null, (int, float))
    if scalar:
        s, nl = float(cos_self), float(cos_null)
        excess = s - nl
        frac = excess / s if s != 0 else float("nan")
        return NullResult(cos_self=s, cos_null=nl, excess=excess,
                          self_specific_fraction=frac, n=1, label=label,
                          stat_name=stat_name, null_name=null_name)

    a = [float(x) for x in cos_self]
    b = [float(x) for x in cos_null]
    if paired and len(a) != len(b):
        raise ValueError(
            "paired check_null needs equal lengths, got {} and {}. Pass "
            "paired=False to compare unpaired samples.".format(len(a), len(b)))
    mean_s = sum(a) / len(a) if a else float("nan")
    mean_n = sum(b) / len(b) if b else float("nan")
    if paired:
        diffs = [x - y for x, y in zip(a, b)]
        # Heads within a layer are not independent draws. Passing their layer
        # index resamples whole layers instead of rows, which is the fix for
        # the pseudoreplication the specification's own bug list flags: 72
        # rows carrying 24 independent values gave an interval far too narrow.
        # Row-level bootstrap remains the default so callers without cluster
        # information are unchanged, but they get a wider-than-honest interval
        # and should pass clusters where they have them.
        ci = (cluster_bootstrap_ci(diffs, clusters, seed=seed)
              if clusters is not None else mean_ci(diffs, seed=seed))
        # cluster_bootstrap_ci reports n_rows/n_clusters rather than n. The
        # reported n stays the number of observations either way.
        excess, lo, hi = ci["mean"], ci["ci_low"], ci["ci_high"]
        n = ci.get("n", ci.get("n_rows", len(diffs)))
    else:
        excess = mean_s - mean_n
        lo = hi = float("nan")
        n = min(len(a), len(b))
    frac = excess / mean_s if mean_s not in (0.0,) else float("nan")
    return NullResult(cos_self=mean_s, cos_null=mean_n, excess=excess,
                      self_specific_fraction=frac, n=int(n),
                      ci_low=lo, ci_high=hi, label=label,
                      stat_name=stat_name, null_name=null_name)


def check_matched(treatment: Sequence[float], control: Sequence[float],
                  baseline: Sequence[float], alpha: float = 0.05,
                  treatment_name: str = "treatment",
                  control_name: str = "control",
                  seed: int = 0) -> MatchedResult:
    """Check 2. Does a matched arbitrary intervention recover the effect?

    All three sequences must be **paired by seed**: element ``i`` of each is a
    run at the same seed, same initialisation, same data order, differing only
    in the intervention. That pairing is what makes a sub-0.001 nat effect
    resolvable with fewer than a dozen runs.

    Lower loss is better, so a negative ``mean_delta`` means the first
    argument improved on the second.
    """
    if not (len(treatment) == len(control) == len(baseline)):
        raise ValueError(
            "check_matched requires seed-paired inputs of equal length; got "
            "{}, {}, {}".format(len(treatment), len(control), len(baseline)))
    return MatchedResult(
        treatment_vs_baseline=paired_test(treatment, baseline, seed=seed),
        control_vs_baseline=paired_test(control, baseline, seed=seed),
        treatment_vs_control=paired_test(treatment, control, seed=seed),
        alpha=alpha, treatment_name=treatment_name, control_name=control_name)


def holm_secondary(pvalues: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni over the secondary arms only.

    The primary endpoint is pre-registered and excluded by design. Including
    it would spend its alpha on comparisons declared secondary in advance.
    """
    return holm_bonferroni(pvalues)
