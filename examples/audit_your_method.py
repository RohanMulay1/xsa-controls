"""Audit your own attention modification with all three checks.

Run it as-is to see the checks on synthetic data that mimics a method whose
motivating statistic is largely inherited from attention's own geometry:

    python examples/audit_your_method.py

Then replace the three ``measure_*`` functions with your own measurements.
Nothing here needs a GPU, a trained model, or this repository's experiments;
``xsac.checks`` takes plain sequences of floats.

The three questions, in the order worth asking them:

1. **Is the effect resolvable at all?** Measure it twice on disjoint halves of
   your evaluation data. If the two estimates do not agree with each other,
   no correlation you compute against them means anything, and the ceiling on
   any observable correlation is ``sqrt(r_delta * r_stat)``. Ask this first:
   it is the cheapest check and it can invalidate the other two.
2. **Does the statistic beat a matched null?** A statistic like
   ``cos(y_i, v_i)`` is partly inherited from anisotropy and from attention's
   own recency structure. Compare it against the same statistic computed
   against a partner the method does not claim is special.
3. **Does an arbitrary intervention recover the gain?** If replacing your
   principled edit with a matched arbitrary one produces the same improvement,
   the mechanism you described is not the one doing the work.
"""

import math
import pathlib
import sys

import numpy as np

# Run from a clone without installing: xsac lives one directory up.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from xsac.checks import check_matched, check_null, check_resolvability  # noqa: E402

RNG = np.random.default_rng(0)
N_HEADS = 96
N_SEEDS = 8

# Memoised so that "the statistic on half 0" is one fixed measurement wherever
# it is used. Without this the demo draws fresh numbers on each call and Check
# 0 and Check 1 silently disagree about the same quantity.
_CACHE: dict = {}


def _once(key, make):
    if key not in _CACHE:
        _CACHE[key] = make()
    return _CACHE[key]


def measure_effect_per_head(eval_half: int) -> np.ndarray:
    """Your intervention's per-head effect, measured on one half of eval data.

    Replace this. It must measure on *disjoint* halves: reusing data makes the
    two estimates agree for the wrong reason and inflates the reliability.
    """
    def draw():
        true_effect = RNG.standard_normal(N_HEADS) * 0.02
        noise = RNG.standard_normal(N_HEADS) * 0.05      # dominates the signal
        return true_effect + noise
    return _once(("effect", eval_half), draw)


def measure_statistic_per_head(eval_half: int) -> np.ndarray:
    """The statistic your method uses to decide where to intervene."""
    return _once(("stat", eval_half),
                 lambda: RNG.standard_normal(N_HEADS) * 0.3 + 0.45)


def measure_null_per_head() -> np.ndarray:
    """The same statistic against a partner the method does not call special.

    Match it to whatever structure the statistic inherits for free. This
    repository's own null matches sequence and causal admissibility by drawing
    uniformly from positions the query could attend; it does NOT match
    position or lag, so it cannot separate a recency effect from a
    self-specific one. Say which structures your null controls, and which it
    does not: an unmatched null makes any method look good, and a null that
    claims more matching than it does is worse than an honest weak one.
    """
    return _once("null", lambda: RNG.standard_normal(N_HEADS) * 0.3 + 0.30)


def measure_validation_loss(arm: str) -> np.ndarray:
    """Final validation loss per seed, for one arm of a paired run.

    All arms must share initialisation and data order per seed, so that a
    paired difference isolates the intervention.
    """
    # Paired design: a large per-seed term shared by every arm, plus a small
    # arm-specific residual. Sharing the seed term is what makes the paired
    # test powerful; the residual is what is left for it to resolve. Without
    # the residual the differences are constant and t is infinite.
    shared = _once("seed_noise",
                   lambda: RNG.standard_normal(N_SEEDS) * 0.01)
    residual = _once(("arm_noise", arm),
                     lambda: RNG.standard_normal(N_SEEDS) * 0.002)
    return (3.20 + shared + residual
            - {"baseline": 0.0, "yours": 0.004, "arbitrary": 0.0035}[arm])


def main() -> int:
    print("=" * 72)
    print("CHECK 0  Is the effect resolvable?")
    print("=" * 72)
    resolvable = check_resolvability(
        effect_half_a=measure_effect_per_head(0),
        effect_half_b=measure_effect_per_head(1),
        statistic_half_a=measure_statistic_per_head(0),
        statistic_half_b=measure_statistic_per_head(1),
    )
    print(resolvable.summary())

    if not resolvable.passed:
        ceiling = math.sqrt(max(resolvable.r_delta, 0.0)
                            * max(resolvable.r_stat, 0.0))
        print("\n  Check 0 did not pass ({}). Any correlation you compute "
              "against this effect is bounded by sqrt(r_delta * r_stat) "
              "= {:.3f}, so a real relationship and measurement noise "
              "cannot be told apart. Fix the measurement -- more eval "
              "data, a larger intervention -- before interpreting Check "
              "1 or Check 2.".format(resolvable.verdict, ceiling))

    print()
    print("=" * 72)
    print("CHECK 1  Does the statistic beat a matched null?")
    print("=" * 72)
    null = check_null(
        cos_self=measure_statistic_per_head(0),
        cos_null=measure_null_per_head(),
        label="my method's motivating statistic",
    )
    print(null.summary())

    print()
    print("=" * 72)
    print("CHECK 2  Does a matched arbitrary intervention do the same?")
    print("=" * 72)
    matched = check_matched(
        treatment=measure_validation_loss("yours"),
        control=measure_validation_loss("arbitrary"),
        baseline=measure_validation_loss("baseline"),
        treatment_name="my method",
        control_name="matched arbitrary direction",
    )
    print(matched.summary())

    print()
    print("=" * 72)
    verdicts = [("Check 0 resolvable", resolvable.passed),
                ("Check 1 beats null", null.passed),
                ("Check 2 beats control", matched.passed)]
    for name, ok in verdicts:
        print("  {:<24} {}".format(name, "PASS" if ok else "FAIL"))
    print("=" * 72)
    if not all(ok for _, ok in verdicts):
        print("\nA failure here is information, not a defeat. Check 0 failing "
              "means the experiment cannot answer the question yet. Check 1 "
              "failing means the statistic is largely structural. Check 2 "
              "failing means the mechanism is not the one described.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
