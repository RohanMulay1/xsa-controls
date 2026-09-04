"""Statistics and the three public checks.

``xsac.checks`` is the artifact the paper ships, so its API is pinned here as
tightly as its arithmetic.
"""

from __future__ import annotations

import math

import pytest

from xsac.checks import (RESOLVABILITY_RULE, check_matched, check_null,
                         check_resolvability, holm_secondary)
from xsac.stats import (cluster_bootstrap_ci, disattenuate, go_no_go,
                        holm_bonferroni, mean_ci,
                        minimum_detectable_effect, paired_test,
                        pinned_at_extremum, replicate_agreement, spearman,
                        split_half_reliability, threshold_sanity)


class TestMeanAlwaysCarriesNAndAnInterval:
    def test_mean_ci_reports_n(self):
        out = mean_ci([1.0, 2.0, 3.0, 4.0])
        assert out["n"] == 4
        assert out["ci_low"] < out["mean"] < out["ci_high"]

    def test_single_value_gives_no_interval_rather_than_a_fake_one(self):
        out = mean_ci([2.0])
        assert out["n"] == 1
        assert math.isnan(out["ci_low"])

    def test_empty_is_nan_not_zero(self):
        assert math.isnan(mean_ci([])["mean"])


class TestPairedTest:
    def test_detects_a_consistent_small_shift(self):
        base = [3.30, 3.31, 3.29, 3.32, 3.28, 3.30, 3.31, 3.29]
        treat = [b - 0.001 for b in base]
        res = paired_test(treat, base)
        assert res["n"] == 8
        assert res["mean_delta"] == pytest.approx(-0.001, abs=1e-9)
        assert res["p"] < 1e-6
        assert res["ci_high"] < 0

    def test_reports_unpaired_sd_alongside(self):
        """The contrast is itself a result: it shows what pairing removes."""
        base = [3.1, 3.5, 2.9, 3.7, 3.3, 3.0, 3.6, 3.2]
        treat = [b - 0.001 for b in base]
        res = paired_test(treat, base)
        assert res["sd_unpaired_treatment"] > 10 * res["sd_paired"]

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            paired_test([1.0, 2.0], [1.0])

    def test_no_pairs_returns_nan_not_zero(self):
        assert math.isnan(paired_test([], [])["mean_delta"])


class TestHolm:
    def test_step_down_is_monotone(self):
        adj = holm_bonferroni({"a": 0.01, "b": 0.02, "c": 0.04})
        assert adj["a"] <= adj["b"] <= adj["c"]

    def test_smallest_p_gets_the_full_family_size(self):
        adj = holm_bonferroni({"a": 0.01, "b": 0.5, "c": 0.6})
        assert adj["a"] == pytest.approx(0.03)

    def test_adjusted_p_never_exceeds_one(self):
        adj = holm_bonferroni({"a": 0.5, "b": 0.6, "c": 0.9})
        assert all(v <= 1.0 for v in adj.values())

    def test_primary_endpoint_is_excluded_by_the_caller(self):
        """holm_secondary takes only the secondary family."""
        adj = holm_secondary({"xsa": 0.02, "meanval": 0.03})
        assert set(adj) == {"xsa", "meanval"}


class TestGoNoGo:
    def test_mde_formula(self):
        assert minimum_detectable_effect(0.001, 8) == pytest.approx(
            2.9 * 0.001 / math.sqrt(8))

    @pytest.mark.parametrize("mde,branch", [
        (0.0015, "proceed"), (0.005, "reduce"), (0.01, "kill")])
    def test_branches(self, mde, branch):
        assert go_no_go(mde)["branch"] == branch

    def test_kill_branch_does_not_proceed(self):
        assert go_no_go(0.02)["proceed"] is False


class TestThresholdSanity:
    def test_flags_a_threshold_far_above_the_signal(self):
        """eps = 0.03 against a 2e-6 signal is a no-op filter."""
        out = threshold_sanity(0.03, [2e-6] * 50)
        assert out["is_noop"] is True
        assert out["ratio"] > 1e4
        assert "not doing what you think" in out["note"]

    def test_accepts_a_matched_threshold(self):
        out = threshold_sanity(0.02, [0.01, 0.03, 0.02])
        assert out["is_noop"] is False


class TestPinnedAtExtremum:
    def test_value_at_the_maximum_is_flagged(self):
        out = pinned_at_extremum(1.386278, 0.0, math.log(4))
        assert out["pinned"] and out["at_max"]
        assert "not a result" in out["note"]

    def test_intermediate_value_is_not_flagged(self):
        assert pinned_at_extremum(0.9, 0.0, math.log(4))["pinned"] is False


class TestReplicateAgreement:
    def test_degenerate_partition_is_undefined_not_one(self):
        """Agreement over a one-class partition measures nothing."""
        out = replicate_agreement([[1e-6, 2e-6, 3e-6], [3e-6, 1e-6, 2e-6]],
                                  eps=0.03, top_k=2)
        assert math.isnan(out["mean_classification_agreement"])
        assert out["classification_is_degenerate"] is True

    def test_real_split_reports_a_number(self):
        out = replicate_agreement([[0.1, 0.001, 0.2], [0.1, 0.001, 0.2]],
                                  eps=0.05, top_k=2)
        assert out["mean_classification_agreement"] == pytest.approx(1.0)

    def test_ranking_metrics_survive_degeneracy(self):
        out = replicate_agreement([[1e-6, 2e-6, 3e-6], [3e-6, 1e-6, 2e-6]],
                                  eps=0.03, top_k=2)
        assert math.isfinite(out["mean_spearman"])


class TestClusterBootstrap:
    def test_cluster_interval_is_wider_than_a_row_interval(self):
        """72 rows carrying 24 independent values must not look like n=72."""
        values, clusters = [], []
        for c in range(6):
            for _ in range(12):
                values.append(float(c))
                clusters.append(c)
        row = mean_ci(values)
        clu = cluster_bootstrap_ci(values, clusters)
        assert clu["n_clusters"] == 6 and clu["n_rows"] == 72
        assert (clu["ci_high"] - clu["ci_low"]) > (row["ci_high"] - row["ci_low"])


class TestDisattenuation:
    def test_corrects_upward(self):
        assert disattenuate(0.3, 0.5, 0.5) == pytest.approx(0.6)

    def test_undefined_when_reliability_is_non_positive(self):
        assert math.isnan(disattenuate(0.3, 0.0, 0.5))
        assert math.isnan(disattenuate(0.3, -0.1, 0.5))


class TestCheck0Resolvability:
    def test_reliable_effect_passes(self):
        a = [0.1, 0.5, 0.2, 0.9, 0.3, 0.7, 0.4, 0.8]
        res = check_resolvability(a, a, a, a, rho_observed=0.3)
        assert res.r_delta == pytest.approx(1.0)
        assert res.verdict == "reliable"
        assert res.passed is True

    def test_unresolvable_effect_blocks_the_correlation_claim(self):
        a = [0.1, 0.5, 0.2, 0.9, 0.3, 0.7, 0.4, 0.8]
        b = list(reversed(a))
        res = check_resolvability(a, b)
        assert res.verdict == "unresolvable"
        assert res.passed is False
        assert "drop the correlation claim" in res.action

    def test_flat_budget_curve_is_reported_as_non_convergent(self):
        """The CRPA signature: replicate Spearman between -0.11 and +0.12,
        unchanged from budget 2 to budget 32."""
        import random
        rng = random.Random(0)
        budgets = {b: [[rng.random() for _ in range(24)] for _ in range(3)]
                   for b in (2, 8, 32)}
        res = check_resolvability([1, 2, 3, 4], [4, 3, 2, 1], budgets=budgets,
                                  eps=0.03, top_k=8)
        assert res.converges is False
        assert set(res.budget_curve) == {2, 8, 32}

    def test_summary_mentions_the_verdict(self):
        res = check_resolvability([1, 2, 3, 4], [4, 3, 2, 1])
        assert "Check 0" in res.summary()

    def test_the_rule_is_the_pre_registered_one(self):
        thresholds = [t for t, _, _ in RESOLVABILITY_RULE]
        assert thresholds == [0.6, 0.3, 0.0]


class TestCheck1Null:
    def test_gpt2_published_values(self):
        res = check_null(0.5406, 0.3798, label="GPT-2")
        assert res.excess == pytest.approx(0.1608, abs=1e-4)
        assert res.self_specific_fraction == pytest.approx(0.297, abs=0.002)
        assert res.passed is False, (
            "under 50% self-specific: most of the statistic is the null")

    def test_sequences_get_an_interval_and_an_n(self):
        a = [0.55, 0.52, 0.58, 0.51, 0.54, 0.53]
        b = [0.38, 0.36, 0.40, 0.35, 0.37, 0.39]
        res = check_null(a, b)
        assert res.n == 6
        assert res.ci_low < res.excess < res.ci_high

    def test_a_statistic_that_survives_its_null_passes(self):
        res = check_null(0.90, 0.10)
        assert res.passed is True

    def test_paired_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            check_null([0.1, 0.2], [0.1])

    def test_summary_states_both_numbers(self):
        s = check_null(0.5406, 0.3798).summary()
        assert "0.5406" in s and "0.3798" in s


class TestCheck2Matched:
    def _seeds(self, n=8):
        return [3.300 + 0.01 * (i % 3) for i in range(n)]

    def test_control_recovering_the_effect_fails_the_check(self):
        base = self._seeds()
        xsa = [b - 0.0010 for b in base]
        rnd = [b - 0.0010 for b in base]
        res = check_matched(xsa, rnd, base, treatment_name="xsa",
                            control_name="random")
        assert res.passed is False
        assert "regulariser, not a mechanism" in res.summary()

    def test_treatment_beating_the_control_passes(self):
        base = self._seeds()
        xsa = [b - 0.0100 for b in base]
        rnd = [b - 0.0010 for b in base]
        res = check_matched(xsa, rnd, base)
        assert res.passed is True

    def test_unequal_lengths_raise(self):
        with pytest.raises(ValueError):
            check_matched([1.0, 2.0], [1.0], [1.0, 2.0])

    def test_all_three_contrasts_are_reported(self):
        base = self._seeds()
        res = check_matched([b - 0.001 for b in base],
                            [b - 0.002 for b in base], base)
        for d in (res.treatment_vs_baseline, res.control_vs_baseline,
                  res.treatment_vs_control):
            assert d["n"] == 8 and "ci_low" in d


class TestSpearmanEdgeCases:
    def test_constant_input_is_nan_not_zero(self):
        assert math.isnan(spearman([1, 1, 1, 1], [1, 2, 3, 4]))

    def test_too_few_points_is_nan(self):
        assert math.isnan(spearman([1, 2], [2, 1]))

    def test_perfect_rank_agreement(self):
        assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


class TestSplitHalfReliability:
    def test_resolvable_flag_uses_the_0_3_threshold(self):
        a = [1, 2, 3, 4, 5, 6]
        assert split_half_reliability(a, a)["resolvable"] is True
        assert split_half_reliability(a, list(reversed(a)))["resolvable"] is False


class TestZeroVarianceLimit:
    """A constant non-zero difference is the strongest evidence, not a NaN.

    It is still flagged, because with small n an exactly constant difference
    usually means the arms were not independent rather than that the effect is
    certain.
    """

    def test_constant_shift_gives_a_finite_verdict(self):
        base = [3.30, 3.31, 3.29, 3.32, 3.28, 3.30, 3.31, 3.29]
        res = paired_test([b - 0.001 for b in base], base)
        assert res["p"] == 0.0
        assert res["t"] == -math.inf
        assert res["zero_variance"] is True
        assert "genuinely independent" in res["degenerate_note"]

    def test_identical_inputs_are_not_significant(self):
        base = [3.30, 3.31, 3.29, 3.32]
        res = paired_test(list(base), base)
        assert res["mean_delta"] == 0.0
        assert res["p"] == 1.0

    def test_normal_variance_is_unflagged(self):
        base = [3.30, 3.31, 3.29, 3.32, 3.28, 3.30]
        treat = [3.29, 3.31, 3.27, 3.32, 3.26, 3.30]
        res = paired_test(treat, base)
        assert res["zero_variance"] is False
        assert math.isfinite(res["t"])


class TestCheck1LabelsAreNotHardcoded:
    """Check 1 applies to any statistic with a null, not only cosines.

    A6 measures attention mass and activation magnitude. Printing
    "cos(y_i, v_i)" beside those would mislabel a correct number, which is a
    reporting defect even though the arithmetic is right.
    """

    def test_default_labels_describe_the_xsa_case(self):
        s = check_null(0.54, 0.38).summary()
        assert "cos(y_i, v_i)" in s and "cos(y_i, v_j)" in s

    def test_labels_can_be_overridden_for_another_method(self):
        s = check_null(0.4028, 0.0034, label="attention_sink",
                       stat_name="mass on position 0",
                       null_name="mass on positions 1-3").summary()
        assert "mass on position 0" in s
        assert "mass on positions 1-3" in s
        assert "cos(y_i" not in s

    def test_a_statistic_that_dwarfs_its_null_passes(self):
        """Attention sinks survive their null decisively, unlike XSA's."""
        res = check_null(0.4028, 0.0034)
        assert res.passed is True
        assert res.self_specific_fraction > 0.99

    def test_the_gaussian_maximum_null_is_the_right_order(self):
        """max of d standard normals grows like sqrt(2 ln d)."""
        import math
        for d in (768, 1024, 4096):
            expected = math.sqrt(2 * math.log(d))
            assert 3.0 < expected < 5.0
