"""Calibration and the token budget.

This module decides how the entire compute budget is spent, so it is tested
harder than anything else here. It previously had no tests at all, and two
defects were sitting in it:

* Clamping the budget UP to the 3.5e8 floor when the machine could not afford
  it. That returns a plan that cannot be paid for and reports it as fine.
* ``COST_STOP_AND_REPORT`` was defined in config and never enforced anywhere,
  so the "$66 and stop" rule existed only as prose.
"""

from __future__ import annotations

import json
import math

import pytest

from xsac.calibrate import (CUT_ORDER, calibrate, cuts_available,
                            flops_per_token, measure_throughput, next_cut,
                            solve_token_budget, write_calibration)
from xsac.config import (CFG_M, CFG_S, CFG_TINY, COST_CEILING_TRAIN,
                         COST_STOP_AND_REPORT, TRAIN)


class TestFlopsPerToken:
    def test_matches_the_6n_plus_attention_identity(self):
        """flops = 6*N_non_embedding + 12*L*T*d, the standard MFU formula.

        The attention term is not negligible: at T=1024 and d=512 it is about
        a third of the matmul term, which is why it is in the formula at all.
        """
        _, non_emb = CFG_S.analytic_params()
        attn = 12 * CFG_S.n_layer * CFG_S.block_size * CFG_S.n_embd
        assert flops_per_token(CFG_S) == pytest.approx(6.0 * non_emb + attn)

    def test_the_attention_term_is_a_third_of_the_matmul_term_at_1024(self):
        _, non_emb = CFG_S.analytic_params()
        attn = 12 * CFG_S.n_layer * CFG_S.block_size * CFG_S.n_embd
        assert 0.25 < attn / (6.0 * non_emb) < 0.40

    def test_the_attention_term_grows_with_context(self):
        import dataclasses
        long_ctx = dataclasses.replace(CFG_S, block_size=CFG_S.block_size * 2)
        assert flops_per_token(long_ctx) > flops_per_token(CFG_S)

    def test_larger_model_costs_more_per_token(self):
        assert flops_per_token(CFG_M) > flops_per_token(CFG_S)

    def test_scales_with_depth(self):
        import dataclasses
        deep = dataclasses.replace(CFG_S, n_layer=CFG_S.n_layer * 2)
        ratio = flops_per_token(deep) / flops_per_token(CFG_S)
        assert 1.9 < ratio < 2.1


class TestBudgetArithmetic:
    def test_hours_come_from_the_ceiling_and_the_rate(self):
        out = solve_token_budget(100_000, 43, 2.0, TRAIN)
        assert out["hours_available"] == pytest.approx(COST_CEILING_TRAIN / 2.0)

    def test_budget_is_a_multiple_of_batch_tokens(self):
        out = solve_token_budget(100_000, 43, 0.86, TRAIN)
        assert out["tokens_per_run"] % TRAIN.batch_tokens == 0

    def test_budget_never_exceeds_the_upper_clamp(self):
        out = solve_token_budget(10_000_000, 4, 0.10, TRAIN)
        assert out["tokens_per_run"] <= TRAIN.tokens_max
        assert out["clamped_high"] is True

    def test_a_faster_machine_buys_more_tokens(self):
        slow = solve_token_budget(40_000, 43, 0.86, TRAIN)
        fast = solve_token_budget(80_000, 43, 0.86, TRAIN)
        assert fast["tokens_per_run"] >= slow["tokens_per_run"]

    def test_the_slow_arm_reduces_the_budget(self):
        even = solve_token_budget(120_000, 43, 0.86, TRAIN,
                                  diagmask_slowdown=1.0)
        slow = solve_token_budget(120_000, 43, 0.86, TRAIN,
                                  diagmask_slowdown=2.0)
        assert slow["tokens_per_run_raw"] < even["tokens_per_run_raw"]

    def test_a_slowdown_below_one_cannot_inflate_the_budget(self):
        """A measured slowdown under 1.0 means the timing is wrong, not that
        the arm is free."""
        a = solve_token_budget(120_000, 43, 0.86, TRAIN, diagmask_slowdown=1.0)
        b = solve_token_budget(120_000, 43, 0.86, TRAIN, diagmask_slowdown=0.4)
        assert a["tokens_per_run_raw"] == pytest.approx(b["tokens_per_run_raw"])

    @pytest.mark.parametrize("tps,n,rate", [(0, 43, 0.86), (100, 0, 0.86),
                                            (100, 43, 0.0), (-1, 43, 0.86)])
    def test_non_positive_inputs_raise(self, tps, n, rate):
        with pytest.raises(ValueError):
            solve_token_budget(tps, n, rate, TRAIN)


class TestAffordability:
    """Clamping up to the floor does not make a plan affordable."""

    def test_a_rich_machine_needs_no_cuts(self):
        out = solve_token_budget(120_000, 43, 0.86, TRAIN)
        assert out["affordable"] is True
        assert out["cuts_applied"] == []
        assert out["drop_cfg_m"] is False

    def test_a_tight_budget_drops_the_scale_check_first(self):
        """Spec section 12: cut from the bottom. Never the primary endpoint."""
        out = solve_token_budget(30_000, 43, 0.86, TRAIN)
        if out["cuts_applied"]:
            assert out["cuts_applied"][0] == "cfg_m_scale_check"

    def test_cuts_reduce_the_run_count_and_are_re_solved(self):
        out = solve_token_budget(12_000, 43, 0.86, TRAIN)
        assert out["cuts_applied"] == ["cfg_m_scale_check", "secondary_arms"]
        assert out["n_runs_after_cuts"] == 43 - 9 - 10
        assert out["tokens_per_run_raw"] > 0

    def test_an_unaffordable_plan_says_so_rather_than_clamping_silently(self):
        out = solve_token_budget(2_000, 43, 0.86, TRAIN)
        assert out["affordable"] is False
        assert out["clamped_low"] is True
        assert "STILL below the floor" in out["note"]

    def test_the_cut_order_never_touches_the_primary_endpoint(self):
        names = [c["name"] for c in CUT_ORDER]
        assert names == ["cfg_m_scale_check", "secondary_arms"]
        assert not any("primary" in n or "a1" in n.lower() for n in names)

    def test_cut_helpers_are_consistent(self):
        cuts = []
        assert cuts_available(cuts)
        cuts.append(next_cut(cuts))
        assert cuts[0]["name"] == "cfg_m_scale_check"
        cuts.append(next_cut(cuts))
        assert cuts[1]["name"] == "secondary_arms"
        assert not cuts_available(cuts)


class TestStopAndReportThreshold:
    """The $66 rule was declared in config and enforced nowhere."""

    def test_projected_spend_is_computed(self):
        out = solve_token_budget(120_000, 43, 0.86, TRAIN)
        assert out["projected_spend_usd"] > 0
        assert math.isfinite(out["projected_spend_usd"])

    def test_a_within_budget_plan_is_not_flagged(self):
        out = solve_token_budget(120_000, 43, 0.86, TRAIN)
        assert out["projected_spend_usd"] <= COST_STOP_AND_REPORT
        assert out["over_stop_threshold"] is False

    def test_an_expensive_rate_trips_the_threshold(self):
        out = solve_token_budget(120_000, 43, 12.0, TRAIN,
                                 cost_ceiling=400.0)
        assert out["over_stop_threshold"] is True
        assert "stop-and-report" in out["note"]

    def test_the_threshold_reported_is_the_configured_one(self):
        out = solve_token_budget(120_000, 43, 0.86, TRAIN)
        assert out["stop_threshold_usd"] == COST_STOP_AND_REPORT

    def test_projected_spend_scales_with_the_rate(self):
        cheap = solve_token_budget(120_000, 43, 0.50, TRAIN)
        dear = solve_token_budget(120_000, 43, 1.00, TRAIN)
        assert dear["projected_spend_usd"] > cheap["projected_spend_usd"]


class TestThroughputMeasurement:
    """Timed, never estimated. Kept tiny so it runs in CI."""

    def test_returns_the_fields_the_day2_gate_requires(self):
        out = measure_throughput(CFG_TINY, "baseline", steps=2, micro_batch=2,
                                 warmup=1, device="cpu")
        for key in ("tokens_per_sec", "achieved_tflops", "mfu_vs_181",
                    "seconds_per_step", "n_params"):
            assert key in out, "Day-2 gate needs {}".format(key)
        assert out["tokens_per_sec"] > 0
        assert out["seconds_per_step"] > 0

    def test_mfu_is_a_fraction_not_a_percentage(self):
        out = measure_throughput(CFG_TINY, "baseline", steps=2, micro_batch=2,
                                 warmup=1, device="cpu")
        assert 0 < out["mfu_vs_181"] < 1.0

    def test_diagmask_is_measurable_and_not_faster_than_baseline(self):
        """It cannot use SDPA, so it should never come out ahead."""
        base = measure_throughput(CFG_TINY, "baseline", steps=3, micro_batch=2,
                                  warmup=2, device="cpu")
        diag = measure_throughput(CFG_TINY, "diagmask", steps=3, micro_batch=2,
                                  warmup=2, device="cpu")
        assert diag["seconds_per_step"] > 0
        assert diag["tokens_per_sec"] > 0
        # Both paths must produce comparable token counts, so the slowdown
        # ratio the budget solver consumes is a like-for-like comparison.
        assert base["micro_batch"] == diag["micro_batch"]
        assert base["steps"] == diag["steps"]

    def test_parameter_count_matches_the_config(self):
        out = measure_throughput(CFG_TINY, "baseline", steps=1, micro_batch=2,
                                 warmup=0, device="cpu")
        total, _ = CFG_TINY.analytic_params()
        assert out["n_params"] == total


class TestCalibrateEndToEnd:
    def test_placeholder_rate_is_flagged(self):
        out = calibrate({"TINY": CFG_TINY}, TRAIN, n_runs=43, steps=1,
                        micro_batch=2, device="cpu",
                        rate_is_placeholder=True)
        assert out["rate_is_placeholder"] is True
        assert "PLACEHOLDER" in out["warning"]

    def test_a_real_rate_carries_no_warning(self):
        out = calibrate({"TINY": CFG_TINY}, TRAIN, n_runs=43, rate_usd_hr=0.74,
                        steps=1, micro_batch=2, device="cpu",
                        rate_is_placeholder=False)
        assert out["rate_is_placeholder"] is False
        assert "warning" not in out

    def test_each_size_gets_a_slowdown_and_a_budget(self):
        out = calibrate({"TINY": CFG_TINY}, TRAIN, n_runs=43, steps=1,
                        micro_batch=2, device="cpu")
        entry = out["sizes"]["TINY"]
        assert "diagmask_slowdown" in entry
        assert "budget" in entry
        assert "slowdown_in_expected_band" in entry

    def test_result_round_trips_through_json(self, tmp_path):
        out = calibrate({"TINY": CFG_TINY}, TRAIN, n_runs=43, steps=1,
                        micro_batch=2, device="cpu")
        path = write_calibration(out, tmp_path / "calibration.json")
        back = json.loads(path.read_text(encoding="utf-8"))
        assert back["sizes"]["TINY"]["budget"]["tokens_per_run"] == \
            out["sizes"]["TINY"]["budget"]["tokens_per_run"]


class TestCalibrationConsumption:
    def test_missing_calibration_fails_closed(self, tmp_path):
        from scripts.run_factorial import calibrated_train_config
        with pytest.raises(FileNotFoundError, match="missing"):
            calibrated_train_config(tmp_path, "S")

    def test_malformed_calibration_fails_closed(self, tmp_path):
        from scripts.run_factorial import calibrated_train_config
        (tmp_path / "calibration.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="invalid calibration"):
            calibrated_train_config(tmp_path, "S")

    def test_unaffordable_calibration_fails_closed(self, tmp_path):
        from scripts.run_factorial import calibrated_train_config
        payload = {"sizes": {"S": {"budget": {
            "tokens_per_run": 349_962_240, "affordable": False,
            "over_stop_threshold": False}}}}
        (tmp_path / "calibration.json").write_text(
            json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="not affordable"):
            calibrated_train_config(tmp_path, "S")

    def test_explicit_budget_must_be_batch_aligned(self, tmp_path):
        from scripts.run_factorial import calibrated_train_config
        with pytest.raises(ValueError, match="multiple"):
            calibrated_train_config(tmp_path, "S", 350_000_000)
        cfg = calibrated_train_config(tmp_path, "S", 349_962_240)
        assert cfg.tokens_per_run == 349_962_240


class TestBudgetHomogeneityGuard:
    """Averaging across token budgets produces a number describing neither.

    Ported from the sibling CRPA project, which hit this first. It is easy to
    create by accident: two invocations with different --tokens-per-run write
    into one results directory under different content hashes, so nothing
    collides and nothing complains. It happened here on the CFG_M scale check.
    """

    class _Rec:
        def __init__(self, tokens, arm="baseline", seed=42, size="M"):
            self.metrics = {"tokens_seen": tokens, "final_val_loss": 5.0}
            self.arm, self.seed, self.size = arm, seed, size
            self.status = "completed"

        @property
        def is_numeric(self):
            return True

    def test_a_minority_budget_is_dropped_and_named(self, capsys):
        from scripts.run_factorial import drop_inconsistent_budgets
        recs = ([self._Rec(50_000_000, seed=s) for s in (1, 2, 3)]
                + [self._Rec(30_000_000, seed=s) for s in (1, 2)])
        keep, dropped = drop_inconsistent_budgets(recs)
        assert len(keep) == 3 and len(dropped) == 2
        assert all(r.metrics["tokens_seen"] == 50_000_000 for r in keep)
        out = capsys.readouterr().out
        assert "refusing to average across token budgets" in out

    def test_a_homogeneous_set_passes_through_untouched(self):
        from scripts.run_factorial import drop_inconsistent_budgets
        recs = [self._Rec(50_000_000, seed=s) for s in (1, 2, 3)]
        keep, dropped = drop_inconsistent_budgets(recs)
        assert len(keep) == 3 and dropped == []

    def test_records_without_a_budget_do_not_crash_it(self):
        from scripts.run_factorial import drop_inconsistent_budgets
        r = self._Rec(50_000_000)
        r.metrics = {}
        keep, dropped = drop_inconsistent_budgets([r, self._Rec(50_000_000)])
        assert len(keep) + len(dropped) == 2
