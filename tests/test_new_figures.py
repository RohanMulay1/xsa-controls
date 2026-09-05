"""Contracts for the A2 scatter and the power curve.

Both figures make a claim in their own right, so both can be wrong in ways a
rendered PNG does not reveal. These pin the parts that would be silently
wrong: a scatter whose annotations come from a different model's row, and a
power curve labelled with a token budget the runs did not actually use.
"""

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xsac.figures import (FigureSkipped, fig6_a2_scatter,  # noqa: E402
                          fig7_power)
from xsac.stats import minimum_detectable_effect  # noqa: E402


def write(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def per_head(model, n=12):
    return [{"model": model, "layer": i // 4, "head": i % 4,
             "cos_self": 0.30 + 0.02 * i, "excess": 0.10 + 0.01 * i,
             "delta_pooled": 0.001 * i, "delta_half_a": 0.001 * i,
             "delta_half_b": 0.0011 * i} for i in range(n)]


class TestA2Scatter:
    def test_missing_per_head_data_is_skipped_not_drawn(self, tmp_path):
        write(tmp_path / "a2_correlations.csv",
              ["model", "statistic", "rho_raw"],
              [{"model": "m", "statistic": "cos_self", "rho_raw": 0.4}])
        with pytest.raises(FigureSkipped, match="a2_per_head"):
            fig6_a2_scatter(tmp_path, tmp_path / "out")

    def test_missing_correlations_is_skipped_not_drawn(self, tmp_path):
        write(tmp_path / "a2_per_head.csv", list(per_head("m")[0]),
              per_head("m"))
        with pytest.raises(FigureSkipped, match="a2_correlations"):
            fig6_a2_scatter(tmp_path, tmp_path / "out")

    def test_writes_png_pdf_and_source_data(self, tmp_path):
        rows = per_head("gpt2")
        write(tmp_path / "a2_per_head.csv", list(rows[0]), rows)
        write(tmp_path / "a2_correlations.csv",
              ["model", "statistic", "rho_raw", "rho_disattenuated",
               "r_delta", "ceiling"],
              [{"model": "gpt2", "statistic": "cos_self", "rho_raw": 0.45,
                "rho_disattenuated": 0.52, "r_delta": 0.75,
                "ceiling": 0.86}])
        out = tmp_path / "figs"
        paths = fig6_a2_scatter(tmp_path, out)
        assert {p.suffix for p in paths} == {".png", ".pdf", ".csv"}
        assert (out / "fig6_a2_scatter_data.csv").exists()

    def test_one_panel_per_model_and_data_covers_every_head(self, tmp_path):
        """A panel silently dropped, or a model's heads plotted under another
        model's annotations, would look entirely normal in the PNG."""
        rows = per_head("gpt2") + per_head("EleutherAI/pythia-160m", 8)
        write(tmp_path / "a2_per_head.csv", list(rows[0]), rows)
        write(tmp_path / "a2_correlations.csv",
              ["model", "statistic", "rho_raw", "rho_disattenuated",
               "r_delta", "ceiling"],
              [{"model": m, "statistic": "cos_self", "rho_raw": r,
                "rho_disattenuated": r, "r_delta": 0.5, "ceiling": 0.7}
               for m, r in (("gpt2", 0.45),
                            ("EleutherAI/pythia-160m", 0.001))])
        out = tmp_path / "figs"
        fig6_a2_scatter(tmp_path, out)
        with (out / "fig6_a2_scatter_data.csv").open(encoding="utf-8") as fh:
            written = list(csv.DictReader(fh))
        assert len(written) == len(rows)
        assert {r["model"] for r in written} == {"gpt2",
                                                 "EleutherAI/pythia-160m"}


class TestPowerCurve:
    def _paired(self, tmp_path, stem, sigma, tokens_seen, n_seeds=8):
        write(tmp_path / ("paired_tests_%s.csv" % stem),
              ["arm", "sd_paired", "n_seeds"],
              [{"arm": "random", "sd_paired": sigma, "n_seeds": n_seeds}])
        write(tmp_path / ("factorial_%s.csv" % stem),
              ["arm", "tokens_seen"],
              [{"arm": "random", "tokens_seen": tokens_seen}])

    def test_no_paired_file_is_skipped(self, tmp_path):
        with pytest.raises(FigureSkipped, match="paired_tests"):
            fig7_power(tmp_path, tmp_path / "out")

    def test_smoke_runs_are_never_plotted(self, tmp_path):
        """A smoke factorial has a paired file too, and plotting its sigma
        as if it were a budget would put a fabricated curve in the paper."""
        self._paired(tmp_path, "smoke", 0.02, 1000)
        with pytest.raises(FigureSkipped):
            fig7_power(tmp_path, tmp_path / "out")

    def test_curve_matches_the_mde_formula(self, tmp_path):
        self._paired(tmp_path, "s", 0.005, 399900672)
        out = tmp_path / "figs"
        fig7_power(tmp_path, out)
        with (out / "fig7_power_data.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            expected = minimum_detectable_effect(0.005, int(r["n_seeds"]))
            assert abs(float(r["mde_nats"]) - expected) < 1e-12

    def test_budget_label_comes_from_tokens_actually_seen(self, tmp_path):
        self._paired(tmp_path, "s", 0.005, 399900672)
        out = tmp_path / "figs"
        fig7_power(tmp_path, out)
        with (out / "fig7_power_data.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert {float(r["tokens_per_run"]) for r in rows} == {399900672.0}

    def test_mixed_budgets_are_not_given_a_single_label(self, tmp_path):
        """Two budgets in one factorial is what the homogeneity guard exists
        to catch; labelling the curve with either one would hide it."""
        write(tmp_path / "paired_tests_s.csv",
              ["arm", "sd_paired", "n_seeds"],
              [{"arm": "random", "sd_paired": 0.005, "n_seeds": 8}])
        write(tmp_path / "factorial_s.csv", ["arm", "tokens_seen"],
              [{"arm": "random", "tokens_seen": 5e7},
               {"arm": "xsa", "tokens_seen": 4e8}])
        out = tmp_path / "figs"
        fig7_power(tmp_path, out)
        with (out / "fig7_power_data.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        import math
        assert all(math.isnan(float(r["tokens_per_run"])) for r in rows)

    def test_both_budgets_plot_when_both_exist(self, tmp_path):
        self._paired(tmp_path, "s", 0.0018, 399900672)
        self._paired(tmp_path, "s_pilot_5e7", 0.005, 49938432)
        out = tmp_path / "figs"
        fig7_power(tmp_path, out)
        with (out / "fig7_power_data.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert {float(r["tokens_per_run"]) for r in rows} == {
            399900672.0, 49938432.0}
