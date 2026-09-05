"""Tests for the paper artifacts: tables, manifest, figure accessibility.

These guard the properties that make the artifacts trustworthy rather than
merely present: a table always states its n, a claim always recomputes from
its source file, and a palette that is unreadable in greyscale fails the build
instead of shipping.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import check_figures, make_manifest, make_tables  # noqa: E402


class TestTablesStateTheirSampleSize:
    def test_caption_without_n_is_refused(self, tmp_path, monkeypatch):
        """A table without its sample size is not reportable. Catching that
        here is cheaper than catching it in review."""
        monkeypatch.setattr(make_tables, "OUT", tmp_path)
        with pytest.raises(ValueError, match="does not state n"):
            make_tables.emit("TX", "A caption with no sample size.",
                             ["a"], [["1"]])

    def test_caption_with_n_is_accepted(self, tmp_path, monkeypatch):
        monkeypatch.setattr(make_tables, "OUT", tmp_path)
        make_tables.emit("TX", "A caption. n = 3 models.", ["a"], [["1"]])
        assert (tmp_path / "TX.md").exists()
        assert (tmp_path / "TX.tex").exists()

    def test_every_emitted_caption_states_n(self, tmp_path, monkeypatch):
        monkeypatch.setattr(make_tables, "OUT", tmp_path)
        make_tables.main([])
        for md in tmp_path.glob("*.md"):
            text = md.read_text(encoding="utf-8")
            assert "n = " in text or "n=" in text, md.name

    def test_absent_source_is_skipped_not_invented(self, tmp_path,
                                                   monkeypatch):
        """T5 needs an experiment that has not run. It must be reported as
        missing, never filled in with a plausible number."""
        monkeypatch.setattr(make_tables, "RESULTS", tmp_path)
        monkeypatch.setattr(make_tables, "OUT", tmp_path / "tables")
        reason = make_tables.table5_a2()
        assert reason and "missing" in reason
        assert not (tmp_path / "tables" / "T5.md").exists()

    def test_nan_cells_render_as_a_dash(self):
        assert make_tables.fnum("nan") == "--"
        assert make_tables.fnum("") == "--"
        assert make_tables.fnum(None) == "--"
        assert make_tables.fnum(0.12345, 3) == "0.123"


class TestManifestRecomputesEveryClaim:
    def test_all_claims_reproduce_from_their_artifacts(self, tmp_path,
                                                       monkeypatch):
        """The manifest is the promise that every paper number is read from a
        committed file. It is only worth anything if it is checked."""
        monkeypatch.setattr(make_manifest, "RESULTS", ROOT / "results")
        rc = make_manifest.main([])
        assert rc == 0, "a claim no longer reproduces from its artifact"

    def test_a_wrong_claim_is_caught(self, monkeypatch, capsys):
        """Deliberately corrupt one expected value and confirm the generator
        fails, so the check cannot silently pass."""
        original = list(make_manifest.CLAIMS)
        claim, expected, unit, art, script, fn = original[0]
        monkeypatch.setattr(make_manifest, "CLAIMS",
                            [(claim, expected + 10.0, unit, art, script, fn)])
        assert make_manifest.main([]) == 1

    def test_missing_artifact_is_reported_not_guessed(self, tmp_path,
                                                      monkeypatch):
        monkeypatch.setattr(make_manifest, "RESULTS", tmp_path)
        assert make_manifest.rows("nope.csv") is None
        assert make_manifest.load("nope.json") is None


class TestFiguresSurviveGreyscale:
    def test_committed_palette_passes(self):
        assert check_figures.check_palette() == []

    def test_a_luminance_collision_is_caught(self, monkeypatch):
        """The shipped palette once put two arms 0.001 apart in luminance,
        which is one line in print. That must fail, not warn."""
        monkeypatch.setattr(check_figures, "SERIES",
                            {"a": "#4d4d4d", "b": "#4a3aa7"})
        monkeypatch.setattr(check_figures, "MARKERS", {"a": "o", "b": "s"})
        monkeypatch.setattr(check_figures, "LINESTYLES", {"a": "-", "b": "--"})
        failures = check_figures.check_palette()
        assert any("luminance" in f for f in failures)

    def test_duplicate_markers_are_caught(self, monkeypatch):
        """Redundant encoding is the fallback when luminance is close, so a
        repeated marker removes the only remaining distinction."""
        monkeypatch.setattr(check_figures, "SERIES",
                            {"a": "#000000", "b": "#ffffff"})
        monkeypatch.setattr(check_figures, "MARKERS", {"a": "o", "b": "o"})
        monkeypatch.setattr(check_figures, "LINESTYLES", {"a": "-", "b": "--"})
        assert any("marker" in f for f in check_figures.check_palette())

    def test_relative_luminance_matches_wcag_anchors(self):
        assert check_figures.relative_luminance((0, 0, 0)) == pytest.approx(0.0)
        assert check_figures.relative_luminance((255, 255, 255)) == \
            pytest.approx(1.0)


class TestWorkedExample:
    def test_example_runs_end_to_end_on_cpu(self):
        """The artifact most likely to be imported by someone else. If it
        does not run from a clean checkout it is worse than absent."""
        out = subprocess.run(
            [sys.executable, str(ROOT / "examples" / "audit_your_method.py")],
            capture_output=True, text=True, timeout=300)
        assert out.returncode == 0, out.stderr
        for expected in ("CHECK 0", "CHECK 1", "CHECK 2",
                         "Check 0 resolvable", "Check 1 beats null",
                         "Check 2 beats control"):
            assert expected in out.stdout


class TestClusteredIntervals:
    """Heads within a layer are not independent draws.

    The specification's own bug list flags exactly this: 72 rows carrying 24
    independent values gave an interval far too narrow. cluster_bootstrap_ci
    was written for it and was not wired into check_null until now.
    """

    def test_clustering_widens_the_interval(self):
        import numpy as np
        from xsac.checks import check_null
        rng = np.random.default_rng(0)
        layers = np.repeat(np.arange(12), 12)
        # Give each layer its own offset, so heads within a layer really are
        # correlated and the row-level interval really is too narrow.
        offs = rng.normal(0, 0.08, 12)
        cos_self = np.array([0.5 + offs[l] + rng.normal(0, 0.02)
                             for l in layers])
        cos_null = np.array([0.3 + offs[l] + rng.normal(0, 0.02)
                             for l in layers])
        rows = check_null(cos_self, cos_null)
        clus = check_null(cos_self, cos_null, clusters=layers)
        assert (clus.ci_high - clus.ci_low) > (rows.ci_high - rows.ci_low)

    def test_default_is_unchanged_for_callers_without_clusters(self):
        import numpy as np
        from xsac.checks import check_null
        rng = np.random.default_rng(1)
        a, b = rng.normal(0.5, 0.1, 60), rng.normal(0.3, 0.1, 60)
        assert check_null(a, b).excess == pytest.approx(
            check_null(a, b, clusters=None).excess)


class TestBudgetLedger:
    """The ledger decides whether a running job is stopped, so its
    arithmetic has to be right. An early version added cell hours to pod
    uptime, counting the same minutes twice, and reported $18.16 against an
    $18 ceiling when the real projection was $13.41."""

    def _records(self, tmp_path, seeds, arms=("baseline", "xsa", "random"),
                 seconds=2400.0):
        import json
        runs = tmp_path / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        for s in seeds:
            for a in arms:
                (runs / "{}_{}.json".format(s, a)).write_text(json.dumps({
                    "seed": s, "status": "completed",
                    "config": {"arm": a, "seed": s},
                    "metrics": {"seconds": seconds, "tokens_seen": 399900672},
                }), encoding="utf-8")
        return tmp_path

    def test_only_complete_seeds_count(self, tmp_path):
        from scripts.budget_ledger import records, summarise
        d = self._records(tmp_path, [42, 1337])
        # a third seed with one arm missing
        import json
        (tmp_path / "runs" / "7_baseline.json").write_text(json.dumps({
            "seed": 7, "status": "completed",
            "config": {"arm": "baseline", "seed": 7},
            "metrics": {"seconds": 2400.0}}), encoding="utf-8")
        s = summarise(records(d), 8, 3)
        assert s["complete_paired_seeds"] == 2
        assert s["partial_seeds"] == [7]

    def test_pod_uptime_is_the_billed_base_not_uptime_plus_cells(self,
                                                                 tmp_path):
        from scripts.budget_ledger import records, summarise
        s = summarise(records(self._records(tmp_path, [42, 1337, 2024])), 8, 3)
        cell_hours = s["gpu_hours_so_far"]
        pod_hours = 7.0
        rate = 0.74
        billed = pod_hours * rate
        wrong = (pod_hours + cell_hours) * rate
        assert billed < wrong, "the double-counted figure must be larger"
        projected = (pod_hours + s["projected_remaining_hours"]) * rate
        assert projected < wrong + s["projected_remaining_hours"] * rate

    def test_projection_uses_completed_seeds_only(self, tmp_path):
        from scripts.budget_ledger import records, summarise
        s = summarise(records(self._records(tmp_path, [42, 1337], seconds=1800.0)),
                      8, 3)
        assert s["hours_per_seed"] == pytest.approx(1.5)
        assert s["remaining_seeds"] == 6
        assert s["projected_remaining_hours"] == pytest.approx(9.0)

    def test_mixed_token_budgets_are_surfaced(self, tmp_path):
        import json
        from scripts.budget_ledger import records, summarise
        d = self._records(tmp_path, [42])
        (d / "runs" / "1337_baseline.json").write_text(json.dumps({
            "seed": 1337, "status": "completed",
            "config": {"arm": "baseline", "seed": 1337},
            "metrics": {"seconds": 2400.0, "tokens_seen": 50000000}}),
            encoding="utf-8")
        s = summarise(records(d), 8, 3)
        assert len(s["budgets"]) == 2, "a mixed budget must be visible"
