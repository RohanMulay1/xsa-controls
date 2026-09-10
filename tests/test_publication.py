"""Checks for retained publication artifacts."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import check_figures  # noqa: E402
from xsac.checks import check_null  # noqa: E402


class TestFiguresSurviveGreyscale:
    def test_committed_palette_passes(self):
        assert check_figures.check_palette() == []

    def test_luminance_collision_is_caught(self, monkeypatch):
        monkeypatch.setattr(
            check_figures, "SERIES", {"a": "#4d4d4d", "b": "#4a3aa7"}
        )
        monkeypatch.setattr(check_figures, "MARKERS", {"a": "o", "b": "s"})
        monkeypatch.setattr(
            check_figures, "LINESTYLES", {"a": "-", "b": "--"}
        )
        assert any("luminance" in item for item in check_figures.check_palette())

    def test_duplicate_markers_are_caught(self, monkeypatch):
        monkeypatch.setattr(
            check_figures, "SERIES", {"a": "#000000", "b": "#ffffff"}
        )
        monkeypatch.setattr(check_figures, "MARKERS", {"a": "o", "b": "o"})
        monkeypatch.setattr(
            check_figures, "LINESTYLES", {"a": "-", "b": "--"}
        )
        assert any("marker" in item for item in check_figures.check_palette())

    def test_relative_luminance_matches_wcag_anchors(self):
        assert check_figures.relative_luminance((0, 0, 0)) == pytest.approx(0.0)
        assert check_figures.relative_luminance((255, 255, 255)) == pytest.approx(1.0)


class TestWorkedExample:
    def test_example_runs_end_to_end_on_cpu(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "examples" / "audit_your_method.py")],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, result.stderr
        for expected in (
            "CHECK 0",
            "CHECK 1",
            "CHECK 2",
            "Check 0 resolvable",
            "Check 1 beats null",
            "Check 2 beats control",
        ):
            assert expected in result.stdout


class TestClusteredIntervals:
    def test_clustering_widens_the_interval(self):
        rng = np.random.default_rng(0)
        layers = np.repeat(np.arange(12), 12)
        offsets = rng.normal(0, 0.08, 12)
        cos_self = np.array(
            [0.5 + offsets[layer] + rng.normal(0, 0.02) for layer in layers]
        )
        cos_null = np.array(
            [0.3 + offsets[layer] + rng.normal(0, 0.02) for layer in layers]
        )
        row_interval = check_null(cos_self, cos_null)
        cluster_interval = check_null(cos_self, cos_null, clusters=layers)
        assert (cluster_interval.ci_high - cluster_interval.ci_low) > (
            row_interval.ci_high - row_interval.ci_low
        )

    def test_default_is_unchanged_without_clusters(self):
        rng = np.random.default_rng(1)
        first = rng.normal(0.5, 0.1, 60)
        second = rng.normal(0.3, 0.1, 60)
        assert check_null(first, second).excess == pytest.approx(
            check_null(first, second, clusters=None).excess
        )


class TestPaperTables:
    def test_reported_tables_match_structured_results(self):
        script = ROOT / "paper" / "iclr2027" / "check_tables.py"
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=script.parent,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
