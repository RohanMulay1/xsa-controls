"""
Resumable factorial orchestrator.

    python scripts/run_factorial.py --smoke                  # CPU, seconds
    python scripts/run_factorial.py --size S --seeds 42 1337 --arms baseline random
    python scripts/run_factorial.py --pilot                  # Day-3 GO/NO-GO

Resumability is free: each cell's filename is a content hash of its config, so
a completed cell is one whose file exists. Killing and restarting skips what is
done. That property is checked by ``tests/test_gates.py``.

Nothing here ever writes a number for a cell that did not run. A crash is
recorded with status ``failed`` and its traceback; an OOM with status ``oom``.
Both are visible in ``factorial_*.csv`` as rows with no loss, rather than as
absent rows that look like they were never specified.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xsac.checks import check_matched  # noqa: E402
from xsac.config import (ARMS, PRIMARY_ENDPOINT, SECONDARY_ARMS,  # noqa: E402
                         ExperimentConfig, TRAIN, TrainConfig, size_config,
                         smoke_variant)
from xsac.data import ensure_smoke_data  # noqa: E402
from xsac.stats import (go_no_go, holm_bonferroni,  # noqa: E402
                        minimum_detectable_effect, paired_test)
from xsac.runmeta import (RunRecord, is_done, numeric_records,  # noqa: E402
                          read_records, records_to_rows, run_id, write_csv,
                          write_record)
from xsac.train import train_one  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data"


def calibrated_train_config(results_dir: Path, size: str,
                            override: Optional[float] = None) -> TrainConfig:
    """Use the budget calibration actually solved for, not the default.

    calibrate.py exists to size ``tokens_per_run`` against measured throughput
    and the real hourly rate. Nothing consumed its output: the factorial used
    the 4.5e8 default regardless, so the whole Day-2 gate computed a number
    that changed nothing. That is fixed here, and the value used is recorded
    on every run record so a result can be traced to the budget that produced
    it.
    """
    if override is not None:
        return replace(TRAIN, tokens_per_run=float(override))
    path = Path(results_dir) / "calibration.json"
    if not path.exists():
        return TRAIN
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        budget = payload["sizes"][size]["budget"]
    except Exception:
        return TRAIN
    tokens = budget.get("tokens_per_run")
    if not tokens:
        return TRAIN
    print("  using calibrated budget: {:.3g} tokens/run "
          "(affordable={}, cuts={})".format(
              tokens, budget.get("affordable"), budget.get("cuts_applied")))
    return replace(TRAIN, tokens_per_run=float(tokens))


def cell_config(arm: str, seed: int, size: str, smoke: bool,
                train_cfg: Optional[TrainConfig] = None) -> ExperimentConfig:
    cfg = ExperimentConfig(arm=arm, seed=seed, size=size,
                           model=size_config(size), train=train_cfg or TRAIN)
    return smoke_variant(cfg) if smoke else cfg


def run_cell(cfg: ExperimentConfig, data_dir: Path, results_dir: Path,
             device: str, force: bool) -> RunRecord:
    payload = cfg.to_dict()
    rid = run_id(payload, cfg.seed)
    if not force and is_done(results_dir, rid):
        print("    skip (done)  {}".format(rid))
        existing = [r for r in read_records(results_dir) if r.run_id == rid]
        return existing[0]

    started = time.time()
    try:
        metrics = train_one(cfg, data_dir, device=device)
        rec = RunRecord(
            run_id=rid, experiment="factorial", arm=cfg.arm, seed=cfg.seed,
            size=cfg.size, status="smoke" if cfg.smoke else "completed",
            config=payload, metrics=metrics,
            duration_s=time.time() - started,
            note="smoke: tiny model and synthetic data, NOT reportable"
                 if cfg.smoke else "")
    except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover
        rec = RunRecord(run_id=rid, experiment="factorial", arm=cfg.arm,
                        seed=cfg.seed, size=cfg.size, status="oom",
                        config=payload, error=str(exc)[:500],
                        duration_s=time.time() - started)
    except Exception as exc:
        rec = RunRecord(run_id=rid, experiment="factorial", arm=cfg.arm,
                        seed=cfg.seed, size=cfg.size, status="failed",
                        config=payload,
                        error="{}: {}\n{}".format(type(exc).__name__, exc,
                                                  traceback.format_exc()[:800]),
                        duration_s=time.time() - started)
    write_record(rec, results_dir)
    return rec


def paired_tables(records: List[RunRecord], size: str) -> List[Dict[str, Any]]:
    """Paired tests per arm, primary endpoint first and labelled."""
    by_arm: Dict[str, Dict[int, float]] = {}
    for r in numeric_records(records):
        if r.size != size:
            continue
        loss = r.metrics.get("final_val_loss")
        if loss is None:
            continue
        by_arm.setdefault(r.arm, {})[r.seed] = float(loss)

    base = by_arm.get("baseline", {})
    if not base:
        return []

    rows: List[Dict[str, Any]] = []
    raw_p: Dict[str, float] = {}
    order = [PRIMARY_ENDPOINT] + [a for a in ARMS
                                  if a not in (PRIMARY_ENDPOINT, "baseline")]
    for arm in order:
        vals = by_arm.get(arm)
        if not vals:
            continue
        seeds = sorted(set(vals) & set(base))
        if len(seeds) < 2:
            continue
        res = paired_test([vals[s] for s in seeds], [base[s] for s in seeds])
        is_primary = arm == PRIMARY_ENDPOINT
        if not is_primary:
            raw_p[arm] = float(res.get("p", float("nan")))
        rows.append({
            "size": size, "arm": arm, "vs": "baseline",
            "endpoint": "PRIMARY (pre-registered)" if is_primary
                        else "secondary",
            "n_seeds": res["n"], "mean_delta": res["mean_delta"],
            "ci_low": res["ci_low"], "ci_high": res["ci_high"],
            "t": res["t"], "p": res["p"], "wilcoxon_p": res["wilcoxon_p"],
            "cohen_dz": res["cohen_dz"], "sd_paired": res["sd_paired"],
            "sd_unpaired_treatment": res["sd_unpaired_treatment"],
            "sd_unpaired_control": res["sd_unpaired_control"],
            "seeds": ",".join(str(s) for s in seeds),
        })

    # Holm over the secondary family only. The primary endpoint was
    # pre-registered and does not spend its alpha here.
    adj = holm_bonferroni(raw_p)
    for row in rows:
        row["p_holm"] = (float("nan") if row["arm"] == PRIMARY_ENDPOINT
                         else adj.get(row["arm"], float("nan")))
    return rows


def pilot_report(records: List[RunRecord], size: str, n_seeds_planned: int
                 ) -> Dict[str, Any]:
    """Day-3 GO/NO-GO. Estimates paired sigma and applies the rule."""
    rows = paired_tables(records, size)
    sigmas = [r["sd_paired"] for r in rows
              if isinstance(r["sd_paired"], float) and r["sd_paired"] > 0]
    if not sigmas:
        # With a single pilot seed there is no paired sigma yet. Say so
        # rather than inventing one.
        return {"sigma_paired": float("nan"), "mde": float("nan"),
                "n_seeds_planned": n_seeds_planned,
                "branch": "undetermined",
                "action": "a single pilot seed gives no paired sigma; run at "
                          "least 2 seeds before applying the rule",
                "proceed": False}
    sigma = max(sigmas)          # the worst arm sets the budget
    mde = minimum_detectable_effect(sigma, n_seeds_planned)
    decision = go_no_go(mde)
    return {"sigma_paired": sigma, "mde": mde,
            "n_seeds_planned": n_seeds_planned, **decision}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", default="S", choices=["S", "M"])
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--seeds", nargs="*", type=int,
                    default=[42, 1337, 2024, 7, 99, 512, 8191, 31337])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pilot", action="store_true",
                    help="one seed, all arms, then the GO/NO-GO rule")
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n-seeds-planned", type=int, default=8)
    ap.add_argument("--tokens-per-run", type=float, default=None,
                    help="override the calibrated budget; recorded on every "
                         "run so the number is traceable")
    args = ap.parse_args(argv)

    seeds = args.seeds[:1] if args.pilot else args.seeds
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = RESULTS / ("factorial_smoke" if args.smoke else
                             "factorial_{}".format(args.size.lower()))
    results_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = calibrated_train_config(RESULTS, args.size,
                                        args.tokens_per_run)
    if args.smoke:
        cfg0 = cell_config("baseline", 0, args.size, True, train_cfg)
        ensure_smoke_data(DATA, cfg0.model.vocab_size)

    print("factorial: size={} arms={} seeds={} device={} smoke={}".format(
        args.size, args.arms, seeds, device, args.smoke))
    for seed in seeds:
        print("  seed {}".format(seed))
        for arm in args.arms:
            cfg = cell_config(arm, seed, args.size, args.smoke, train_cfg)
            rec = run_cell(cfg, DATA, results_dir, device, args.force)
            if rec.is_numeric:
                print("    {:9s} val_loss {:.6f}  tokens {}  {:.1f}s".format(
                    arm, rec.metrics.get("final_val_loss", float("nan")),
                    rec.metrics.get("tokens_seen", 0), rec.duration_s))
            else:
                print("    {:9s} status={} :: {}".format(
                    arm, rec.status, rec.error[:120]))

    records = read_records(results_dir)
    suffix = "smoke" if args.smoke else args.size.lower()

    rows = records_to_rows(records)
    write_csv(rows, RESULTS / "factorial_{}.csv".format(suffix))

    paired = paired_tables(records, args.size)
    if paired:
        write_csv(paired, RESULTS / "paired_tests_{}.csv".format(suffix))
        print("\npaired tests (primary endpoint first):")
        for r in paired:
            print("  {:9s} {:<24s} mean {:+.6f}  CI [{:+.6f},{:+.6f}]  "
                  "t {:+.2f}  p {:.4g}  n {}".format(
                      r["arm"], r["endpoint"], r["mean_delta"], r["ci_low"],
                      r["ci_high"], r["t"], r["p"], r["n_seeds"]))

    if args.pilot:
        rep = pilot_report(records, args.size, args.n_seeds_planned)
        (RESULTS / "pilot_decision.json").write_text(
            json.dumps(rep, indent=2), encoding="utf-8")
        write_csv([{k: v for k, v in rep.items()}],
                  RESULTS / "pilot.csv")
        print("\nDay-3 GO/NO-GO")
        print("  sigma_paired = {}".format(rep["sigma_paired"]))
        print("  MDE          = {}".format(rep["mde"]))
        print("  branch       = {}".format(rep["branch"]))
        print("  action       = {}".format(rep["action"]))
        print("\nThis decision must be written into BUDGET.md with the MDE "
              "quoted before any further GPU spend.")

    n_bad = sum(1 for r in records if not r.is_numeric)
    if n_bad:
        print("\n{} cell(s) did not produce a number and are recorded as "
              "such. They are not silently dropped.".format(n_bad))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
