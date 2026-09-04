"""
Day-2 calibration: measure throughput, solve the token budget.

    python scripts/calibrate_cli.py --rate 0.86 --n-runs 43
    python scripts/calibrate_cli.py --smoke        # CPU, tiny, seconds

The budget MUST be solved against the actual hourly rate of the machine that
will run the factorial. Passing --rate is what clears the Day-2 gate; without
it the placeholder is used and the gate fails by design.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xsac.calibrate import calibrate, write_calibration
from xsac.config import CFG_M, CFG_S, CFG_TINY, L40S_RATE_PLACEHOLDER, TRAIN

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rate", type=float, default=None,
                    help="ACTUAL USD/hr for this machine")
    ap.add_argument("--n-runs", type=int, default=43)
    # Required, with no default. The factorial was voided because this was
    # invoked with --cost-ceiling 3.00 while the spec's figure is $56: the
    # solver correctly recorded affordable:false and the run proceeded anyway.
    # A silent default is what let a wrong number look like a considered one.
    ap.add_argument("--cost-ceiling", type=float, required=True,
                    help="REQUIRED. USD available for the training leg (the "
                         "spec's figure is $56). The budget solver sheds work "
                         "in priority order when it cannot afford the token "
                         "floor, and that decision must be based on money "
                         "that actually exists.")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    configs = ({"TINY": CFG_TINY} if args.smoke
               else {"S": CFG_S, "M": CFG_M})
    steps = 3 if args.smoke else args.steps
    micro = 2 if args.smoke else args.micro_batch
    rate = args.rate if args.rate is not None else L40S_RATE_PLACEHOLDER

    ceiling = args.cost_ceiling
    payload = calibrate(configs, TRAIN, args.n_runs, rate_usd_hr=rate,
                        steps=steps, micro_batch=micro, device=args.device,
                        rate_is_placeholder=args.rate is None,
                        cost_ceiling=ceiling)
    out = ROOT / "results" / ("calibration_smoke.json" if args.smoke
                              else "calibration.json")
    write_calibration(payload, out)
    print(json.dumps(payload, indent=2, default=str)[:3000])
    print("\nwrote {}".format(out.relative_to(ROOT)))

    # The decision this file exists to inform, stated plainly. A reader must
    # not have to find `affordable` inside 3000 characters of JSON.
    print("\n" + "=" * 68)
    print("BUDGET DECISION  (ceiling ${:.2f}, rate ${:.4f}/hr, {} runs)".format(
        ceiling, rate, args.n_runs))
    print("=" * 68)
    for size, entry in sorted((payload.get("sizes") or {}).items()):
        b = entry.get("budget") or {}
        tpr = b.get("tokens_per_run")
        floor_ok = tpr is not None and float(tpr) >= TRAIN.tokens_min
        spend = b.get("projected_spend_usd")
        print("  CFG_{}: tokens_per_run={}  projected_spend=${}  "
              "affordable={}".format(
                  size,
                  "{:.3g}".format(float(tpr)) if tpr else "none",
                  "{:.2f}".format(float(spend)) if spend is not None else "?",
                  b.get("affordable")))
        if not b.get("affordable") or not floor_ok:
            print("         NOT USABLE as the primary endpoint: "
                  "run_factorial.py will refuse to start.")
    print("=" * 68)
    if payload.get("rate_is_placeholder"):
        print("\n" + payload["warning"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
