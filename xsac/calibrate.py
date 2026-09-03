"""
xsac.calibrate - Day-1 throughput measurement and the token budget.

The token budget is not a constant. It is solved from measured throughput
against the **actual** hourly rate of the machine that will run the factorial,
because a budget computed against a placeholder rate is how a project
discovers on day 8 that it could only afford half its runs.

Procedure, per the spec:

1. Fifty steps of ``baseline`` and fifty of ``diagmask`` (the slowest arm) at
   both model sizes.
2. Report tokens/s, achieved TFLOP/s, and MFU against the 181 TF/s dense bf16
   peak of the L40S.
3. Solve the token budget against the real rate.
4. Round down to a multiple of ``batch_tokens`` and clamp to [3.5e8, 6e8].
5. If the clamp forces below 3.5e8, drop the CFG_M scale check first and
   record the decision, with its arithmetic, in BUDGET.md.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from xsac.config import (COST_CEILING_TRAIN, COST_STOP_AND_REPORT,
                         L40S_PEAK_TFLOPS_BF16,
                         L40S_RATE_PLACEHOLDER, ModelConfig, TrainConfig)
from xsac.model import GPT


def flops_per_token(cfg: ModelConfig) -> float:
    """Forward+backward FLOPs per token, the standard 6N + attention term.

    ``6 * N_non_embedding`` for the matmuls plus ``12 * L * T * d`` for the
    attention score and value products, which is not negligible at T=1024.
    """
    _, non_emb = cfg.analytic_params()
    return 6.0 * non_emb + 12.0 * cfg.n_layer * cfg.block_size * cfg.n_embd


def measure_throughput(cfg: ModelConfig, arm: str, steps: int = 50,
                       micro_batch: int = 4, device: Optional[str] = None,
                       warmup: int = 5, seed: int = 0) -> Dict[str, Any]:
    """Time a real forward+backward loop. Never an estimate.

    Uses random token ids: throughput does not depend on the data, and this
    keeps calibration runnable before tokenisation finishes.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available()
                                  else "cpu"))
    torch.manual_seed(seed)
    model = GPT(cfg, arm=arm).to(dev)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randint(0, cfg.vocab_size, (micro_batch, cfg.block_size),
                      device=dev)
    y = torch.randint(0, cfg.vocab_size, (micro_batch, cfg.block_size),
                      device=dev)
    use_amp = dev.type == "cuda"

    def one_step():
        opt.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)
        loss.backward()
        opt.step()

    for _ in range(warmup):
        one_step()
    if dev.type == "cuda":
        torch.cuda.synchronize()

    started = time.time()
    for _ in range(steps):
        one_step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - started

    tokens = steps * micro_batch * cfg.block_size
    tps = tokens / elapsed if elapsed > 0 else 0.0
    achieved = tps * flops_per_token(cfg) / 1e12
    return {
        "arm": arm, "device": str(dev), "steps": steps,
        "micro_batch": micro_batch, "seconds": elapsed,
        "seconds_per_step": elapsed / steps if steps else float("nan"),
        "tokens_per_sec": tps,
        "achieved_tflops": achieved,
        "mfu_vs_181": achieved / L40S_PEAK_TFLOPS_BF16,
        "n_params": model.num_params(),
    }


def solve_token_budget(tokens_per_sec: float, n_runs: int, rate_usd_hr: float,
                       train_cfg: TrainConfig,
                       cost_ceiling: float = COST_CEILING_TRAIN,
                       diagmask_slowdown: float = 1.0) -> Dict[str, Any]:
    """Tokens per run affordable at this throughput and this rate.

    ``diagmask_slowdown`` weights the slowest arm so the budget is not blown
    by the one arm that cannot use SDPA.
    """
    if tokens_per_sec <= 0 or n_runs <= 0 or rate_usd_hr <= 0:
        raise ValueError("throughput, run count and rate must all be positive")
    hours = cost_ceiling / rate_usd_hr
    effective_runs = n_runs * max(1.0, diagmask_slowdown)
    tokens_total = hours * 3600.0 * tokens_per_sec
    raw = tokens_total / effective_runs

    step = train_cfg.batch_tokens
    rounded = math.floor(raw / step) * step

    # Clamping UP to the floor does not make the plan affordable, it only
    # hides that it is not. When the budget cannot reach 3.5e8 tokens per run,
    # the spec's priority order says to shed work rather than shrink the
    # primary endpoint: drop the CFG_M scale check first (9 runs), then the
    # secondary arms (10 runs). Each cut is re-solved, so the returned budget
    # is one that can actually be paid for.
    cuts: List[Dict[str, Any]] = []
    effective_n = n_runs
    while rounded < train_cfg.tokens_min and cuts_available(cuts):
        cut = next_cut(cuts)
        effective_n = max(1, effective_n - cut["runs"])
        cuts.append(cut)
        raw = (hours * 3600.0 * tokens_per_sec) / (
            effective_n * max(1.0, diagmask_slowdown))
        rounded = math.floor(raw / step) * step

    affordable = rounded >= train_cfg.tokens_min
    clamped = min(max(rounded, train_cfg.tokens_min), train_cfg.tokens_max)
    clamped = math.floor(clamped / step) * step

    # What the plan will actually cost at the returned budget, so the $66
    # stop-and-report threshold can be enforced instead of merely declared.
    seconds_needed = (clamped * effective_n * max(1.0, diagmask_slowdown)
                      / tokens_per_sec)
    projected_usd = seconds_needed / 3600.0 * rate_usd_hr

    return {
        "hours_available": hours,
        "rate_usd_hr": rate_usd_hr,
        "cost_ceiling": cost_ceiling,
        "n_runs": n_runs,
        "n_runs_after_cuts": effective_n,
        "diagmask_slowdown": diagmask_slowdown,
        "tokens_per_run_raw": raw,
        "tokens_per_run_rounded": rounded,
        "tokens_per_run": float(clamped),
        "clamped_low": not affordable,
        "clamped_high": rounded > train_cfg.tokens_max,
        "affordable": bool(affordable),
        "cuts_applied": [c["name"] for c in cuts],
        "drop_cfg_m": any(c["name"] == "cfg_m_scale_check" for c in cuts),
        "drop_secondary_arms": any(
            c["name"] == "secondary_arms" for c in cuts),
        "projected_spend_usd": projected_usd,
        "over_stop_threshold": bool(projected_usd > COST_STOP_AND_REPORT),
        "stop_threshold_usd": COST_STOP_AND_REPORT,
        "note": _budget_note(affordable, cuts, projected_usd),
    }


#: Priority order from spec section 12: cut from the bottom, and never cut the
#: primary endpoint below 8 seeds or drop A1.
CUT_ORDER = [
    {"name": "cfg_m_scale_check", "runs": 9,
     "why": "the scale check is the first thing to go; the primary endpoint "
            "and A1 are the paper"},
    {"name": "secondary_arms", "runs": 10,
     "why": "meanval and diagmask are secondary; the pre-registered primary "
            "endpoint keeps its 8 seeds"},
]


def cuts_available(cuts: Sequence[Dict[str, Any]]) -> bool:
    return len(cuts) < len(CUT_ORDER)


def next_cut(cuts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return dict(CUT_ORDER[len(cuts)])


def _budget_note(affordable: bool, cuts: Sequence[Dict[str, Any]],
                 projected: float) -> str:
    bits = []
    if cuts:
        bits.append("budget did not reach the 3.5e8 floor, so {} were dropped "
                    "and the budget re-solved".format(
                        " then ".join(c["name"] for c in cuts)))
    if not affordable:
        bits.append("STILL below the floor after every permitted cut. The "
                    "design is not affordable at this rate; stop and report "
                    "rather than shrinking the primary endpoint")
    if projected > COST_STOP_AND_REPORT:
        bits.append("projected spend ${:.2f} exceeds the ${:.2f} "
                    "stop-and-report threshold".format(
                        projected, COST_STOP_AND_REPORT))
    return "; ".join(bits)


def calibrate(configs: Dict[str, ModelConfig], train_cfg: TrainConfig,
              n_runs: int, rate_usd_hr: float = L40S_RATE_PLACEHOLDER,
              steps: int = 50, micro_batch: int = 4,
              device: Optional[str] = None,
              rate_is_placeholder: bool = True,
              cost_ceiling: float = COST_CEILING_TRAIN) -> Dict[str, Any]:
    """Full Day-2 calibration for every model size."""
    out: Dict[str, Any] = {"rate_usd_hr": rate_usd_hr,
                           "rate_is_placeholder": bool(rate_is_placeholder),
                           "cost_ceiling": cost_ceiling,
                           "sizes": {}}
    if rate_is_placeholder:
        out["warning"] = (
            "TOKENS_PER_RUN was computed against the $%.2f/hr PLACEHOLDER, "
            "not a measured rate. The Day-2 gate requires the actual rate. "
            "Re-run with --rate before spending." % rate_usd_hr)

    for name, cfg in configs.items():
        base = measure_throughput(cfg, "baseline", steps=steps,
                                  micro_batch=micro_batch, device=device)
        diag = measure_throughput(cfg, "diagmask", steps=steps,
                                  micro_batch=micro_batch, device=device)
        slowdown = (diag["seconds_per_step"] / base["seconds_per_step"]
                    if base["seconds_per_step"] > 0 else float("nan"))
        budget = solve_token_budget(base["tokens_per_sec"], n_runs,
                                    rate_usd_hr, train_cfg,
                                    cost_ceiling=cost_ceiling,
                                    diagmask_slowdown=slowdown)
        out["sizes"][name] = {
            "config": asdict(cfg),
            "baseline": base,
            "diagmask": diag,
            "diagmask_slowdown": slowdown,
            # The spec expects 1.5-2.0x. Outside that band something is wrong
            # with the explicit-attention path and it should be investigated
            # rather than accepted.
            "slowdown_in_expected_band": bool(1.4 <= slowdown <= 2.2)
            if slowdown == slowdown else False,
            "budget": budget,
        }
    return out


def write_calibration(payload: Dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str),
                    encoding="utf-8")
    return path
