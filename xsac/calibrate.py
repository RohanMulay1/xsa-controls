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
from typing import Any, Dict, List, Optional

import torch

from xsac.config import (COST_CEILING_TRAIN, L40S_PEAK_TFLOPS_BF16,
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
    clamped = min(max(rounded, train_cfg.tokens_min), train_cfg.tokens_max)
    clamped = math.floor(clamped / step) * step

    return {
        "hours_available": hours,
        "rate_usd_hr": rate_usd_hr,
        "cost_ceiling": cost_ceiling,
        "n_runs": n_runs,
        "diagmask_slowdown": diagmask_slowdown,
        "tokens_per_run_raw": raw,
        "tokens_per_run_rounded": rounded,
        "tokens_per_run": float(clamped),
        "clamped_low": rounded < train_cfg.tokens_min,
        "clamped_high": rounded > train_cfg.tokens_max,
        # If the clamp binds low, the CFG_M scale check is dropped FIRST and
        # the decision goes into BUDGET.md with its arithmetic. Cutting the
        # primary endpoint's seeds instead would be cutting the paper.
        "drop_cfg_m": bool(rounded < train_cfg.tokens_min),
        "note": ("budget forces below the 3.5e8 floor: drop the CFG_M scale "
                 "check and put the time into CFG_S seeds"
                 if rounded < train_cfg.tokens_min else ""),
    }


def calibrate(configs: Dict[str, ModelConfig], train_cfg: TrainConfig,
              n_runs: int, rate_usd_hr: float = L40S_RATE_PLACEHOLDER,
              steps: int = 50, micro_batch: int = 4,
              device: Optional[str] = None,
              rate_is_placeholder: bool = True) -> Dict[str, Any]:
    """Full Day-2 calibration for every model size."""
    out: Dict[str, Any] = {"rate_usd_hr": rate_usd_hr,
                           "rate_is_placeholder": bool(rate_is_placeholder),
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
