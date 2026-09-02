"""
xsac.train - a single paired run.

The paired protocol, restated because deviating from it silently invalidates
every downstream test:

* Identical init across arms. ``torch.manual_seed(seed)`` immediately before
  model construction, so the shared parameters get identical values regardless
  of arm. The arms differ only in the extra ``alpha`` vector, which is
  zero-initialised and therefore adds no randomness.
* Identical data order. The loader is seeded by the seed only.
* Identical LR schedule, step count and eval batches.

The only difference between two runs at the same seed is the arm. Self-test 10
verifies this holds at step 0 to within 1e-6, and the Days 4-6 gate verifies
every arm at a given seed saw the same ``tokens_seen``.
"""

from __future__ import annotations

import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from xsac.config import ExperimentConfig, ModelConfig, TrainConfig
from xsac.data import FixedEvalLoader, PairedLoader, TokenDataset
from xsac.model import GPT


def set_seed(seed: int) -> None:
    """Seed every generator we touch, including CUDA.

    Incomplete seeding is one of the defects this project exists to avoid.
    Note that some CUDA kernels remain nondeterministic even after this; the
    paired design tolerates that because both arms of a pair see the same
    kernels, and self-test 9 pins bit-identity on CPU where it is achievable.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lr_at(step: int, total_steps: int, cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay to 10% of peak."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    if total_steps <= cfg.warmup_steps:
        return cfg.lr * 0.1
    progress = (step - cfg.warmup_steps) / (total_steps - cfg.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def build_model(model_cfg: ModelConfig, arm: str, seed: int,
                device: torch.device) -> GPT:
    """Construct a model with init that does not depend on the arm."""
    set_seed(seed)
    model = GPT(model_cfg, arm=arm)
    return model.to(device)


@torch.no_grad()
def evaluate(model: GPT, loader: FixedEvalLoader) -> float:
    """Mean token cross-entropy over the fixed validation sweep.

    This is the ONLY quantity reported as loss or perplexity. No auxiliary
    term is ever folded in: mixing objectives into a reported metric inflates
    treatment arms relative to controls by construction, which is exactly the
    defect this project was written to avoid repeating.
    """
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        _, loss = model(x, y)
        total += float(loss) * x.shape[0]
        n += x.shape[0]
    if was_training:
        model.train()
    return total / max(1, n)


def train_one(cfg: ExperimentConfig, data_dir: Path,
              device: Optional[str] = None,
              log_every: int = 0,
              max_steps: Optional[int] = None) -> Dict[str, Any]:
    """Run one cell of the factorial and return its metrics.

    Returns a plain dict. The caller wraps it in a RunRecord with a status, so
    that a crash here surfaces as ``failed`` rather than as a missing file.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available()
                                  else "cpu"))
    train_cfg = cfg.train
    model_cfg = cfg.model

    train_ds = TokenDataset(Path(data_dir) / "train.bin")
    val_ds = TokenDataset(Path(data_dir) / "val.bin")

    tokens_per_step = train_cfg.batch_tokens
    total_steps = max(1, int(train_cfg.tokens_per_run // tokens_per_step))
    if max_steps is not None:
        total_steps = min(total_steps, int(max_steps))

    micro = train_cfg.micro_batch
    tokens_per_micro = micro * model_cfg.block_size
    accum = max(1, tokens_per_step // max(1, tokens_per_micro))

    model = build_model(model_cfg, cfg.arm, cfg.seed, dev)
    optimiser = model.configure_optimizer(train_cfg)
    # Seeded by the seed alone. Never by the arm: that is the pairing.
    loader = PairedLoader(train_ds, model_cfg.block_size, micro, cfg.seed,
                          device=str(dev))
    val_loader = FixedEvalLoader(val_ds, model_cfg.block_size, micro,
                                 train_cfg.eval_tokens, device=str(dev))

    use_amp = dev.type == "cuda" and train_cfg.dtype == "bfloat16"
    started = time.time()
    tokens_seen = 0
    history: List[Dict[str, float]] = []
    loss_at_step0 = float("nan")

    model.train()
    for step in range(total_steps):
        lr = lr_at(step, total_steps, train_cfg)
        for group in optimiser.param_groups:
            group["lr"] = lr

        optimiser.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(accum):
            x, y = loader.batch()
            if use_amp:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)
            else:
                _, loss = model(x, y)
            (loss / accum).backward()
            step_loss += float(loss) / accum
            tokens_seen += x.numel()

        if step == 0:
            loss_at_step0 = step_loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimiser.step()

        if log_every and step % log_every == 0:
            history.append({"step": step, "loss": step_loss, "lr": lr})

    final_val = evaluate(model, val_loader)
    duration = time.time() - started

    return {
        "final_val_loss": final_val,
        "final_val_ppl": float(math.exp(min(20.0, final_val))),
        "loss_at_step0": loss_at_step0,
        "tokens_seen": tokens_seen,
        "steps": total_steps,
        "wall_seconds": duration,
        "tokens_per_sec": tokens_seen / duration if duration > 0 else 0.0,
        "learned_alpha": model.gate_table(),
        "n_params": model.num_params(),
        "n_params_non_embedding": model.num_params(non_embedding=True),
        "device": str(dev),
        "arm": cfg.arm,
        "seed": cfg.seed,
        "size": cfg.size,
        "history": history,
    }


def step0_losses(arms: List[str], model_cfg: ModelConfig, seed: int,
                 data_dir: Path, micro_batch: int = 2,
                 device: str = "cpu") -> Dict[str, float]:
    """Loss at step 0 for each arm at one seed, on identical data.

    This is self-test 10, exposed as a function so the test suite and the
    Day-1 gate script call exactly the same code path.
    """
    train_ds = TokenDataset(Path(data_dir) / "train.bin")
    out: Dict[str, float] = {}
    for arm in arms:
        model = build_model(model_cfg, arm, seed, torch.device(device))
        model.eval()
        loader = PairedLoader(train_ds, model_cfg.block_size, micro_batch,
                              seed, device=device)
        x, y = loader.batch()
        with torch.no_grad():
            _, loss = model(x, y)
        out[arm] = float(loss)
    return out
