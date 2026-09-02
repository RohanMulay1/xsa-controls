"""
xsac.config - model, training and experiment configuration.

Two model sizes exist for one reason, stated in the spec: "51M only" is the
single biggest reviewer attack. Two sizes lets us report whether the effect
grows, shrinks or holds.

Position encoding
-----------------
The spec gives parameter counts but never says which position encoding to use.
The counts settle it. With RoPE:

    CFG_S  50.93M total, 25.2M non-embedding   (spec: ~50.9M / ~25.2M)
    CFG_M 123.59M total, 84.9M non-embedding   (spec: ~124M  / ~85M)

Learned position embeddings would add 1024 * n_embd and give 51.45M / 124.37M,
which misses the stated totals. ``test_config.py`` pins both counts so this
cannot drift.

dropout is 0.0 deliberately. We are trying to resolve a ~0.001-0.015 nat
effect; adding stochastic regularisation to the measurement is the fastest way
to lose it in the noise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class ModelConfig:
    """A GPT configuration. Frozen: a mutable global config is how the CRPA
    campaign ended up with a routing experiment silently mutating and restoring
    a shared dict."""

    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    block_size: int = 1024
    vocab_size: int = 50304
    dropout: float = 0.0
    bias: bool = False
    tie_weights: bool = True
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        if self.n_embd % self.n_head:
            raise ValueError(
                "n_embd {} is not divisible by n_head {}".format(
                    self.n_embd, self.n_head))
        return self.n_embd // self.n_head

    def analytic_params(self) -> Tuple[int, int]:
        """(total, non_embedding) parameter counts, computed not measured.

        Returned analytically so a test can compare it against the built
        model. If the two disagree the model does not match its own config.
        """
        e, layers = self.n_embd, self.n_layer
        per_layer = (e * 3 * e) + (e * e) + (e * 4 * e) + (4 * e * e)
        if self.bias:
            per_layer += 3 * e + e + 4 * e + e
        non_embedding = per_layer * layers
        layernorms = 2 * e * layers + e
        if self.bias:
            layernorms *= 2
        total = self.vocab_size * e + non_embedding + layernorms
        if not self.tie_weights:
            total += self.vocab_size * e
        return total, non_embedding


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation settings shared by every arm.

    ``tokens_per_run`` is deliberately not final here. calibrate.py sets it
    from measured throughput against the actual GPU hourly rate. The default
    is the spec's fallback for when calibration has not run.
    """

    batch_tokens: int = 2 ** 17
    lr: float = 6e-4
    warmup_steps: int = 200
    schedule: str = "cosine_to_10pct"
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    betas: Tuple[float, float] = (0.9, 0.95)
    dtype: str = "bfloat16"
    eval_interval: int = 250
    eval_tokens: int = 2 ** 22
    tokens_per_run: float = 4.5e8
    micro_batch: int = 8

    #: Spec section 7 clamp. Outside this band the design is not affordable
    #: and the CFG_M scale check must be dropped explicitly, in writing.
    tokens_min: float = 3.5e8
    tokens_max: float = 6.0e8


@dataclass(frozen=True)
class ExperimentConfig:
    """A single cell of the factorial: one arm, one seed, one model size."""

    arm: str = "baseline"
    seed: int = 0
    size: str = "S"
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    smoke: bool = False

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["model"] = asdict(self.model)
        out["train"] = asdict(self.train)
        return out


#: Primary configuration. All five arms run here.
CFG_S = ModelConfig(n_layer=8, n_head=8, n_embd=512, block_size=1024,
                    vocab_size=50304, dropout=0.0, bias=False,
                    tie_weights=True)

#: Scale check. Three arms only (baseline / xsa / random).
CFG_M = ModelConfig(n_layer=12, n_head=12, n_embd=768, block_size=1024,
                    vocab_size=50304, dropout=0.0, bias=False,
                    tie_weights=True)

SIZES = {"S": CFG_S, "M": CFG_M}

TRAIN = TrainConfig()

#: A deliberately tiny model for CPU self-tests and smoke runs. Never used for
#: any reported number; every record it produces carries status "smoke".
CFG_TINY = ModelConfig(n_layer=2, n_head=4, n_embd=64, block_size=128,
                       vocab_size=512, dropout=0.0, bias=False,
                       tie_weights=True)

#: The five arms, in the order the paper reports them. ``random`` is the
#: pre-registered primary endpoint (spec section 8); it is not merely one of
#: four treatments.
ARMS = ("baseline", "xsa", "random", "meanval", "diagmask")
PRIMARY_ENDPOINT = "random"
SECONDARY_ARMS = ("xsa", "meanval", "diagmask")

#: Spec section 12. Recomputed against the real rate in calibrate.py.
COST_CEILING_TRAIN = 56.00
COST_CEILING_TOTAL = 70.00
COST_STOP_AND_REPORT = 66.00
L40S_RATE_PLACEHOLDER = 0.86
L40S_PEAK_TFLOPS_BF16 = 181.0


def size_config(size: str) -> ModelConfig:
    if size not in SIZES:
        raise KeyError("unknown size {!r}; expected one of {}".format(
            size, sorted(SIZES)))
    return SIZES[size]


def smoke_variant(cfg: ExperimentConfig) -> ExperimentConfig:
    """Shrink a config to something a CPU can finish in seconds.

    Used by --smoke everywhere. The resulting record is tagged ``smoke`` by
    the caller so it can never be mistaken for a measurement.
    """
    train = replace(cfg.train, tokens_per_run=8 * cfg.train.batch_tokens // 64,
                    batch_tokens=2 ** 10, warmup_steps=2, eval_interval=2,
                    eval_tokens=2 ** 11, micro_batch=2)
    return replace(cfg, model=CFG_TINY, train=train, smoke=True)
