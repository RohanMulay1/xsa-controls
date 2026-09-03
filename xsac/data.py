"""
xsac.data - uint16 memmap token loaders with a strict pairing guarantee.

The paired protocol lives or dies here. For a given seed every arm must see
**identical** batches in an identical order, because the only difference
between two runs at the same seed is meant to be the arm. The loader is
therefore seeded by the seed alone and never by the arm name, and
``test_data.py`` asserts that two loaders built with different arms but the
same seed emit bit-identical batches.

Validation batches are stronger still: they are a fixed, deterministic sweep
over a disjoint holdout, identical for every run, arm and seed. Sampling them
randomly would add between-run variance to the exact quantity we are trying to
resolve at the 0.001 nat level.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch


class TokenDataset:
    """A uint16 memmap of token ids."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                "{} does not exist. Run: python data/prepare.py".format(path))
        size = self.path.stat().st_size
        if size % 2:
            raise ValueError(
                "{} has an odd byte count ({}); a uint16 memmap must be even. "
                "The file is truncated or was written with the wrong dtype."
                .format(path, size))
        self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    def verify_size(self, n_tokens: int) -> bool:
        """Day-1 gate: file bytes must equal 2 * n_tokens exactly."""
        return self.path.stat().st_size == 2 * int(n_tokens)


class PairedLoader:
    """Training batches. Seeded by the run seed only, never by the arm.

    Sampling is with replacement from random offsets, which is the nanoGPT
    convention and keeps the stream independent of dataset length so the same
    seed gives the same batches across machines.
    """

    def __init__(self, dataset: TokenDataset, block_size: int,
                 micro_batch: int, seed: int, device: str = "cpu"):
        if len(dataset) <= block_size + 1:
            raise ValueError(
                "dataset has {} tokens, needs more than block_size+1 = {}"
                .format(len(dataset), block_size + 1))
        self.ds = dataset
        self.block_size = block_size
        self.micro_batch = micro_batch
        self.seed = int(seed)
        self.device = device
        # A dedicated generator, so unrelated torch/numpy consumers elsewhere
        # in the process cannot shift the data order and break the pairing.
        self.rng = np.random.default_rng(self.seed)

    def batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        hi = len(self.ds) - self.block_size - 1
        idx = self.rng.integers(0, hi, size=self.micro_batch)
        toks = self.ds.tokens
        x = np.stack([toks[i:i + self.block_size].astype(np.int64)
                      for i in idx])
        y = np.stack([toks[i + 1:i + 1 + self.block_size].astype(np.int64)
                      for i in idx])
        return (torch.from_numpy(x).to(self.device),
                torch.from_numpy(y).to(self.device))

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)


class FixedEvalLoader:
    """Deterministic, contiguous, non-overlapping validation batches.

    Identical for every run, arm and seed. There is no randomness here at all,
    which is the point: shared eval noise would sit directly on top of the
    effect we are trying to measure.
    """

    def __init__(self, dataset: TokenDataset, block_size: int,
                 micro_batch: int, max_tokens: int, device: str = "cpu"):
        self.ds = dataset
        self.block_size = block_size
        self.micro_batch = micro_batch
        self.device = device
        usable = (len(dataset) - 1) // block_size
        wanted = max(1, int(max_tokens) // block_size)
        self.n_sequences = max(1, min(usable, wanted))

    @property
    def n_batches(self) -> int:
        return max(1, self.n_sequences // self.micro_batch)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        toks = self.ds.tokens
        seq = 0
        for _ in range(self.n_batches):
            xs, ys = [], []
            for _ in range(self.micro_batch):
                start = seq * self.block_size
                xs.append(toks[start:start + self.block_size].astype(np.int64))
                ys.append(toks[start + 1:start + 1 + self.block_size]
                          .astype(np.int64))
                seq += 1
            yield (torch.from_numpy(np.stack(xs)).to(self.device),
                   torch.from_numpy(np.stack(ys)).to(self.device))


def synthetic_tokens(path: Path, n_tokens: int, vocab_size: int,
                     seed: int = 0) -> Path:
    """Write a deterministic synthetic uint16 memmap.

    For CPU self-tests and smoke runs only. Every record produced from it is
    tagged ``smoke``, so it can never reach a reported table. It exists so the
    whole pipeline is runnable and testable without a 700M-token download.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    # A little structure so the loss can actually fall: a low-order Markov
    # chain rather than uniform noise, which no model can beat.
    base = rng.integers(0, vocab_size, size=n_tokens, dtype=np.uint16)
    out = base.copy()
    out[1:] = np.where(rng.random(n_tokens - 1) < 0.35,
                       (base[:-1] + 1) % vocab_size, base[1:])
    arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(n_tokens,))
    arr[:] = out
    arr.flush()
    del arr
    return path


def load_split(data_dir: Path, split: str) -> TokenDataset:
    return TokenDataset(Path(data_dir) / "{}.bin".format(split))


def ensure_smoke_data(data_dir: Path, vocab_size: int,
                      n_train: int = 200_000,
                      n_val: int = 40_000) -> Tuple[Path, Path]:
    """Create small synthetic train/val files if they are absent.

    The two splits use different generator seeds, so they are disjoint by
    construction rather than by an overlap check that could silently pass.
    """
    data_dir = Path(data_dir)
    train = data_dir / "train.bin"
    val = data_dir / "val.bin"
    if not train.exists():
        synthetic_tokens(train, n_train, vocab_size, seed=1234)
    if not val.exists():
        synthetic_tokens(val, n_val, vocab_size, seed=99999)
    return train, val
