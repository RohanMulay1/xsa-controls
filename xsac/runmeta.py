"""
xsac.runmeta - run records, status handling and resumability.

Ported from the CRPA campaign, where this shape is what made a multi-day
unattended factorial survive interruption. Three properties matter:

* **The filename is a content hash of the config.** Resumability is then free:
  if the hash file exists, the cell is done. No separate ledger to fall out of
  sync.
* **Status is an enum, and ``numeric_records`` is the only way to get records
  for aggregation or plotting.** A status column that any caller may read past
  is decoration. Making the filtered accessor the sole path is what turns
  "an OOM never becomes a number" from a habit into a structural property.
* **Writes are atomic.** Temp file plus ``os.replace``, so a kill during a
  write leaves the previous record rather than a truncated one.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import socket
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

#: Only these two statuses may contribute a number to a table or a figure.
NUMERIC_STATUSES = ("completed", "smoke")

#: Everything a run can be. ``not_run`` is deliberately a first-class value:
#: an experiment that was specified but never executed must be visible as
#: absent rather than missing from the file entirely.
STATUSES = ("completed", "smoke", "not_run", "oom", "unsupported", "failed")


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def run_id(config: Dict[str, Any], seed: int) -> str:
    """Deterministic 12-hex id from the config content and the seed."""
    payload = canonical_json(config) + "|seed={}".format(seed)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.run(["git", "status", "--porcelain"],
                             capture_output=True, text=True, timeout=5)
        return bool(out.stdout.strip())
    except Exception:
        return False


def environment() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["cuda"] = torch.version.cuda
            env["bf16"] = bool(torch.cuda.is_bf16_supported())
        else:
            env["gpu"] = "cpu"
    except Exception:
        env["torch"] = "unavailable"
    return env


@dataclass
class RunRecord:
    """One cell of one experiment, with everything needed to trust it."""

    run_id: str
    experiment: str
    status: str
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    arm: str = ""
    seed: int = 0
    size: str = ""
    note: str = ""
    error: str = ""
    duration_s: float = 0.0
    timestamp: str = ""
    git_sha: str = ""
    git_dirty: bool = False
    env: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(
                "unknown status {!r}; expected one of {}".format(
                    self.status, STATUSES))
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        if not self.git_sha:
            self.git_sha = _git_sha()
            self.git_dirty = _git_dirty()
        if not self.env:
            self.env = environment()

    @property
    def is_numeric(self) -> bool:
        return self.status in NUMERIC_STATUSES

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def write_record(record: RunRecord, results_dir: Path) -> Path:
    """Atomically write one record as ``runs/<run_id>.json``."""
    runs = Path(results_dir) / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / "{}.json".format(record.run_id)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record.to_dict(), indent=2, default=str),
                   encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_records(results_dir: Path) -> List[RunRecord]:
    runs = Path(results_dir) / "runs"
    if not runs.exists():
        return []
    out: List[RunRecord] = []
    for path in sorted(runs.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        known = {k: v for k, v in data.items()
                 if k in RunRecord.__dataclass_fields__}
        try:
            out.append(RunRecord(**known))
        except Exception:
            continue
    return out


def numeric_records(records: Iterable[RunRecord]) -> List[RunRecord]:
    """The ONLY accessor that may feed a table, an aggregate or a figure.

    An OOM, a failure or an unexecuted cell has no number to contribute. Going
    through this function is what makes that structural instead of a rule
    someone has to remember.
    """
    return [r for r in records if r.is_numeric]


def is_done(results_dir: Path, rid: str) -> bool:
    """Resumability: a completed cell is one whose hash file already exists."""
    path = Path(results_dir) / "runs" / "{}.json".format(rid)
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return data.get("status") in NUMERIC_STATUSES


def write_csv(rows: List[Dict[str, Any]], path: Path) -> Optional[Path]:
    """Write rows to CSV with a union-of-keys header. Never hand-edited."""
    if not rows:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)
    return path


def records_to_rows(records: Iterable[RunRecord]) -> List[Dict[str, Any]]:
    """Flatten records for a CSV. Status is always carried through."""
    rows: List[Dict[str, Any]] = []
    for r in records:
        row: Dict[str, Any] = {
            "run_id": r.run_id, "experiment": r.experiment, "arm": r.arm,
            "seed": r.seed, "size": r.size, "status": r.status,
            "duration_s": r.duration_s, "note": r.note,
            "git_sha": r.git_sha,
        }
        for k, v in r.metrics.items():
            row[k] = canonical_json(v) if isinstance(v, (dict, list)) else v
        rows.append(row)
    return rows
