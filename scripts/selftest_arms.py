"""
The ten self-tests. All must pass before any GPU spend.

CPU only, tiny random model, no downloads. Exits 0 and prints "10/10 PASS"
only when every one of them passes.

These are not unit tests in the ordinary sense. Each one catches a specific
silent failure that would invalidate the experiment rather than crash it:

  * If test 10 fails, the arms do not start from a common point, and the
    entire paired statistical design is void. Do not proceed.
  * If test 4 fails, the "fixed arbitrary direction" control is actually a
    resampled-direction control, which is a different hypothesis.
  * If test 5 or 6 fails, MeanValue differs from RandomDir in optimisation as
    well as in direction, and the comparison is confounded.

Marking any of these skipped, xfail, or loosening a tolerance to make one pass
is forbidden by the project's own anti-pattern table. If test 2 needs 1e-3
instead of 1e-4, the projection is wrong; investigate it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xsac.arms import MeanValue, RandomDir, XSA, build_arm  # noqa: E402
from xsac.config import ARMS, CFG_TINY  # noqa: E402
from xsac.model import GPT  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"

B, H, T, DH = 2, 4, 16, 8


def _yv(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    y = torch.randn(B, H, T, DH, generator=g)
    v = torch.randn(B, H, T, DH, generator=g)
    return y, v


class Report:
    def __init__(self) -> None:
        self.rows = []

    def add(self, num: int, name: str, ok: bool, detail: str) -> None:
        self.rows.append((num, name, bool(ok), detail))
        print("  [{}] {:2d}. {:<34s} {}".format(
            "PASS" if ok else "FAIL", num, name, detail))

    @property
    def n_pass(self) -> int:
        return sum(1 for _, _, ok, _ in self.rows if ok)

    @property
    def all_pass(self) -> bool:
        return all(ok for _, _, ok, _ in self.rows)


def test_1_zero_init_identity(rep: Report) -> None:
    """Every arm with alpha=0 returns y exactly."""
    y, v = _yv(1)
    worst, worst_arm = 0.0, ""
    for name in ("baseline", "xsa", "random", "meanval"):
        arm = build_arm(name, H, DH, layer_idx=0)
        arm.eval()
        out = arm(y, v)
        err = float((out - y).abs().max())
        if err > worst:
            worst, worst_arm = err, name
    rep.add(1, "Zero-init identity", worst < 1e-6,
            "max |z - y| = {:.3e} over 4 arms (worst: {})".format(
                worst, worst_arm))


def test_2_full_strength_orthogonality(rep: Report) -> None:
    """At alpha=10 (tanh~1) the removed direction is gone from the output."""
    y, v = _yv(2)

    xsa = XSA(H)
    with torch.no_grad():
        xsa.alpha.fill_(10.0)
    z = xsa(y, v)
    vhat = v / v.norm(dim=-1, keepdim=True)
    xsa_err = float((z * vhat).sum(-1).abs().max())

    rnd = RandomDir(H, DH, layer_idx=0)
    with torch.no_grad():
        rnd.alpha.fill_(10.0)
    z2 = rnd(y, v)
    rhat = rnd.r / rnd.r.norm(dim=-1, keepdim=True)
    rnd_err = float((z2 * rhat).sum(-1).abs().max())

    ok = xsa_err < 1e-4 and rnd_err < 1e-4
    rep.add(2, "Full-strength orthogonality", ok,
            "|<z,v_hat>| = {:.3e}, |<z,r_hat>| = {:.3e}".format(
                xsa_err, rnd_err))


def test_3_position_zero_degeneracy(rep: Report) -> None:
    """a_00 = 1, so y_0 = v_0 exactly, so XSA's z_0 = 0 identically.

    The measured ||z_0|| is recorded to results/position0.txt because it goes
    in the paper: XSA never mentions this degeneracy, and its own Figure 1
    restricts the diagonal panel to i > 1, so the author knew about it in the
    diagnostic but left it in the method.
    """
    cfg = CFG_TINY
    torch.manual_seed(0)
    model = GPT(cfg, arm="xsa")
    for block in model.h:
        with torch.no_grad():
            block.attn.arm.alpha.fill_(10.0)
    model.eval()

    idx = torch.randint(0, cfg.vocab_size, (1, 24))
    caps = [dict() for _ in range(cfg.n_layer)]
    with torch.no_grad():
        model(idx, captures=caps)

    a00 = float(caps[0]["att"][0, :, 0, 0].min())
    z0 = float(caps[0]["y"][0, :, 0, :].norm(dim=-1).max())

    # y_0 = v_0 check on a baseline model, where no arm has touched y.
    torch.manual_seed(0)
    base = GPT(cfg, arm="baseline")
    base.eval()
    bcaps = [dict() for _ in range(cfg.n_layer)]
    with torch.no_grad():
        base(idx, captures=bcaps)
    y0v0 = float((bcaps[0]["y"][0, :, 0, :]
                  - bcaps[0]["v"][0, :, 0, :]).norm(dim=-1).max())

    ok = abs(a00 - 1.0) < 1e-6 and y0v0 < 1e-5 and z0 < 1e-5
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "position0.txt").write_text(
        "Position-0 degeneracy, measured on CFG_TINY (self-test 3)\n"
        "=========================================================\n"
        "a_00                       = {:.10f}   (exactly 1 by construction:\n"
        "                                        causal softmax over one key)\n"
        "max ||y_0 - v_0||          = {:.3e}\n"
        "max ||z_0|| with XSA a=10  = {:.3e}\n\n"
        "Consequence: XSA's intervention is identically zero at position 0 in\n"
        "every layer and every head. The method cannot act there. XSA does not\n"
        "mention this; its Figure 1 restricts the diagonal panel to i > 1.\n"
        "The paper must state it, and diagmask must exclude position 0 or the\n"
        "softmax row becomes all -inf and produces NaN.\n".format(
            a00, y0v0, z0), encoding="utf-8")
    rep.add(3, "Position-0 degeneracy", ok,
            "a_00 = {:.6f}, ||y_0-v_0|| = {:.2e}, ||z_0|| = {:.2e} "
            "-> results/position0.txt".format(a00, y0v0, z0))


def test_4_randomdir_frozen(rep: Report) -> None:
    """r is not updated by the optimiser and does not depend on the run seed."""
    y, v = _yv(4)
    arm = RandomDir(H, DH, layer_idx=3)
    before = arm.r.clone()
    opt = torch.optim.SGD(arm.parameters(), lr=1.0)
    with torch.no_grad():
        arm.alpha.fill_(0.5)
    loss = arm(y, v).pow(2).mean()
    loss.backward()
    opt.step()
    unchanged = float((arm.r - before).abs().max())

    torch.manual_seed(11111)
    a = RandomDir(H, DH, layer_idx=3)
    torch.manual_seed(22222)
    b = RandomDir(H, DH, layer_idx=3)
    same_across_seeds = float((a.r - b.r).abs().max())

    c = RandomDir(H, DH, layer_idx=4)
    differs_across_layers = float((a.r - c.r).abs().max())

    ok = (unchanged == 0.0 and same_across_seeds == 0.0
          and differs_across_layers > 1e-3)
    rep.add(4, "RandomDir frozen", ok,
            "delta after step = {:.1e}, across seeds = {:.1e}, "
            "across layers = {:.3f}".format(
                unchanged, same_across_seeds, differs_across_layers))


def test_5_meanvalue_no_gradient_leak(rep: Report) -> None:
    """m carries no gradient; it moves only through the EMA."""
    y, v = _yv(5)
    arm = MeanValue(H, DH)
    arm.train()
    with torch.no_grad():
        arm.alpha.fill_(0.3)
    out = arm(y, v)
    out.pow(2).mean().backward()
    no_grad = arm.m.grad is None
    alpha_has_grad = arm.alpha.grad is not None and bool(
        torch.isfinite(arm.alpha.grad).all())

    before = arm.m.clone()
    y2, v2 = _yv(6)
    arm(y2, v2)
    moved = float((arm.m - before).abs().max())

    ok = no_grad and alpha_has_grad and moved > 0.0
    rep.add(5, "MeanValue no gradient leak", ok,
            "m.grad is None = {}, EMA moved m by {:.3e}".format(
                no_grad, moved))


def test_6_meanvalue_frozen_in_eval(rep: Report) -> None:
    """In eval mode two forwards with different v leave m untouched."""
    arm = MeanValue(H, DH)
    arm.train()
    y, v = _yv(7)
    arm(y, v)                       # initialise the buffer
    arm.eval()
    before = arm.m.clone()
    arm(*_yv(8))
    arm(*_yv(9))
    moved = float((arm.m - before).abs().max())
    rep.add(6, "MeanValue frozen in eval", moved == 0.0,
            "max |m_after - m_before| = {:.1e}".format(moved))


def test_7_diagmask_row_zero_valid(rep: Report) -> None:
    """No NaN or Inf anywhere, and att[...,0,0] == 1 exactly.

    Checked on BOTH diagmask paths, because position 0 is exactly where a
    naive implementation produces an all -inf row and NaN. The gated path is
    also checked at full gate strength, which is the only setting where the
    diagonal is actually suppressed: at alpha = 0 the arm is the baseline by
    design, so asserting a zero diagonal there would be asserting the gate
    does not work.
    """
    cfg = CFG_TINY
    idx = torch.randint(0, cfg.vocab_size, (2, 20),
                        generator=torch.Generator().manual_seed(3))
    detail, ok = [], True

    for hard in (False, True):
        torch.manual_seed(0)
        model = GPT(cfg, arm="diagmask", diagmask_hard=hard)
        if not hard:
            for block in model.h:
                with torch.no_grad():
                    block.attn.diag_alpha.fill_(10.0)
        model.eval()
        caps = [dict() for _ in range(cfg.n_layer)]
        with torch.no_grad():
            logits, _ = model(idx, captures=caps)

        att = caps[0]["att"]
        finite = bool(torch.isfinite(att).all()) and bool(
            torch.isfinite(logits).all())
        a00 = float(att[..., 0, 0].min())
        rows_sum_to_one = float(
            (att.sum(dim=-1) - 1.0).abs().max())
        diag = att[..., torch.arange(1, 20), torch.arange(1, 20)]
        diag_max = float(diag.abs().max())

        path_ok = (finite and abs(a00 - 1.0) < 1e-6
                   and rows_sum_to_one < 1e-5 and diag_max < 1e-9)
        ok = ok and path_ok
        detail.append("{}: finite={} att[0,0]={:.6f} diag={:.1e}".format(
            "hard" if hard else "gated@a=10", finite, a00, diag_max))

    rep.add(7, "DiagMask row 0 valid", ok, "; ".join(detail))


def test_8_gradient_reaches_alpha(rep: Report) -> None:
    """alpha.grad is finite and non-zero for arms 2-4."""
    worst = []
    ok = True
    for name in ("xsa", "random", "meanval"):
        cfg = CFG_TINY
        torch.manual_seed(0)
        model = GPT(cfg, arm=name)
        model.train()
        idx = torch.randint(0, cfg.vocab_size, (2, 16))
        tgt = torch.randint(0, cfg.vocab_size, (2, 16))
        _, loss = model(idx, tgt)
        loss.backward()
        grads = [b.attn.arm.alpha.grad for b in model.h]
        finite = all(g is not None and bool(torch.isfinite(g).all())
                     for g in grads)
        nonzero = max(float(g.abs().max()) for g in grads)
        worst.append("{}={:.2e}".format(name, nonzero))
        ok = ok and finite and nonzero > 0
    rep.add(8, "Gradient reaches alpha", ok,
            "max |dL/dalpha|: " + ", ".join(worst))


def test_9_paired_determinism(rep: Report) -> None:
    """Same arm, same seed gives a bit-identical loss at steps 0 and 1."""
    torch.use_deterministic_algorithms(True)
    try:
        losses = []
        for _ in range(2):
            torch.manual_seed(4242)
            model = GPT(CFG_TINY, arm="xsa")
            model.train()
            opt = torch.optim.SGD(model.parameters(), lr=0.01)
            g = torch.Generator().manual_seed(7)
            seq = [(torch.randint(0, CFG_TINY.vocab_size, (2, 16),
                                  generator=g),
                    torch.randint(0, CFG_TINY.vocab_size, (2, 16),
                                  generator=g)) for _ in range(2)]
            run = []
            for x, y in seq:
                opt.zero_grad(set_to_none=True)
                _, loss = model(x, y)
                run.append(float(loss))
                loss.backward()
                opt.step()
            losses.append(run)
        identical = losses[0] == losses[1]
        detail = "step0 {:.12f} / step1 {:.12f}, bit-identical = {}".format(
            losses[0][0], losses[0][1], identical)
    finally:
        torch.use_deterministic_algorithms(False)
    rep.add(9, "Paired determinism", identical, detail)


def test_10_cross_arm_pairing(rep: Report) -> None:
    """At step 0 every arm's loss equals baseline's to <1e-6.

    If this fails the entire statistical design is void: the arms would not be
    starting from a common point, so a paired difference would confound the
    intervention with a different initialisation.
    """
    cfg = CFG_TINY
    seed = 1234
    g = torch.Generator().manual_seed(99)
    x = torch.randint(0, cfg.vocab_size, (2, 24), generator=g)
    y = torch.randint(0, cfg.vocab_size, (2, 24), generator=g)

    losses = {}
    for arm in ARMS:
        torch.manual_seed(seed)
        model = GPT(cfg, arm=arm)
        model.eval()
        with torch.no_grad():
            _, loss = model(x, y)
        losses[arm] = float(loss)

    base = losses["baseline"]
    worst = max(abs(v - base) for v in losses.values())
    ok = worst < 1e-6
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "step0_pairing.json").write_text(
        json.dumps({"losses": losses, "max_abs_deviation": worst,
                    "tolerance": 1e-6, "passed": ok}, indent=2),
        encoding="utf-8")
    rep.add(10, "Cross-arm pairing at step 0", ok,
            "max |loss_arm - loss_baseline| = {:.3e} over {} arms".format(
                worst, len(ARMS)))


TESTS = (test_1_zero_init_identity, test_2_full_strength_orthogonality,
         test_3_position_zero_degeneracy, test_4_randomdir_frozen,
         test_5_meanvalue_no_gradient_leak, test_6_meanvalue_frozen_in_eval,
         test_7_diagmask_row_zero_valid, test_8_gradient_reaches_alpha,
         test_9_paired_determinism, test_10_cross_arm_pairing)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=str, default="",
                        help="also write the report to this path")
    args = parser.parse_args(argv)

    print("xsa-controls self-tests (CPU, no downloads)")
    print("=" * 62)
    rep = Report()
    for test in TESTS:
        try:
            test(rep)
        except Exception as exc:  # a crashing test is a failing test
            n = int(test.__name__.split("_")[1])
            rep.add(n, test.__name__, False, "raised {}: {}".format(
                type(exc).__name__, exc))
    print("=" * 62)
    print("{}/{} {}".format(rep.n_pass, len(TESTS),
                            "PASS" if rep.all_pass else "FAIL"))

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(
            {"n_pass": rep.n_pass, "n_total": len(TESTS),
             "all_pass": rep.all_pass,
             "tests": [{"n": n, "name": nm, "passed": ok, "detail": d}
                       for n, nm, ok, d in rep.rows]}, indent=2),
            encoding="utf-8")

    if not rep.all_pass:
        print("\nDo NOT proceed to GPU spend. Fix the failures above.")
        print("Loosening a tolerance to make one pass is forbidden.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
