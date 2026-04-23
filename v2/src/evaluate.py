"""Held-out comparison: optimized vs vanilla seed prompt.

Uses a *different* random seed for the position sample so we don't measure
overfit to the optimizer's eval set.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .budget import Budget
from .evaluator import evaluate, load_positions
from .seed_prompt import SEED_PROMPT


def _summary(name: str, rep) -> dict:
    return {
        "name": name,
        "mean_cp_loss": rep.mean_cp_loss,
        "legal_rate": rep.legal_rate,
        "fmt_rate": rep.fmt_rate,
        "n_positions": len(rep.scores),
        "blunder_rate": sum(1 for s in rep.scores if s.cp_loss > 200) / len(rep.scores),
        "perfect_rate": sum(1 for s in rep.scores if s.cp_loss == 0) / len(rep.scores),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--best-prompt", required=True,
                   help="path to v2/runs/<run>/best_prompt.txt")
    p.add_argument("--budget", type=float, default=1.00)
    p.add_argument("--n", type=int, default=40, help="held-out positions")
    p.add_argument("--seed", type=int, default=999, help="different from opt seed")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    optimized = Path(args.best_prompt).read_text()
    rng = random.Random(args.seed)
    positions = load_positions()
    if len(positions) > args.n:
        positions = rng.sample(positions, args.n)
    print(f"[eval] held-out set size = {len(positions)} (seed={args.seed})")

    budget = Budget(cap_usd=args.budget)

    print("[eval] running BASELINE (v1 seed prompt)…")
    base_rep = evaluate(SEED_PROMPT, positions, budget,
                        n_workers=args.workers, tag="eval_baseline")
    print(f"  cp_loss={base_rep.mean_cp_loss:.1f} legal={base_rep.legal_rate:.2f} "
          f"fmt={base_rep.fmt_rate:.2f}  spent=${budget.spent:.4f}")

    print("[eval] running OPTIMIZED prompt…")
    opt_rep = evaluate(optimized, positions, budget,
                       n_workers=args.workers, tag="eval_optimized")
    print(f"  cp_loss={opt_rep.mean_cp_loss:.1f} legal={opt_rep.legal_rate:.2f} "
          f"fmt={opt_rep.fmt_rate:.2f}  spent=${budget.spent:.4f}")

    base_s = _summary("baseline", base_rep)
    opt_s = _summary("optimized", opt_rep)
    delta = {k: opt_s[k] - base_s[k] for k in base_s if isinstance(base_s[k], (int, float))}
    delta["name"] = "delta (optimized - baseline)"

    out = {"baseline": base_s, "optimized": opt_s, "delta": delta,
           "spent": budget.spent}
    print("\n=== HELD-OUT COMPARISON ===")
    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"[eval] saved -> {args.out}")


if __name__ == "__main__":
    main()
