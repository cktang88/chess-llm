"""Held-out comparison: v1 baseline vs v2 seed (unoptimized pipeline)
vs v2 optimized (GEPA-evolved pipeline).

Uses a different RNG seed than the optimizer so we don't measure overfit.
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from .budget import Budget
from .evaluator import evaluate as evaluate_pipeline, load_positions
from .grader import Grader
from .pipeline import PromptSet
from .player import SingleCallPlayer
from .seed_prompts import PROPOSE_SEED, SELECT_SEED, V1_BASELINE_PROMPT


@dataclass
class SummaryRow:
    name: str
    mean_cp_loss: float
    legal_rate: float
    fmt_rate: float
    blunder_rate: float     # cp_loss > 200
    perfect_rate: float     # cp_loss == 0
    n_positions: int
    spent: float


def _summarize_pipeline(name: str, rep, spent: float) -> SummaryRow:
    return SummaryRow(
        name=name,
        mean_cp_loss=rep.mean_cp_loss,
        legal_rate=rep.legal_rate,
        fmt_rate=rep.fmt_rate,
        blunder_rate=sum(1 for s in rep.scores if s.cp_loss > 200) / max(1, len(rep.scores)),
        perfect_rate=sum(1 for s in rep.scores if s.cp_loss == 0) / max(1, len(rep.scores)),
        n_positions=len(rep.scores),
        spent=spent,
    )


def _run_baseline(positions: list[dict], budget: Budget,
                  *, n_workers: int = 8) -> "list":
    """Run v1 single-call baseline. Returns list of dicts mirroring PositionScore."""
    graders = [Grader() for _ in range(n_workers)]
    for g in graders:
        g._engine_or_open()
    player = SingleCallPlayer(V1_BASELINE_PROMPT, budget, tag="baseline")

    def task(i_pos):
        idx, pos = i_pos
        g = graders[idx % n_workers]
        res = player.get_move(pos["fen"])
        grade = g.grade(pos["fen"], res.move_uci)
        return {
            "fen": pos["fen"], "tag": pos["tag"], "phase": pos["phase"],
            "move_uci": res.move_uci, "legal": grade.legal,
            "cp_loss": grade.centipawn_loss, "best_uci": grade.best_uci,
            "best_eval_cp": grade.best_eval_cp,
            "played_eval_cp": grade.played_eval_cp,
            "fmt_ok": res.fmt_ok,
        }

    try:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(task, list(enumerate(positions))))
    finally:
        for g in graders:
            g.close()
    return results


@dataclass
class _BaselineReport:
    scores: list

    @property
    def mean_cp_loss(self) -> float:
        return sum(s["cp_loss"] for s in self.scores) / max(1, len(self.scores))

    @property
    def legal_rate(self) -> float:
        return sum(1 for s in self.scores if s["legal"]) / max(1, len(self.scores))

    @property
    def fmt_rate(self) -> float:
        return sum(1 for s in self.scores if s["fmt_ok"]) / max(1, len(self.scores))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--optimized-dir", required=True,
                   help="v2/runs/<run>/ directory (with best_propose.txt + best_select.txt)")
    p.add_argument("--budget", type=float, default=2.00)
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=999, help="RNG seed != optimizer seed")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--n-propose", type=int, default=3)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    run_dir = Path(args.optimized_dir)
    optimized = PromptSet(
        propose=(run_dir / "best_propose.txt").read_text(),
        select=(run_dir / "best_select.txt").read_text(),
    )

    rng = random.Random(args.seed)
    positions = load_positions()
    if len(positions) > args.n:
        positions = rng.sample(positions, args.n)
    print(f"[eval] held-out set size = {len(positions)} (seed={args.seed})")

    budget = Budget(cap_usd=args.budget, log_path=run_dir / "holdout_calls.jsonl")

    # 1. v1 baseline: single prompt, single call
    print("\n[eval] (1/3) v1 baseline (single-call, v1 prompt)…")
    base_scores = _run_baseline(positions, budget, n_workers=args.workers)
    base_rep = _BaselineReport(base_scores)
    base_spent = budget.spent
    print(f"   cp={base_rep.mean_cp_loss:.1f} legal={base_rep.legal_rate:.2f} "
          f"fmt={base_rep.fmt_rate:.2f}  spent=${budget.spent:.4f}")

    # 2. v2 seed: pipeline with seed prompts, unoptimized
    print("\n[eval] (2/3) v2 seed pipeline (unoptimized propose+select)…")
    seed_set = PromptSet(propose=PROPOSE_SEED, select=SELECT_SEED)
    spent_before = budget.spent
    seed_rep = evaluate_pipeline(seed_set, positions, budget,
                                 n_workers_positions=args.workers,
                                 n_propose=args.n_propose, tag="eval_v2seed")
    seed_spent = budget.spent - spent_before
    print(f"   cp={seed_rep.mean_cp_loss:.1f} legal={seed_rep.legal_rate:.2f} "
          f"fmt={seed_rep.fmt_rate:.2f}  spent=${budget.spent:.4f}")

    # 3. v2 optimized
    print("\n[eval] (3/3) v2 optimized pipeline…")
    spent_before = budget.spent
    opt_rep = evaluate_pipeline(optimized, positions, budget,
                                n_workers_positions=args.workers,
                                n_propose=args.n_propose, tag="eval_v2opt")
    opt_spent = budget.spent - spent_before
    print(f"   cp={opt_rep.mean_cp_loss:.1f} legal={opt_rep.legal_rate:.2f} "
          f"fmt={opt_rep.fmt_rate:.2f}  spent=${budget.spent:.4f}")

    rows = [
        _summarize_pipeline("v1_baseline_singlecall", base_rep, base_spent),
        _summarize_pipeline("v2_seed_pipeline", seed_rep, seed_spent),
        _summarize_pipeline("v2_optimized_pipeline", opt_rep, opt_spent),
    ]

    print("\n=== HELD-OUT COMPARISON ===")
    headers = ["name", "cp_loss", "legal", "fmt", "blunder>200", "perfect=0", "cost"]
    print(f"{'name':<28} {'cp':>8} {'legal':>6} {'fmt':>6} {'blun':>6} {'perf':>6} {'cost':>8}")
    for r in rows:
        print(f"{r.name:<28} {r.mean_cp_loss:>8.1f} "
              f"{r.legal_rate:>6.2f} {r.fmt_rate:>6.2f} "
              f"{r.blunder_rate:>6.2f} {r.perfect_rate:>6.2f} "
              f"${r.spent:>7.4f}")

    out = {"rows": [asdict(r) for r in rows], "total_spent": budget.spent}
    out_path = Path(args.out) if args.out else (run_dir / "holdout.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[eval] saved -> {out_path}")


if __name__ == "__main__":
    main()
