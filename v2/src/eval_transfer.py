"""One-shot eval: run the optimized v2 pipeline with a different player model
(default: openai/gpt-5.4 full, low reasoning) on the same held-out 40 positions
at seed=999 we used for the final 3-way comparison.

This tests whether the GEPA-optimized prompts (evolved against gpt-5.4-mini)
still help — or hurt — when handed to a stronger base model.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

from .budget import Budget
from .evaluator import evaluate, load_positions
from .pipeline import PromptSet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--optimized-dir", default="v2/runs/run_004")
    p.add_argument("--player-model", default="openai/gpt-5.4")
    p.add_argument("--budget", type=float, default=8.00)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=999)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--n-propose", type=int, default=3)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    run_dir = Path(args.optimized_dir)
    prompts = PromptSet(
        propose=(run_dir / "best_propose.txt").read_text(),
        select=(run_dir / "best_select.txt").read_text(),
    )

    rng = random.Random(args.seed)
    positions = load_positions()
    if len(positions) > args.n:
        positions = rng.sample(positions, args.n)
    print(f"[eval-xfer] N={len(positions)} seed={args.seed} "
          f"player_model={args.player_model}")

    out_path = Path(args.out) if args.out else (run_dir / "holdout_full_model.json")
    log_path = run_dir / "holdout_full_model_calls.jsonl"
    budget = Budget(cap_usd=args.budget, log_path=log_path)

    rep = evaluate(
        prompts, positions, budget,
        n_workers_positions=args.workers,
        n_propose=args.n_propose,
        tag="eval_xfer",
        model=args.player_model,
    )

    blunder = sum(1 for s in rep.scores if s.cp_loss > 200) / max(1, len(rep.scores))
    perfect = sum(1 for s in rep.scores if s.cp_loss == 0) / max(1, len(rep.scores))

    summary = {
        "player_model": args.player_model,
        "optimized_dir": str(run_dir),
        "n_positions": len(rep.scores),
        "rng_seed": args.seed,
        "mean_cp_loss": rep.mean_cp_loss,
        "stderr_cp_loss": rep.stderr_cp_loss,
        "legal_rate": rep.legal_rate,
        "fmt_rate": rep.fmt_rate,
        "blunder_rate_gt200": blunder,
        "perfect_rate_eq0": perfect,
        "total_spent": budget.spent,
        "scores": [asdict(s) for s in rep.scores],
    }

    print(f"\n=== TRANSFER EVAL ({args.player_model}, n={len(rep.scores)}, seed={args.seed}) ===")
    print(f"  mean cp_loss:  {rep.mean_cp_loss:.1f} ± {rep.stderr_cp_loss:.1f}")
    print(f"  legal_rate:    {rep.legal_rate:.2f}")
    print(f"  fmt_rate:      {rep.fmt_rate:.2f}")
    print(f"  blunder >200:  {blunder:.2f}")
    print(f"  perfect =0:    {perfect:.2f}")
    print(f"  spent:         ${budget.spent:.4f}")

    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[eval-xfer] saved -> {out_path}")


if __name__ == "__main__":
    main()
