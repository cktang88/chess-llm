"""GEPA-style reflective prompt evolution for chess LLM.

Hand-rolled per recommendation in v2/tmp/research_optimizers.md.
- Population P=6, iterations T<=16, eval set N=40
- Pareto on (-cp_loss, legal_rate, fmt_rate)
- Reflection minibatch = 8 worst-CP-loss positions
- Maestro tweak: reflector picks which section of prompt to edit
- autocontext tweak: "refine, don't rewrite" instruction
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .budget import Budget, BudgetExceeded
from .evaluator import EvalReport, evaluate, load_positions, report_to_dict
from .player import call_llm_text
from .seed_prompt import SEED_PROMPT


def _hash(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:10]


def _dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def pareto_front(scored: dict[str, EvalReport]) -> list[str]:
    keys = list(scored.keys())
    front = []
    for k in keys:
        ak = scored[k].aggregate()
        if not any(_dominates(scored[o].aggregate(), ak) for o in keys if o != k):
            front.append(k)
    return front


REFLECTION_TEMPLATE = """You are refining an existing system prompt for a chess-playing LLM.
Do NOT rewrite from scratch. Keep what works; surgically fix the failure modes shown below.

# CURRENT SYSTEM PROMPT
<<<PROMPT>>>
{parent_prompt}
<<<END_PROMPT>>>

# FAILURE TRACE
The model played the moves below. Stockfish (depth 14) graded them.
Centipawn loss: how much worse the played move was than the engine best.
Top engine candidates show what the model could have played.

{trace_table}

# YOUR TASK
1. Identify the dominant failure mode in the trace (e.g. tactical blindness,
   poor opening repertoire, weak endgame technique, format errors, illegal moves).
2. Pick ONE section of the current prompt to revise (or add a new section if
   the issue isn't covered). Sections are separated by markdown headers (#, ##).
3. Output the FULL revised system prompt. Keep edits minimal and targeted.

Constraints:
- Keep the prompt under 6000 characters total.
- The prompt MUST instruct the model to respond with JSON containing a UCI move.
- Do NOT remove the rule that the move must be legal in the position.

Respond with JSON:
{{
  "diagnosis": "<2-3 sentences naming the failure mode>",
  "section_edited": "<header text or 'NEW: <name>'>",
  "revised_prompt": "<the FULL revised system prompt as a single string>"
}}
"""


def _format_trace(worst: list) -> str:
    rows = ["| FEN | played | best | cp_loss | legal | top_engine_candidates |",
            "|---|---|---|---|---|---|"]
    for s in worst:
        cands = ", ".join(f"{c['uci']}({c['eval_cp']:+d})"
                          for c in s.get("top_k", [])[:3]) if s.get("top_k") else ""
        rows.append(f"| {s['fen']} | {s['move_uci']} | {s['best_uci']} | "
                    f"{s['cp_loss']} | {s['legal']} | {cands} |")
    return "\n".join(rows)


def reflect(parent_prompt: str, worst_scores: list, budget: Budget) -> tuple[str, dict]:
    """Ask reflector LM for a revised prompt. Returns (revised_prompt, meta)."""
    prompt = REFLECTION_TEMPLATE.format(
        parent_prompt=parent_prompt,
        trace_table=_format_trace(worst_scores),
    )
    raw = call_llm_text(prompt=prompt, budget=budget, tag="reflect")
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:].strip()
    try:
        data = json.loads(text)
        revised = data.get("revised_prompt", "").strip()
        meta = {"diagnosis": data.get("diagnosis", ""),
                "section_edited": data.get("section_edited", ""),
                "raw_len": len(raw)}
        if revised and len(revised) <= 6000:
            return revised, meta
    except Exception as e:
        return parent_prompt, {"error": f"parse failed: {e}", "raw": raw[:300]}
    return parent_prompt, {"error": "no revised_prompt", "raw": raw[:300]}


@dataclass
class IterationLog:
    iter: int
    parent_hash: str
    child_hash: str
    diagnosis: str
    section_edited: str
    parent_score: tuple
    child_score: tuple | None
    accepted: bool
    spent_after: float


@dataclass
class RunState:
    out_dir: Path
    budget: Budget
    population: dict[str, EvalReport] = field(default_factory=dict)
    iterations: list[IterationLog] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    def save(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # Best by aggregate (lex order: cp first, then legal, then fmt)
        best_key = max(self.population.keys(),
                       key=lambda k: self.population[k].aggregate())
        best_rep = self.population[best_key]
        (self.out_dir / "best_prompt.txt").write_text(best_key_text := best_rep.prompt)
        (self.out_dir / "best_metrics.json").write_text(json.dumps({
            "mean_cp_loss": best_rep.mean_cp_loss,
            "legal_rate": best_rep.legal_rate,
            "fmt_rate": best_rep.fmt_rate,
            "prompt_len": len(best_rep.prompt),
            "prompt_hash": _hash(best_rep.prompt),
        }, indent=2))
        # Per-candidate metrics
        pop_summary = {
            _hash(p): {
                "mean_cp_loss": r.mean_cp_loss,
                "legal_rate": r.legal_rate,
                "fmt_rate": r.fmt_rate,
                "prompt_len": len(p),
            }
            for p, r in self.population.items()
        }
        (self.out_dir / "population.json").write_text(json.dumps(pop_summary, indent=2))
        (self.out_dir / "iterations.jsonl").write_text(
            "\n".join(json.dumps(asdict(i), default=str) for i in self.iterations)
        )
        (self.out_dir / "budget.json").write_text(json.dumps(self.budget.summary(), indent=2))


def run(out_dir: Path, *, cap_usd: float, P: int = 6, T: int = 16,
        n_eval: int = 40, n_minibatch: int = 8, seed: int = 0,
        n_workers: int = 8) -> RunState:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    budget = Budget(cap_usd=cap_usd, log_path=out_dir / "calls.jsonl")
    state = RunState(out_dir=out_dir, budget=budget)

    positions = load_positions()
    if len(positions) > n_eval:
        positions = rng.sample(positions, n_eval)
    print(f"[opt] eval set = {len(positions)} positions, cap=${cap_usd}, "
          f"P={P}, T={T}, minibatch={n_minibatch}")

    # Initial: evaluate seed
    print(f"[opt] iter 0: evaluating SEED ({len(SEED_PROMPT)} chars)")
    try:
        seed_report = evaluate(SEED_PROMPT, positions, budget, n_workers=n_workers, tag="eval_seed")
    except BudgetExceeded as e:
        print(f"[opt] budget exceeded during seed eval: {e}")
        return state
    state.population[SEED_PROMPT] = seed_report
    print(f"[opt] SEED: cp_loss={seed_report.mean_cp_loss:.1f} "
          f"legal={seed_report.legal_rate:.2f} fmt={seed_report.fmt_rate:.2f} "
          f"spent=${budget.spent:.4f}")

    for t in range(1, T + 1):
        if budget.remaining() < 0.50:
            print(f"[opt] stopping early: <$0.50 budget remaining (${budget.remaining():.2f})")
            break

        # Pick parent uniformly from current Pareto front
        front = pareto_front(state.population)
        parent = rng.choice(front)
        parent_rep = state.population[parent]

        # Build reflection minibatch: worst CP-loss positions for this parent
        sorted_scores = sorted(parent_rep.scores, key=lambda s: s.cp_loss, reverse=True)
        minibatch = [asdict(s) for s in sorted_scores[:n_minibatch]]

        print(f"[opt] iter {t}: parent={_hash(parent)} "
              f"(cp={parent_rep.mean_cp_loss:.1f}) minibatch_top_loss="
              f"{[s['cp_loss'] for s in minibatch[:3]]}")

        # Reflect
        try:
            child, meta = reflect(parent, minibatch, budget)
        except BudgetExceeded as e:
            print(f"[opt] budget exceeded during reflection: {e}")
            break
        if child in state.population:
            print(f"[opt] iter {t}: dedup (child already in pop), skipping")
            state.iterations.append(IterationLog(
                iter=t, parent_hash=_hash(parent), child_hash=_hash(child),
                diagnosis=meta.get("diagnosis", ""),
                section_edited=meta.get("section_edited", ""),
                parent_score=parent_rep.aggregate(), child_score=None,
                accepted=False, spent_after=budget.spent,
            ))
            continue

        # Evaluate child
        try:
            child_rep = evaluate(child, positions, budget, n_workers=n_workers,
                                 tag=f"eval_iter{t}")
        except BudgetExceeded as e:
            print(f"[opt] budget exceeded during child eval iter {t}: {e}")
            break
        state.population[child] = child_rep

        # Truncate population: keep Pareto front + top-(P - |front|) by aggregate
        if len(state.population) > P:
            front2 = pareto_front(state.population)
            rest = [c for c in state.population if c not in front2]
            rest.sort(key=lambda c: state.population[c].aggregate(), reverse=True)
            keep = set(front2) | set(rest[:max(0, P - len(front2))])
            state.population = {k: v for k, v in state.population.items() if k in keep}

        accepted = child in state.population
        state.iterations.append(IterationLog(
            iter=t, parent_hash=_hash(parent), child_hash=_hash(child),
            diagnosis=meta.get("diagnosis", ""),
            section_edited=meta.get("section_edited", ""),
            parent_score=parent_rep.aggregate(),
            child_score=child_rep.aggregate(),
            accepted=accepted, spent_after=budget.spent,
        ))
        print(f"[opt] iter {t}: child={_hash(child)} "
              f"cp={child_rep.mean_cp_loss:.1f} legal={child_rep.legal_rate:.2f} "
              f"fmt={child_rep.fmt_rate:.2f} accepted={accepted} "
              f"section={meta.get('section_edited','')!r} spent=${budget.spent:.4f}")

        # Save snapshot every iter
        state.save()

    state.save()
    return state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=float, default=9.00)
    p.add_argument("--out", type=str, default="v2/runs/run_001")
    p.add_argument("--P", type=int, default=6)
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--n-eval", type=int, default=40)
    p.add_argument("--minibatch", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out)
    state = run(out_dir, cap_usd=args.budget, P=args.P, T=args.T,
                n_eval=args.n_eval, n_minibatch=args.minibatch,
                n_workers=args.workers, seed=args.seed)
    print(f"\n[opt] DONE. spent=${state.budget.spent:.4f} "
          f"out={out_dir}")
    if state.population:
        best = max(state.population, key=lambda p: state.population[p].aggregate())
        rep = state.population[best]
        print(f"[opt] BEST: cp_loss={rep.mean_cp_loss:.1f} "
              f"legal={rep.legal_rate:.2f} fmt={rep.fmt_rate:.2f}")


if __name__ == "__main__":
    main()
