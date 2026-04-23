"""Multi-prompt GEPA: jointly optimize {propose_prompt, select_prompt}.

Based on v2/tmp/research_optimizers.md, with lite C (Maestro edit-priority):
the reflector sees the parent PromptSet + worst-case traces and picks ONE
module ("propose" or "select") to revise.

- Population P=6 PromptSets
- Iterations T=12 (tighter than single-prompt because each eval is ~4x cost)
- Eval set = 30 positions
- Minibatch = 8 worst-cp positions
- 3-objective Pareto on (-cp_loss, legal_rate, fmt_rate)
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .budget import Budget, BudgetExceeded
from .evaluator import EvalReport, evaluate, load_positions
from .llm import chat
from .pipeline import PromptSet
from .seed_prompts import PROPOSE_SEED, SELECT_SEED


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


REFLECTION_TEMPLATE = """You are refining a 2-module chess-playing LLM pipeline.

The pipeline has two modules, each a system prompt:
- PROPOSE: called 3 times in parallel (temperature 1.0) to suggest diverse candidate moves.
- SELECT:  called once on the legal, de-duplicated candidates to pick the best.

Your job: identify the dominant failure mode from the trace, pick ONE module
to edit, and output the revised prompt for that module. Keep the other module's
prompt unchanged. Do NOT rewrite from scratch — keep what works, surgically
fix what doesn't.

# CURRENT PROPOSE PROMPT
<<<PROPOSE>>>
{propose_prompt}
<<<END_PROPOSE>>>

# CURRENT SELECT PROMPT
<<<SELECT>>>
{select_prompt}
<<<END_SELECT>>>

# FAILURE TRACE
The pipeline was run on the positions below. Stockfish (depth 14) graded the
final selected move. For each position you see: the FEN, all 3 proposer
candidate moves (with their reasoning), whether SELECT was used and what it
chose, what Stockfish's best move was, and the centipawn loss.

{trace_table}

# DIAGNOSIS GUIDE
Ask: was the best move ever proposed?
- If Stockfish's best move appears among the proposer candidates, but SELECT
  did not pick it, SELECT is at fault. Edit SELECT.
- If the best move does not appear in any proposer candidate (or proposers
  disagree wildly on bad moves), PROPOSE is at fault. Edit PROPOSE.
- If candidates are often illegal or malformed JSON, PROPOSE's output format
  section needs strengthening.

# RULES
- Keep the edited prompt under 5000 characters.
- Keep JSON output + UCI notation rules intact.
- Do not remove the instruction that moves must be legal.

# OUTPUT — JSON ONLY, no markdown fences:
{{
  "diagnosis": "<2-3 sentences: the failure mode you identified>",
  "module_to_edit": "propose" | "select",
  "revised_prompt": "<the FULL revised prompt for the chosen module>"
}}
"""


def _format_trace(worst: list) -> str:
    rows = []
    for s in worst:
        cands = "; ".join(
            f"{m}({r[:60]})" if r else f"{m}(?)"
            for m, r in zip(s["candidate_moves"], s["candidate_reasonings"])
        )
        sel = "SELECT->" + (s["move_uci"] or "NONE") if s["selector_used"] else "UNANIMOUS->" + (s["move_uci"] or "NONE")
        rows.append(
            f"- FEN: {s['fen']}\n"
            f"  proposers: [{cands}]\n"
            f"  {sel}\n"
            f"  stockfish best: {s['best_uci']}  (played eval {s.get('played_eval_cp')}, best eval {s['best_eval_cp']})\n"
            f"  cp_loss={s['cp_loss']}  legal={s['legal']}  fmt_ok={s['fmt_ok']}"
        )
    return "\n\n".join(rows)


def reflect(parent: PromptSet, worst_scores: list, budget: Budget,
            *, forced_module: str | None = None,
            temperature: float | None = None) -> tuple[PromptSet, dict]:
    prompt = REFLECTION_TEMPLATE.format(
        propose_prompt=parent.propose,
        select_prompt=parent.select,
        trace_table=_format_trace(worst_scores),
    )
    if forced_module in ("propose", "select"):
        prompt += (
            f"\n\n# MANDATORY OVERRIDE\n"
            f"You MUST set \"module_to_edit\" to \"{forced_module}\" this iteration, "
            f"regardless of your diagnosis. Even if the other module looks more at "
            f"fault, revise the {forced_module.upper()} prompt. Find SOMETHING in it "
            f"to improve based on the trace.\n"
        )
    try:
        content, _ = chat(
            messages=[{"role": "user", "content": prompt}],
            budget=budget, tag="reflect",
            reasoning_effort="low",
            temperature=temperature,
        )
    except BudgetExceeded:
        raise
    except Exception as e:
        return parent, {"error": f"reflect call failed: {e}"}

    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:].strip()
    try:
        data = json.loads(text)
    except Exception as e:
        return parent, {"error": f"parse failed: {e}", "raw": content[:300]}

    module = data.get("module_to_edit", "")
    revised = (data.get("revised_prompt") or "").strip()
    if not revised or len(revised) > 7000:
        return parent, {"error": f"bad revised_prompt len={len(revised)}"}

    if module == "propose":
        new_set = PromptSet(propose=revised, select=parent.select)
    elif module == "select":
        new_set = PromptSet(propose=parent.propose, select=revised)
    else:
        return parent, {"error": f"unknown module_to_edit={module!r}"}

    # If a forced module was requested and reflector still edited the other one,
    # reject (reflector disobeyed the mandatory override).
    if forced_module and module != forced_module:
        return parent, {"error": f"reflector edited {module!r}, forced was {forced_module!r}"}

    return new_set, {
        "diagnosis": data.get("diagnosis", ""),
        "module_to_edit": module,
        "revised_len": len(revised),
    }


@dataclass
class IterationLog:
    iter: int
    parent_key: str
    child_key: str
    module_edited: str
    diagnosis: str
    parent_score: tuple
    child_score: tuple | None
    accepted: bool
    spent_after: float


@dataclass
class RunState:
    out_dir: Path
    budget: Budget
    population: dict[str, EvalReport] = field(default_factory=dict)  # key = PromptSet.key()
    prompts_by_key: dict[str, PromptSet] = field(default_factory=dict)
    iterations: list[IterationLog] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    def add(self, prompts: PromptSet, report: EvalReport) -> str:
        k = prompts.key()
        self.prompts_by_key[k] = prompts
        self.population[k] = report
        return k

    def save(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if not self.population:
            return
        best_key = max(self.population.keys(),
                       key=lambda k: self.population[k].aggregate())
        best_prompts = self.prompts_by_key[best_key]
        best_rep = self.population[best_key]

        (self.out_dir / "best_propose.txt").write_text(best_prompts.propose)
        (self.out_dir / "best_select.txt").write_text(best_prompts.select)
        (self.out_dir / "best_metrics.json").write_text(json.dumps({
            "mean_cp_loss": best_rep.mean_cp_loss,
            "legal_rate": best_rep.legal_rate,
            "fmt_rate": best_rep.fmt_rate,
            "propose_len": len(best_prompts.propose),
            "select_len": len(best_prompts.select),
            "key": best_key,
        }, indent=2))

        pop_summary = {
            k: {
                "mean_cp_loss": r.mean_cp_loss,
                "legal_rate": r.legal_rate,
                "fmt_rate": r.fmt_rate,
                "propose_len": len(self.prompts_by_key[k].propose),
                "select_len": len(self.prompts_by_key[k].select),
            }
            for k, r in self.population.items()
        }
        (self.out_dir / "population.json").write_text(json.dumps(pop_summary, indent=2))
        (self.out_dir / "iterations.jsonl").write_text(
            "\n".join(json.dumps(asdict(i), default=str) for i in self.iterations)
        )
        (self.out_dir / "budget.json").write_text(json.dumps(self.budget.summary(), indent=2))


def _load_warm_start(warm_dir: Path) -> PromptSet | None:
    p_file = warm_dir / "best_propose.txt"
    s_file = warm_dir / "best_select.txt"
    if not (p_file.exists() and s_file.exists()):
        return None
    return PromptSet(propose=p_file.read_text(), select=s_file.read_text())


def run(out_dir: Path, *, cap_usd: float, P: int = 6, T: int = 12,
        n_eval: int = 30, n_minibatch: int = 8, n_propose: int = 3,
        n_workers: int = 4, seed: int = 0,
        warm_start: PromptSet | None = None,
        force_module_rotation: str | None = None,
        reflect_temperature: float | None = None) -> RunState:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    budget = Budget(cap_usd=cap_usd, log_path=out_dir / "calls.jsonl")
    state = RunState(out_dir=out_dir, budget=budget)

    positions = load_positions()
    if len(positions) > n_eval:
        positions = rng.sample(positions, n_eval)
    print(f"[opt] eval={len(positions)} cap=${cap_usd} P={P} T={T} "
          f"propose_N={n_propose} workers={n_workers}")

    seed_set = warm_start or PromptSet(propose=PROPOSE_SEED, select=SELECT_SEED)
    label = "WARM" if warm_start else "SEED"
    print(f"[opt] iter 0: evaluating {label} (propose={len(seed_set.propose)}ch, "
          f"select={len(seed_set.select)}ch)")
    try:
        seed_rep = evaluate(seed_set, positions, budget,
                            n_workers_positions=n_workers, n_propose=n_propose,
                            tag=f"eval_{label.lower()}")
    except BudgetExceeded as e:
        print(f"[opt] budget exceeded during {label.lower()} eval: {e}")
        return state
    state.add(seed_set, seed_rep)
    print(f"[opt] {label}: cp={seed_rep.mean_cp_loss:.1f} "
          f"legal={seed_rep.legal_rate:.2f} fmt={seed_rep.fmt_rate:.2f} "
          f"spent=${budget.spent:.4f}")
    state.save()

    for t in range(1, T + 1):
        if budget.remaining() < 0.80:
            print(f"[opt] stop: <$0.80 remaining (${budget.remaining():.2f})")
            break

        front = pareto_front(state.population)
        parent_key = rng.choice(front)
        parent_prompts = state.prompts_by_key[parent_key]
        parent_rep = state.population[parent_key]

        # Minibatch: 8 worst-cp positions for this parent
        sorted_scores = sorted(parent_rep.scores, key=lambda s: s.cp_loss, reverse=True)
        minibatch = [asdict(s) for s in sorted_scores[:n_minibatch]]

        # Optional module rotation. "alt-select-first" means odd iters force
        # select (understudied in run_001), even iters force propose.
        forced = None
        if force_module_rotation == "alt-select-first":
            forced = "select" if (t % 2 == 1) else "propose"
        elif force_module_rotation == "alt-propose-first":
            forced = "propose" if (t % 2 == 1) else "select"
        elif force_module_rotation in ("propose", "select"):
            forced = force_module_rotation

        print(f"[opt] iter {t}: parent={parent_key} "
              f"cp={parent_rep.mean_cp_loss:.1f} worst_loss="
              f"{[s['cp_loss'] for s in minibatch[:3]]}"
              + (f" forced={forced}" if forced else ""))

        try:
            child_prompts, meta = reflect(parent_prompts, minibatch, budget,
                                          forced_module=forced,
                                          temperature=reflect_temperature)
        except BudgetExceeded as e:
            print(f"[opt] budget exceeded in reflection: {e}")
            break

        if "error" in meta or child_prompts.key() == parent_key:
            state.iterations.append(IterationLog(
                iter=t, parent_key=parent_key, child_key=child_prompts.key(),
                module_edited=meta.get("module_to_edit", ""),
                diagnosis=meta.get("diagnosis", meta.get("error", "")),
                parent_score=parent_rep.aggregate(), child_score=None,
                accepted=False, spent_after=budget.spent,
            ))
            print(f"[opt] iter {t}: reflection no-op ({meta.get('error', 'unchanged')})")
            continue

        child_key = child_prompts.key()
        if child_key in state.population:
            print(f"[opt] iter {t}: dedup, skipping child={child_key}")
            state.iterations.append(IterationLog(
                iter=t, parent_key=parent_key, child_key=child_key,
                module_edited=meta["module_to_edit"], diagnosis=meta["diagnosis"],
                parent_score=parent_rep.aggregate(), child_score=None,
                accepted=False, spent_after=budget.spent,
            ))
            continue

        try:
            child_rep = evaluate(child_prompts, positions, budget,
                                 n_workers_positions=n_workers,
                                 n_propose=n_propose, tag=f"eval_iter{t}")
        except BudgetExceeded as e:
            print(f"[opt] budget exceeded in child eval iter {t}: {e}")
            break
        state.add(child_prompts, child_rep)

        # Truncate to P
        if len(state.population) > P:
            front2 = pareto_front(state.population)
            rest = [k for k in state.population if k not in front2]
            rest.sort(key=lambda k: state.population[k].aggregate(), reverse=True)
            keep = set(front2) | set(rest[:max(0, P - len(front2))])
            state.population = {k: v for k, v in state.population.items() if k in keep}
            state.prompts_by_key = {k: v for k, v in state.prompts_by_key.items() if k in keep}

        accepted = child_key in state.population
        state.iterations.append(IterationLog(
            iter=t, parent_key=parent_key, child_key=child_key,
            module_edited=meta["module_to_edit"], diagnosis=meta["diagnosis"],
            parent_score=parent_rep.aggregate(),
            child_score=child_rep.aggregate(),
            accepted=accepted, spent_after=budget.spent,
        ))
        print(f"[opt] iter {t}: child={child_key} module={meta['module_to_edit']} "
              f"cp={child_rep.mean_cp_loss:.1f} legal={child_rep.legal_rate:.2f} "
              f"fmt={child_rep.fmt_rate:.2f} accepted={accepted} "
              f"spent=${budget.spent:.4f}")
        state.save()

    state.save()
    return state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=float, default=9.00)
    p.add_argument("--out", type=str, default="v2/runs/run_001")
    p.add_argument("--P", type=int, default=6)
    p.add_argument("--T", type=int, default=12)
    p.add_argument("--n-eval", type=int, default=30)
    p.add_argument("--minibatch", type=int, default=8)
    p.add_argument("--n-propose", type=int, default=3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warm-start-from", type=str, default=None,
                   help="v2/runs/<run>/ dir to load best_propose.txt+best_select.txt from")
    p.add_argument("--force-module-rotation", type=str, default=None,
                   choices=[None, "alt-select-first", "alt-propose-first",
                            "propose", "select"])
    p.add_argument("--reflect-temperature", type=float, default=None)
    args = p.parse_args()

    warm = _load_warm_start(Path(args.warm_start_from)) if args.warm_start_from else None
    if args.warm_start_from and warm is None:
        raise SystemExit(f"warm-start dir missing best_propose.txt/best_select.txt: {args.warm_start_from}")

    out = Path(args.out)
    state = run(out, cap_usd=args.budget, P=args.P, T=args.T,
                n_eval=args.n_eval, n_minibatch=args.minibatch,
                n_propose=args.n_propose, n_workers=args.workers, seed=args.seed,
                warm_start=warm,
                force_module_rotation=args.force_module_rotation,
                reflect_temperature=args.reflect_temperature)
    print(f"\n[opt] DONE. spent=${state.budget.spent:.4f}  out={out}")
    if state.population:
        best_key = max(state.population, key=lambda k: state.population[k].aggregate())
        rep = state.population[best_key]
        print(f"[opt] BEST: cp={rep.mean_cp_loss:.1f} "
              f"legal={rep.legal_rate:.2f} fmt={rep.fmt_rate:.2f} key={best_key}")


if __name__ == "__main__":
    main()
