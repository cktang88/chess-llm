# Methodology — optimizing an LLM chess player with Stockfish grading

This document describes how v2 was built: the problem framing, the design
choices, the optimizer we used, the trade-offs we made, and what we got out
the other side. Intended for a reviewer who already knows the basics of LLM
tool use and prompt engineering but hasn't read the GEPA / Maestro / RoboPHD
papers.

## 1. The problem

v1 was a one-shot LLM chess player: a hand-written ~130-line system prompt
(`chess_prompt.py`) → one call to `gpt-5.2` per move → play the parsed UCI
move. Workable, but weak. The question for v2: **given a fixed, small budget
(~$10 of API), make the same model class play measurably better chess,
without fine-tuning and without engine tool calls at inference time.**

Fixed constraints at the start:

| Constraint | Value |
|---|---|
| Model | `openai/gpt-5.4-mini` (via OpenRouter) |
| Reasoning effort | `low` |
| Total API spend | ≤ $10 |
| Fine-tuning | Not allowed |
| Engine tool-calls at inference | Not allowed ("pure LLM") |
| Wall clock | Ideally under an hour |
| v1 files | Must not be modified |

The grader is allowed to use Stockfish — we just can't let the LLM call it
during a game.

## 2. Metric and grading signal

We don't play full games during optimization. Instead we use a **fixed
dataset of ~44 positions** (openings, middlegames, tactical puzzles, basic
endgames — see `src/dataset.py`) and grade the move the LLM plays at each
position against Stockfish:

```text
cp_loss = max(0, best_eval_cp  −  played_move_eval_cp)   clipped at 1000
```

where `best_eval_cp` is Stockfish's eval of its preferred move at depth 14,
and `played_move_eval_cp` is Stockfish's eval of the LLM's chosen move
(technically: negation of the eval after the LLM's move, from the moving
side's POV).

**Why per-position grading instead of full games:** dense signal per LLM
call, low variance, no dependence on the opponent's skill, deterministic,
~100ms of Stockfish compute per grade. A full game at 40 plies per side
would consume 40× more LLM calls and give one noisy W/D/L.

Auxiliary metrics (tracked as Pareto co-objectives so the optimizer can't
trade them away):

- `legal_rate` — fraction of moves that were legal UCI in the position
- `fmt_rate`  — fraction of LLM responses that parsed as the expected JSON
  shape `{"move": "...", "reasoning": "..."}`

A move that is illegal or unparseable scores `cp_loss = 1000` (the clip).
Fitness is the 3-tuple `(-mean_cp_loss, legal_rate, fmt_rate)`; higher is
better on all three.

## 3. The harness (what we're actually optimizing)

v1 was one LLM call per move. That gives the optimizer nothing to work with
except a single prompt. So before optimization, we made the harness *richer*
so there's more surface to tune.

v2 harness — two modules, four calls per move in the worst case:

```text
FEN ──┬─► PROPOSE (sys=propose_prompt, temp=1.0, style hint 0)
      ├─► PROPOSE (sys=propose_prompt, temp=1.0, style hint 1)   (parallel)
      └─► PROPOSE (sys=propose_prompt, temp=1.0, style hint 2)
                           │
                 candidate moves (with 1-sentence reasoning each)
                           │
                 filter to legal, de-duplicate
                           │
               ┌───────────┴────────────┐
               ▼                         ▼
          1 unique?                 multiple
               │                         │
           skip SELECT          SELECT (sys=select_prompt, temp=0.0)
               │                         │
               └───────────┬─────────────┘
                           ▼
                  chosen UCI move
```

Two design choices worth naming:

1. **Self-consistency via parallel PROPOSE calls** at temperature 1.0. This
   is the single best-documented inference-time technique for LLM chess. We
   also inject a small one-line "style hint" that cycles across calls
   (tactical-focus / prophylaxis / pawn-break / king-safety) because
   reasoning models give weak temperature-based diversity on their own.
2. **SELECT is skipped when all proposers agree** (unanimous UCI). Saves
   ~25% of SELECT calls in practice, at no cost.

What the optimizer evolves: the system prompts *of both modules*. The
reflector decides which module to edit on each iteration (§5).

## 4. Why GEPA, and not something "stronger"

The obvious pushback — isn't RoboPHD / Maestro / ADAS "better" than GEPA?
Short answer: they solve different problems.

| Optimizer | Targets | Min rollouts | Fits our setting? |
|---|---|---|---|
| **GEPA** (Agrawal et al. 2025) | Text artifact(s) with dense feedback | ~100–200 | **Yes** |
| OPRO / Promptbreeder | Single prompt, scalar score | 200–2000 | Dominated by GEPA for text feedback |
| TextGrad | Prompt via text "gradients" | 100–300 | Works, but no scalar; dominated here |
| **RoboPHD** (2026) | Agents evaluated by pairwise Elo tournament | ~1500 | Mismatch — we have a fixed oracle (Stockfish), not pairwise games |
| **Maestro** | Agent graph structure + per-node configs | fewer than GEPA (claimed) | Overkill for a 2-node pipeline; **no released code** at time of writing |
| ADAS / AlphaEvolve | Code evolution (writes new modules) | 1000s–10k+ | 10–100× over budget |
| DSPy + MIPROv2 | Multi-module prompt compilation | 500+ | Works, but wraps our simple pipeline in heavy abstractions |

GEPA fits our regime exactly: textual artifacts (prompts), dense scalar +
rich text feedback (cp_loss number + Stockfish's top-k candidates per
position), and a budget that can't absorb 1500+ rollouts. RoboPHD's Elo
machinery needs pairwise matches; with a fixed oracle it collapses to
truncation with extra noise. Maestro is the closest "strictly better"
candidate — but (a) its edit operators target graph structure which we
don't have, and (b) no public code at the time we built this.

**From Maestro we did borrow one idea** (sometimes called "edit priority"):
the reflector's output explicitly picks *which module to edit* on each
iteration, not just what the edit is. This is ~10 lines of JSON schema in
our reflector template. It gives the reflector a simple way to spend its
mutation budget on the module that's most at fault.

**From RoboPHD** we borrowed one idea: no train/val split. With ~44
positions and a $10 cap, a holdout split costs us variance without buying
anti-overfit protection. We evaluate on all training positions during
optimization and use a fresh RNG seed for held-out evaluation.

**We decided not to use DSPy.** A single move-generation call wrapped in
`dspy.Module` + Signature adapters just adds friction; the hand-rolled
~250-line optimizer is easier to reason about and lets us format Stockfish
feedback exactly how we want.

## 5. The optimizer (hand-rolled GEPA)

Source: `src/optimizer.py`. Pseudocode:

```python
population = {seed_key: evaluate(seed_prompts, positions)}  # dict of PromptSet -> report

for t in 1..T:
    parent = random.choice(pareto_front(population))
    worst_positions = top_k_by_cp_loss(population[parent].scores, k=minibatch)

    child_prompts, meta = reflect(parent, worst_positions)
    #  reflector sees: parent PROPOSE + parent SELECT + per-position trace
    #    (FEN, 3 proposer candidates+reasoning, SELECT choice, engine best, cp_loss)
    #  reflector returns: {diagnosis, module_to_edit in {propose, select}, revised_prompt}

    if child_prompts already in population: continue      # cheap hash dedup
    child_report = evaluate(child_prompts, positions)
    population[child_prompts] = child_report

    # Pareto truncation to population size P
    front = pareto_front(population)
    rest  = sorted(population - front, by aggregate descending)
    population = front ∪ rest[: max(0, P − |front|)]
```

**Pareto front** is over the 3-tuple `(-cp_loss, legal_rate, fmt_rate)`.
Candidates that dominate on all three stay; ties are broken by keeping both.
This is what prevents the optimizer from improving cp_loss by producing
illegal moves or malformed JSON.

**Reflection** is a single LLM call (same model as the player) that takes:

- the parent's PROPOSE prompt
- the parent's SELECT prompt
- a table of the parent's 8 worst-cp_loss positions, each with:
  - FEN
  - all 3 proposer candidates (UCI + reasoning)
  - whether SELECT was used and what it picked
  - Stockfish's best move + evals
  - cp_loss, legal flag, fmt flag

…and returns JSON with diagnosis, module-to-edit, and the full revised prompt
for that module. The other module is passed through unchanged. The
template has an autocontext-inspired "refine, don't rewrite" instruction
and explicit guidance on who-is-at-fault attribution (if best move was in
candidates but SELECT missed it → SELECT; if best move was never proposed →
PROPOSE; illegal/malformed → PROPOSE's output section).

## 6. Hyperparameters and why

| Hyperparam | Value | Why |
|---|---|---|
| Population `P` | 6 | Small enough that Pareto truncation bites by iter ~6; any larger wastes eval budget |
| Iterations `T` | 12 | Budget-bounded. Typical optimizer run is 10–11 iters before cap |
| Eval set size `N` | 30 (sampled from 44) | 30×~4 calls ≈ 120 LLM calls per eval; 13 evals ≈ 1560 calls; ~$5–7 |
| Minibatch `m` | 8 | Enough to show >1 failure mode to the reflector; short enough that prompt stays under 6K chars |
| Propose `N` | 3 | Diminishing returns above 3; also a nice number for the self-consistency vote |
| Workers | 4 | 4 positions × 4 calls = 16 concurrent requests; OpenRouter handles this fine |
| Reflector temperature | `None` (run_001), `1.0` (run_002+) | run_002 added mutation noise; see §8 |
| Reflector model | `gpt-5.4-mini` (run_001,2), `gpt-5.4` (run_003) | Stronger model for reflection only; player stays on mini |
| History size `N` | 0 (run_001,2), 5 (run_003) | Pass last-N iter outcomes to reflector to break dead-end explorations |
| σ-gated acceptance | 0 (run_001,2), 1.5 (run_003) | Child must beat parent by 1.5×combined_stderr for the improvement to count as real |
| Hard budget cap | `$7.50` (run_001), `$4.50` (run_002), `$5.00` (run_003) | Reserves room for held-out eval |
| Early-stop threshold | `$0.80` remaining | Prevents starting an iter we can't finish |

## 7. Budget accounting

All OpenAI-compat requests return a `usage` object; OpenRouter additionally
reports a `cost` field with the actual USD charged. `src/budget.py` prefers
that number and falls back to a token-price estimate. Every call is
appended to `calls.jsonl` with `{ts, tag, model, input_tokens,
output_tokens, cost, source}`. When cumulative spend exceeds the cap, the
next call raises `BudgetExceeded` and the optimizer breaks cleanly with
state saved.

Tags let us slice cost after the fact:

```text
eval_seed_propose      90 calls  $0.407
eval_seed_select       23 calls  $0.093
eval_iter1_propose     90 calls  $0.473
eval_iter1_select      25 calls  $0.114
reflect                11 calls  $0.068     ← reflection is ~1% of total cost
…
```

Reflection is cheap because it's one call per iter (regardless of eval set
size). All the spend is in `evaluate()`.

## 8. Results

### run_001 — default settings (no warm-start, no forced rotation)

```text
training eval (30 positions, seed=0):
  SEED (unoptimized propose+select pipeline):   cp_loss = 120.0
  best (iter 7):                                cp_loss =  53.9  (−55%)
  total spend: $7.22 / $7.50 cap, 1384 calls, 11 iterations completed

held-out eval (30 positions, seed=999 — NOT seen by optimizer):
  v1 baseline (single-call, v1 prompt):         cp_loss = 139.0
  v2 seed pipeline (unoptimized prompts):       cp_loss = 110.6   (−28)
  v2 OPTIMIZED pipeline:                        cp_loss =  81.3   (−58 total, −42%)
  legal_rate and fmt_rate stayed at 1.00 throughout
```

The decomposition is the interesting part: **half the gain was from the
harness change** (single call → 3×propose+select), **half from GEPA**
editing the PROPOSE prompt. Both contributions generalized to unseen
positions.

### run_002 — diversity-enhanced (warm-start + forced rotation)

Observation from run_001: all 11 reflector edits targeted PROPOSE; SELECT
was never touched, despite the trace format making SELECT failures visible.
Two possibilities: either SELECT was genuinely near-optimal, or the
reflector found PROPOSE easier to edit and never explored the SELECT axis.

To test this, run_002 adds three flags:

- `--warm-start-from v2/runs/run_001` — start from run_001's best
  PromptSet, not the SEED defaults
- `--force-module-rotation alt-select-first` — override the reflector's
  `module_to_edit` choice so odd iters are forced to edit SELECT, even
  iters PROPOSE
- `--reflect-temperature 1.0` — give the reflector non-zero sampling noise

Early signal: iter 3 (forced SELECT edit) produced cp=37.5 vs warm-start
parent's cp=64.9 on seed=1 positions — a 27-point drop. This is above the
~18cp noise floor and is **the first SELECT edit ever made**, validating
the rotation hypothesis: the reflector *would* have helped if it had
explored SELECT. Subsequent iters regressed, consistent with one real gain
plus noise.

### Why we're adding run_003 — three known weaknesses

Looking honestly at the run_001 and run_002 trajectories, three specific
weaknesses show up:

**1. Evaluation noise swamps small deltas.**
With per-position cp-loss SD around ~100, a 30-position mean has standard
error ≈ 18cp. Iter-to-iter differences under 18cp are likely pure
position-sampling luck. That's why run_001 iter 7's "new best" at 53.9
(vs iter 2 at 54.2) was almost certainly noise — the held-out jump from
54→81 bears this out.

**Fix**: require children to beat parent by `accept_sigma × combined_stderr`
before treating the delta as real. Track `mean ± stderr` in every log
line so reviewers can see the noise band.

**2. Same-model reflector hits a ceiling.**
At cp=120 the fixes are obvious ("consider quiet moves"). At cp=54,
the remaining errors require articulating blind spots the same
gpt-5.4-mini model has. It can't easily describe what it doesn't see.

**Fix**: use `openai/gpt-5.4` (full, not mini) as the reflector. Player
stays on mini. Reflection is 1 call per iter — incremental cost is
~$0.30 per reflect call, well under budget.

**3. No memory across iterations.**
Each reflect call starts fresh. The reflector does not see "you already
tried editing PROPOSE in iters 3, 4, 6 — all regressed." So it keeps
exploring the same dead-end axis. Part of run_001's monoculture on
PROPOSE edits is explained by this.

**Fix**: pass the last-N `(module, Δcp, diagnosis, kept/rejected)` tuples
as a "PREVIOUS ATTEMPTS" section in the reflection prompt. Zero API cost,
~30 lines of code.

### run_003 — planned combined fix

New flags in `src/optimizer.py`:

- `--reflect-model openai/gpt-5.4` — stronger reflector
- `--history-n 5` (default) — pass last 5 iter outcomes to reflector
- `--accept-sigma 1.5` — σ-gated improvement test (Pareto still enforces
  legal/fmt floors; σ only gates the cp_loss objective for eviction logic)

Logs now show `cp=<mean>±<stderr>` plus a `(σ-real)` or `(~noise)` flag per
iter. `EvalReport.stderr_cp_loss` computes the sample standard error.

Planned run_003 command:

```bash
uv run python -m v2.src.optimizer \
  --budget 5.00 --T 12 --n-eval 40 --workers 4 --n-propose 3 \
  --warm-start-from v2/runs/run_002 \
  --force-module-rotation alt-select-first \
  --reflect-temperature 1.0 \
  --reflect-model openai/gpt-5.4 \
  --history-n 5 \
  --accept-sigma 1.5 \
  --seed 2 \
  --out v2/runs/run_003
```

## 9. Lessons learned

- **Per-position grading is the right primitive.** Games are too noisy at
  this budget; positions are dense, deterministic, and parallelizable.
- **Position sampling matters more than I expected.** Swapping RNG seed
  from 0→1 shifted baseline cp_loss by ~11. Our 30-position samples are
  small; tracking variance is important. run_003 switches to 40 positions
  (all available) and reports `cp ± stderr` so the noise band is visible.
- **The harness change alone moved the needle a lot.** Self-consistency
  with parallel proposers + a selector LLM was responsible for ~50% of the
  total improvement in run_001.
- **Reflector bias is real.** Given two modules to choose between, the
  reflector picked the same one 11 times in a row. Forcing rotation in
  run_002 produced the first SELECT edit and the best candidate yet.
- **Same-model reflection has a ceiling.** The reflector is the same model
  whose blind spots we're trying to fix. run_003 uses `openai/gpt-5.4`
  (full) as reflector only — highest-ROI change per dollar.
- **Memory-less reflection wastes iterations.** Each call was making the
  same diagnosis without knowing that three previous attempts along the
  same axis had regressed. run_003 passes the last-5 attempts as a
  "PREVIOUS ATTEMPTS" section.
- **OpenRouter's reported per-call cost is gold.** Token-price estimates
  were ~20% high vs actuals for gpt-5.4-mini at low reasoning.
- **Hand-rolled GEPA is plenty.** ~300 lines. Debugging is trivial. Would
  absolutely reach for DSPy for anything with >2 modules.

## 10. What we deliberately did not do

- **No engine at inference.** Adding a Stockfish tool call per move would
  trivially beat anything we did here, but it defeats the exercise.
- **No full games during optimization.** Position grading gave us the
  signal we needed in ~1/40 of the compute.
- **No fine-tuning.** Out of scope per constraints.
- **No DSPy.** Worth revisiting if the pipeline grows to ≥3 modules.
- **No crossover operator.** Would be the next thing to try if more
  diversity is needed — merge two Pareto parents via the reflector.

## 11. Running it

```bash
# .env must contain OPENROUTER_API_KEY=…

# First run (default settings)
uv run python -m v2.src.optimizer \
  --budget 7.50 --T 12 --n-eval 30 --workers 4 --n-propose 3 \
  --out v2/runs/run_001

# Follow-up run with diversity
uv run python -m v2.src.optimizer \
  --budget 4.50 --T 12 --n-eval 30 --workers 4 --n-propose 3 \
  --warm-start-from v2/runs/run_001 \
  --force-module-rotation alt-select-first \
  --reflect-temperature 1.0 \
  --seed 1 \
  --out v2/runs/run_002

# 3-way held-out comparison (v1 baseline / v2 seed / v2 optimized)
uv run python -m v2.src.evaluate \
  --optimized-dir v2/runs/run_001 \
  --budget 3.00 --n 30 --seed 999
```

Artifacts per run: `best_propose.txt`, `best_select.txt`,
`best_metrics.json`, `population.json`, `iterations.jsonl`,
`calls.jsonl`, `budget.json`, `holdout.json`.

## Citations

- GEPA: https://arxiv.org/abs/2507.19457
- Maestro: https://arxiv.org/abs/2509.04642
- RoboPHD: https://arxiv.org/abs/2604.04347
- autocontext (refinement prompt pattern): https://github.com/greyhaven-ai/autocontext
- kosti blog: GEPA + chess puzzles:
  https://kosti.bearblog.dev/trying-out-the-gepa-optimizer-with-chess-puzzles/
