# v2 — Optimized LLM Chess Player

## Goal
Make `gpt-5.4-mini` (reasoning_effort=low) play meaningfully stronger chess than the
vanilla prompt in `chess_prompt.py`, by optimizing the prompt / harness with a
GEPA-family algorithm and grading every move with Stockfish.

## Hard constraints (from user)
- Model under optimization: `gpt-5.4-mini`, `reasoning_effort=low`
- Total API spend: **≤ $10**
- Pure LLM (no engine tool-calls at inference time — Stockfish is *only* the grader)
- Fast (overnight at most; ideally < 1 hour)
- Do **not** modify anything outside `v2/`. Existing `chess_prompt.py`, `ai_controller.py`, etc. are untouched

## Approach (high-level)
1. **Per-position grading, not full games.** Game rollouts are expensive and
   high-variance. Centipawn-loss vs Stockfish best-move on a fixed dataset of
   ~50 positions gives dense, low-variance signal at ~50 LLM calls per
   evaluation.
2. **Reflective prompt evolution** (GEPA-style). Each iteration: evaluate
   current prompt(s) on the dataset, collect worst-scoring traces, ask a
   reflector LLM to propose a mutated prompt, keep the Pareto frontier of
   (per-position) scores, sample parents proportional to Pareto rank.
3. **Budget-aware.** Token-level $ accounting with a hard kill-switch at $10.

## Phases

### Phase 0 — Setup (DONE)
- [x] `brew install stockfish` (verified at /opt/homebrew/bin/stockfish)
- [x] Create `v2/{src,data,runs,tmp}` scaffold
- [x] Write this plan

### Phase 1 — Research (DONE)
- [x] **R1**: `v2/tmp/research_optimizers.md` -> recommends hand-rolled GEPA
      (P=6, T=16, eval=40, minibatch=8, Pareto on 3 objectives).
- [x] **R2**: `v2/tmp/research_repos.md` -> only `autocontext` is on-topic;
      stole the "refine, don't rewrite" template + section-edit pattern.
      Three other repos ignored.

### Phase 2 — Infrastructure (DONE)
- [x] `v2/src/grader.py` — Stockfish wrapper, depth=14, multipv=5
- [x] `v2/data/positions.jsonl` — 44 positions (12 openings + 19 curated +
      ~13 self-play snapshots, dedup'd)
- [x] `v2/src/player.py` — `Player(prompt).get_move(fen)` + `call_llm_text`
      reflector helper, both with budget tracking
- [x] `v2/src/budget.py` — `Budget(cap_usd)` with hard kill-switch

### Phase 3 — Optimizer (DONE)
- [x] `v2/src/optimizer.py` — hand-rolled GEPA per R1:
      - Pop=6, T=16, EVAL_SET=40, minibatch=8 worst CP-loss
      - 3-obj Pareto on (-cp_loss, legal_rate, fmt_rate)
      - Reflector = same gpt-5.4-mini, low reasoning, with Maestro-style
        section-pick hint and autocontext-style "refine, don't rewrite"
      - Snapshots after every iter to `v2/runs/<name>/`
- [x] `v2/src/evaluator.py` — parallel grading (8 workers, 8 Stockfish procs)
- [x] `v2/program.md` — seed doc (autoresearch-style)

### Phase 4 — Run (BLOCKED on user: provide OPENAI_API_KEY + go-ahead)
- [ ] `OPENAI_API_KEY=… uv run python -m v2.src.optimizer --budget 9.00 --out v2/runs/run_001`

### Phase 5 — Eval (BLOCKED on Phase 4)
- [x] `v2/src/evaluate.py` — held-out comparison (different RNG seed),
      reports cp_loss / legal / fmt / blunder_rate / perfect_rate, baseline
      vs optimized
- [ ] Run: `uv run python -m v2.src.evaluate --best-prompt v2/runs/run_001/best_prompt.txt --out v2/runs/run_001/holdout.json`
- [ ] (Stretch, deferred) Short games vs Stockfish skill_level=3

## Open decisions (resolve after Phase 1)
- Whether to depend on `dspy-ai` (has GEPA built in) vs roll a 200-line
  optimizer ourselves. Lean toward roll-our-own for speed and no extra deps,
  unless R1 strongly recommends DSPy.
- Reflector model: same gpt-5.4-mini (cheap) vs a stronger model for
  mutation only (better proposals, ~10× cost per call but rare). Start with
  same-model.

## Out of scope
- Engine tool-calls at inference (would be flavor B from the discussion).
- RL / fine-tuning weights.
- Anything that touches files outside `v2/`.
