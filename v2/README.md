# v2 — GEPA-optimized multi-module chess-LLM

Optimizes a 2-module chess harness on top of `openai/gpt-5.4-mini`
(reasoning_effort=low, via OpenRouter) so it plays better chess than the v1
single-call baseline. Stockfish grades every move; reflective prompt evolution
improves the prompts inside a $10 budget.

## Harness shape

```
FEN ──┬─► PROPOSE  (temp=1.0, style hint i)  ─► candidate_i
      ├─► PROPOSE  (temp=1.0, style hint j)  ─► candidate_j    (parallel)
      └─► PROPOSE  (temp=1.0, style hint k)  ─► candidate_k
                                │
                                ▼
                  unique legal candidates
                                │
                     ┌──────────┴──────────┐
                     ▼                      ▼
               1 unique?               multiple
                     │                      │
                     ▼                      ▼
                skip SELECT         SELECT (temp=0.0)
                     │                      │
                     └──────┬───────────────┘
                            ▼
                       chosen UCI move
                            │
                            ▼
                   Stockfish depth=14, multipv=5
                            │
                            ▼
                cp_loss = best_eval - played_eval
```

Per move: **3 parallel propose calls + 0 or 1 select call**.

## Optimizer

Hand-rolled GEPA (`src/optimizer.py`) — picked by comparison in
`tmp/research_optimizers.md`. Each iteration:

1. Sample a parent `{propose, select}` from the Pareto front of the current
   population (P=6 max).
2. Build a reflection minibatch: the parent's 8 worst-CP-loss positions.
3. Call the reflector LM on the parent + trace. It must return
   `{diagnosis, module_to_edit, revised_prompt}` — only ONE module is edited
   per step (Maestro-lite edit-priority).
4. Evaluate the child `PromptSet` on the full eval set (N=30 positions).
5. Truncate the population to Pareto front + top-(P-|front|) by aggregate.

Pareto objectives (all maximize): `(-mean_cp_loss, legal_rate, fmt_rate)`.

Budget is tracked with OpenRouter's reported per-call cost where available,
else a token-price estimate. Hard abort on cap.

## Running

```bash
# Optimize
uv run python -m v2.src.optimizer \
    --budget 7.50 --T 12 --n-eval 30 --workers 4 --n-propose 3 \
    --out v2/runs/run_001

# 3-way held-out comparison (different RNG seed)
uv run python -m v2.src.evaluate \
    --optimized-dir v2/runs/run_001 \
    --budget 1.50 --n 30
```

Artifacts in `v2/runs/<run>/`:

| File | What |
|---|---|
| `best_propose.txt` | Optimized PROPOSE prompt |
| `best_select.txt`  | Optimized SELECT prompt |
| `best_metrics.json` | Per-objective score of the best PromptSet |
| `population.json`  | All surviving PromptSets, scored |
| `iterations.jsonl` | Per-iter diagnosis + module edited + accept/reject |
| `calls.jsonl`      | Every LLM call with token count + cost |
| `budget.json`      | Final rollup |
| `holdout.json`     | After `evaluate.py`: 3-way comparison |

## Not touched

- `../chess_prompt.py`, `../ai_controller.py`, the game GUI — v1 still works
  as before.
