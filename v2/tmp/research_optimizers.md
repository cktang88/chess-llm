# Research: Prompt Optimizer for Chess-Playing LLM

Budget: ~$10 at ~$0.001/call -> ~10,000 LLM calls total (task model + reflection model combined). Single system-prompt target. Feedback per move = Stockfish centipawn loss (dense scalar) plus we can cheaply generate text feedback from (top-engine-move, played-move, CP-loss, phase).

## 1. TL;DR recommendation

**Use GEPA (Reflective Prompt Evolution), rolled as a ~200-line custom implementation, with Pareto selection and truncation.** GEPA fits our regime: single textual artifact, scalar + rich text feedback, tiny rollout budget (beats MIPROv2 +13%, GRPO +20% with 35x fewer rollouts). Chess gives an unusually strong feedback channel — every move yields a CP-loss number and a natural-language critique ("Nf3 lost 120cp; engine prefers Bxh7+, f7 is weak") — exactly what GEPA's reflection step consumes. RoboPHD's Elo tournament needs 1,500 evals and assumes pairwise self-play, not a fixed oracle; Maestro's graph-edit operators are wasted on a single prompt. OPRO, Promptbreeder, TextGrad predate GEPA and are dominated on instruction-only + textual feedback. ADAS/AlphaEvolve optimize code, not prompts, with 10–100x our budget.

## 2. Comparison table

| Algorithm       | Min rollouts to work | Code complexity | Impl available               | Uses dense scalar+text feedback? | Single-prompt fit            |
| --------------- | -------------------- | --------------- | ---------------------------- | -------------------------------- | ---------------------------- |
| **GEPA**        | ~100–200             | Medium (~300 LoC) | `pip install gepa`, DSPy   | **Yes, first-class**             | **Excellent**                |
| MIPROv2         | ~500–1000+           | High            | DSPy                         | Scalar only                      | OK but wants >=200 trainset  |
| BootstrapFewShot| ~50                  | Low             | DSPy                         | Scalar only                      | Weak — only edits demos, not instructions |
| OPRO            | ~200–500             | Low             | DeepMind repo                | Scalar only (score-sorted history) | OK but obsolete vs GEPA     |
| Promptbreeder   | ~2000+               | High            | No canonical impl            | Scalar only                      | OK, heavy; needs big pop     |
| TextGrad        | ~100–300             | Medium          | `pip install textgrad`       | Text-gradient only (no scalar)   | OK, but single-node = overkill |
| ADAS / Meta-AS  | ~1000–5000           | Very high       | GitHub repo                  | Scalar; meta-agent writes code   | Poor — optimizes code graphs |
| AlphaEvolve     | ~10k+                | Very high       | Closed; CodeEvolve OSS clone | Scalar                           | Poor — optimizes code        |
| RoboPHD         | 1,500 (paper default)| High            | MIT toolkit (recent)         | Scalar via Elo pairwise          | Poor — assumes A-vs-B evals  |
| Maestro         | Fewer than GEPA (claimed) | Very high  | Not released at time of writing | Text + scalar, trace-driven   | Overkill — targets graphs    |

## 3. Key insights from RoboPHD and Maestro

### RoboPHD (arxiv 2604.04347)
- **Elo tournament selection** drives survival from pairwise agent comparisons; removes the train/val split ("validation-free evolution").
- **Self-instrumenting seeds**: seed prompt includes diagnostic scaffolding the evolver expands. Good practice to steal: seed should output reasoning tags (chosen-move, candidate-moves, plan) that Stockfish feedback can latch onto.
- **Published budget = 1,500 evals**, ~10x ours. Tournament machinery only pays off with many head-to-head games; at 6 candidates it's just truncation with extra noise.
- **Worth porting?** Validation-free idea: **yes** — with a 40-position set don't split train/val, score on all. Elo itself: **no**.
- Unbounded seed growth (22->1013 lines): skip — we cap prompt length for context cost.

### Maestro (arxiv 2509.04642)
- **Joint graph + config optimization**: edits modules *and* their prompts. Irrelevant to a single-node system.
- **Reflective trace-driven feedback that prioritizes edits** — same core idea as GEPA, extended with edit-priority. Claims "far fewer rollouts than GEPA" and +4.9% over GEPA.
- **Prompt-only mode beats GEPA by ~2.4%**: small delta, full paper/code not released — don't reimplement on trust.
- **Worth porting?** One trick: ask the reflection LM to rank *which section of the prompt* to edit (opening-principles vs tactical-checklist vs output-format). Cheap addition that matches chess's phase decomposition.
- Graph edits, merge operators: inapplicable.

## 4. Concrete algorithm sketch (budget-sized for $10)

**Budget accounting.** We target ~3k input + ~200 output tokens per task call. At gpt-5.4-mini low-reasoning prices (assume ~$0.001/call), ~$10 gives ~10k calls. Split:

- Evaluation calls: one LLM move per position. Use 40-position eval set, ~10 moves scored per position = ~400 task calls per full evaluation. Budget ~20 such evaluations -> ~8,000 calls.
- Reflection calls: ~1 per iteration (longer; use same model to stay in budget) -> ~20 calls, negligible cost.
- Margin: ~2,000 calls for tie-breaks / re-eval.

**Hyperparameters.**
- Population `P = 6` candidates (not 20 — we cannot afford it).
- Iterations `T = 16`.
- Eval set `N = 40` positions, drawn from diverse openings/middlegames/endgames, fixed across run.
- Minibatch for reflection `m = 8` positions (sample the 8 worst-CP-loss positions for that candidate — "hard cases first").
- Reflection LM = same gpt-5.4-mini (cost-matched), temperature 1.0.
- Selection = **Pareto on (CP-loss, legal-move-rate, format-compliance)** — three-objective Pareto. Chess needs legal-move and format as hard floors; CP-loss is the continuous score.

**Pseudocode.**

```python
# Pareto-reflective evolution for single system prompt. ~50 lines.
SEED = "<baseline chess prompt>"
POP  = [SEED]                        # candidate pool
SCORES = {SEED: evaluate(SEED, EVAL_SET)}  # dict prompt -> per-position (cp_loss, legal, fmt)

def evaluate(prompt, positions):
    # returns list of per-position dicts: {cp_loss, legal, fmt, move, trace}
    # cost: len(positions) task-LM calls
    return [run_one(prompt, p) for p in positions]

def aggregate(scores):
    cp   = mean(s["cp_loss"]        for s in scores)
    leg  = mean(s["legal"]          for s in scores)
    fmt  = mean(s["fmt"]            for s in scores)
    return (-cp, leg, fmt)           # higher is better on all three

def pareto_front(pop):
    return [c for c in pop if not any(dominates(aggregate(SCORES[o]),
                                                 aggregate(SCORES[c]))
                                       for o in pop if o != c)]

def reflect(parent, worst_cases):
    # worst_cases: list of (fen, played_move, engine_best, cp_loss, short_engine_note)
    # Prompt the reflection LM with: parent prompt + trace table + Maestro trick:
    # "Which section (principles/tactics/format) is to blame? Propose a minimal edit."
    return llm_reflect(parent, worst_cases)   # one reflection call

for t in range(T):                   # 16 iterations
    # 1. Pick parent: uniform from current Pareto front (GEPA-style)
    parent = random.choice(pareto_front(POP))

    # 2. Build reflection minibatch: 8 worst positions for this parent
    worst = sorted(SCORES[parent], key=lambda s: s["cp_loss"], reverse=True)[:8]

    # 3. Reflect -> child
    child = reflect(parent, worst)
    if child in SCORES: continue      # dedupe cheaply by hash

    # 4. Evaluate child on full EVAL_SET
    SCORES[child] = evaluate(child, EVAL_SET)
    POP.append(child)

    # 5. Truncate: keep Pareto front + top-(P - |front|) by -cp_loss
    front = pareto_front(POP)
    rest  = [c for c in POP if c not in front]
    rest.sort(key=lambda c: aggregate(SCORES[c]), reverse=True)
    POP   = front + rest[:max(0, P - len(front))]

    if early_stop(plateau_window=4): break    # stop if best cp_loss hasn't moved in 4 iters

best = max(POP, key=lambda c: aggregate(SCORES[c]))
return best
```

**Rollout total.** Initial eval: 40. Each of 16 iters: 1 reflection + up to 40 child evals = ~656 task calls in the worst case. Adding dedup/plateau cutoff brings realistic total to 400–600 task calls and ~16 reflection calls. Well under budget; leaves headroom for a second run with a different seed prompt.

## 5. DSPy-or-not

**Roll our own.** Reasons:

1. **DSPy's GEPA wants `dspy.Module` with typed Signatures.** Our system is one LLM call returning UCI; wrapping forces DSPy's LM/adapter abstraction. Direct SDK calls are simpler.
2. **Standalone `pip install gepa` is closer to right-sized** (`seed_candidate={"system_prompt": ...}` matches our shape). If we *must* use a library, use standalone `gepa`, not `dspy.GEPA`.
3. **Maestro-style "which section to edit" tweak** is 10 lines in a handwritten reflect(); awkward as a `gepa.ReflectionConfig` subclass.
4. **Debuggability under 2-hour deadline**: 200 lines is diff-able; GEPA's Pareto bookkeeping, merge candidates, tracking internals are not.
5. **Rich Stockfish feedback formatting** (PV, eval delta, SAN tactical motifs) needs direct control — library adapter gets in the way.

**Reconsider if:** rolled version is thrashing after 1–2 iterations — fall back to `pip install gepa` with `auto="light"` and DefaultAdapter; it encodes merge dedupe, Pareto epsilon, and minibatch sampling edge cases we'd otherwise rediscover.

## 6. Citations

- GEPA paper: https://arxiv.org/abs/2507.19457
- GEPA project site: https://gepa-ai.github.io/gepa/
- GEPA in DSPy: https://dspy.ai/api/optimizers/GEPA/overview/
- Maestro: https://arxiv.org/abs/2509.04642
- RoboPHD: https://arxiv.org/abs/2604.04347
- OPRO: https://arxiv.org/abs/2309.03409
- Promptbreeder: https://arxiv.org/abs/2309.16797
- TextGrad (Nature): https://arxiv.org/abs/2406.07496
- ADAS (ICLR 2025): https://arxiv.org/abs/2408.08435
- AlphaEvolve: https://arxiv.org/abs/2506.13131
- DSPy MIPROv2 docs: https://dspy.ai/api/optimizers/MIPROv2/
- DSPy optimizers overview: https://dspy.ai/learn/optimization/optimizers/
