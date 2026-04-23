# Program — what the optimizer is doing

(Seed doc, in the spirit of karpathy/autoresearch's `program.md`.)

## Objective
Lower the **mean centipawn loss** (CP-loss) of `gpt-5.4-mini` (reasoning_effort=low)
on a fixed set of 40 chess positions, while keeping **legal-move rate = 1.0**
and **JSON format rate = 1.0**.

## Signal
For each (position, played_move), Stockfish at depth 14 returns:
- `best_move`, `best_eval_cp`
- `played_eval_cp` (eval after the played move, from mover's POV)
- `cp_loss = max(0, best_eval_cp - played_eval_cp)`, clipped at 1000

CP-loss is the primary fitness. Legal-rate and format-rate are hard floors
(treated as Pareto co-objectives so a regression on either is rejected).

## What gets optimized
A single string: the system prompt passed to the LLM. The seed is the
v1 prompt at `chess_prompt.py` (verbatim). Mutations are produced by a
reflector LLM (same model, low reasoning) shown the worst-N traces of the
parent prompt.

## Loop
```
seed -> evaluate -> [iter 16 times: pick parent from Pareto front,
                     reflect on its 8 worst positions -> child,
                     evaluate child, truncate population to P=6]
```

## Definition of "improvement"
On a held-out 40-position sample (different RNG seed):
- mean CP-loss drops by >= 30 points vs baseline, AND
- legal-move rate >= 0.95, AND
- format rate >= 0.95.

## What is OUT of scope
- Engine tool calls at inference time (would be flavor B from the spec).
- Few-shot examples in the user message (this would change the harness).
- Prompts >6000 chars (cost + latency).
