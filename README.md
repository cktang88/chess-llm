# Chess LLM

A chess game where you play against an LLM. Two variants live in this repo.

## Headline result

A hand-rolled multi-component GEPA optimizer evolved the v2 system prompts
against Stockfish on a $10 budget. Evaluated on **40 held-out positions
the optimizer never saw** (`seed=999`):

| Setup | mean cp_loss | legal | fmt | blunder >200cp | perfect =0cp |
|---|---|---|---|---|---|
| v1 baseline (single-call, hand-written prompt) | 116.8 | 0.97 | 1.00 | 15% | 15% |
| v2 pipeline with unoptimized seed prompts | 141.5 | 1.00 | 1.00 | 23% | 10% |
| **v2 pipeline with GEPA-evolved prompts** | **50.0** | 1.00 | 0.97 | **3%** | **25%** |

**v1 → v2-optimized: −66.8 cp_loss (−57%), 5× fewer blunders, 67% more
perfect moves.** Surprising finding: the harness change alone (v2 seed row)
is actually a pessimization — GEPA's prompt evolution is doing all the
real work. Full writeup: [**v2/METHODOLOGY.md**](v2/METHODOLOGY.md).

## Variants

- **v1** — interactive pygame GUI, one LLM call per AI move with a
  hand-written tactical prompt. Lives in the repo root + `python-chess-ai-yt/`.
- **v2** — the GEPA optimizer that produced the result above, plus a
  single-file drop-in controller
  (`python-chess-ai-yt/src/ai_controller_v2.py`) you can swap into the
  GUI to **play against the optimized prompts directly**. Lives in
  [`v2/`](v2/).

### Quickstart: play against v2 in the GUI

```bash
uv sync
echo "OPENROUTER_API_KEY=sk-or-v1-..." > .env  # v2 uses OpenRouter

# Swap v1 -> v2 with a one-line edit in python-chess-ai-yt/src/game.py:
#   from ai_controller import AIController
# becomes:
#   from ai_controller_v2 import AIControllerV2 as AIController

uv run python run_gui.py
```

`ai_controller_v2.py` is self-contained: the GEPA-optimized PROPOSE and
SELECT prompts are embedded as string constants, no `v2/` runtime imports.

---

## v1 — Play chess against GPT in a pygame GUI

The original, interactive game. One LLM call per AI move using the
hand-written tactical prompt in [`chess_prompt.py`](chess_prompt.py).

### Setup

Requirements: Python 3.10+, `uv` (or `pip`), an OpenAI API key.

```bash
uv sync
echo "OPENAI_API_KEY=sk-..." > .env
```

### Run

```bash
uv run python run_gui.py
```

On startup, you'll be asked whether to play against the AI and which color
to play. Then drag pieces with the mouse.

Keyboard controls:
- `T` — change theme
- `R` — reset game
- `A` — toggle AI on/off mid-game

### v1 file layout

```
chess-llm/
├── run_gui.py                      # GUI launcher
├── chess_prompt.py                 # single tactical system prompt
├── test_integration.py             # FEN + UCI sanity checks
├── python-chess-ai-yt/
│   └── src/
│       ├── main.py                 # pygame main loop
│       ├── game.py                 # game logic + AI integration
│       ├── board.py                # board with FEN conversion
│       ├── ai_controller.py        # calls GPT-5.2 for each AI move
│       └── …                       # GUI components (pieces, squares, themes)
└── v2/                             # see next section
```

### How v1 works

1. The pygame board is serialized to FEN.
2. `ai_controller.py` sends the FEN + legal move list + [`chess_prompt.py`](chess_prompt.py)
   to `gpt-5.2` (reasoning_effort=low) and asks for a JSON `{"move": "e2e4",
   "reasoning": "…"}` response.
3. The UCI move is parsed, validated against legal moves, and applied to
   the pygame board.

v1 code is self-contained in the repo root + `python-chess-ai-yt/` and is
**not modified by anything in `v2/`.**

---

## v2 — Stockfish-graded prompt optimization

v2 does not change the game UI. It evolves **better system prompts** for a
richer inference harness so the same class of model plays stronger chess.

### What's different from v1

| Aspect | v1 | v2 |
|---|---|---|
| Provider | OpenAI directly | OpenRouter (`openai/gpt-5.4-mini`) |
| LLM calls per move | 1 | Up to 4 (3 parallel PROPOSE + 1 SELECT) |
| Prompt | hand-written | GEPA-evolved (reflective mutation) |
| Diversity at inference | temperature only | temperature + style-hint rotation + self-consistency + LLM verify-and-select |
| Graded by | nothing | Stockfish at depth 14 |
| Inference-time engine use | none | none (pure LLM) |

### Setup

```bash
uv sync
echo "OPENROUTER_API_KEY=sk-or-v1-..." > .env
brew install stockfish                # /opt/homebrew/bin/stockfish
```

### Run the optimizer + eval

```bash
# Run 1: optimize from scratch (budget-capped)
uv run python -m v2.src.optimizer \
  --budget 7.50 --T 12 --n-eval 30 --workers 4 --n-propose 3 \
  --out v2/runs/run_001

# Held-out comparison: v1 baseline / v2 seed pipeline / v2 optimized
uv run python -m v2.src.evaluate \
  --optimized-dir v2/runs/run_001 \
  --budget 3.00 --n 30 --seed 999

# Optional: diversity-enhanced follow-up from run_001's best
uv run python -m v2.src.optimizer \
  --budget 4.50 --T 12 --n-eval 30 --workers 4 --n-propose 3 \
  --warm-start-from v2/runs/run_001 \
  --force-module-rotation alt-select-first \
  --reflect-temperature 1.0 --seed 1 \
  --out v2/runs/run_002

# Run_003: noise-aware + stronger reflector + reflector memory
# (addresses the three plateau hypotheses — see v2/METHODOLOGY.md §8)
uv run python -m v2.src.optimizer \
  --budget 5.00 --T 12 --n-eval 40 --workers 4 --n-propose 3 \
  --warm-start-from v2/runs/run_002 \
  --force-module-rotation alt-select-first \
  --reflect-temperature 1.0 \
  --reflect-model openai/gpt-5.4 \
  --history-n 5 --accept-sigma 1.5 --seed 2 \
  --out v2/runs/run_003
```

### v2 file layout

```
v2/
├── METHODOLOGY.md                   # full writeup — start here
├── README.md                        # short harness diagram + commands
├── plan.md                          # phase-by-phase build log
├── program.md                       # seed doc: objective, signal, loop
├── data/
│   └── positions.jsonl              # 44 graded positions (openings + tactics + endgames)
├── src/
│   ├── budget.py                    # $10 cap + per-call ledger (uses OpenRouter cost)
│   ├── llm.py                       # OpenRouter client + chat helper
│   ├── grader.py                    # Stockfish depth=14, multipv=5
│   ├── dataset.py                   # position dataset generator
│   ├── seed_prompts.py              # PROPOSE + SELECT seed prompts
│   ├── pipeline.py                  # 2-module harness (PROPOSE×3 + SELECT)
│   ├── player.py                    # single-call player (used only for v1 baseline grading)
│   ├── evaluator.py                 # parallel evaluate(prompts, positions) → report
│   ├── optimizer.py                 # hand-rolled multi-prompt GEPA
│   └── evaluate.py                  # 3-way held-out comparison
├── runs/
│   └── run_001/                     # per-run artifacts
│       ├── best_propose.txt         # optimized PROPOSE prompt
│       ├── best_select.txt          # optimized SELECT prompt
│       ├── best_metrics.json        # per-objective scores of best PromptSet
│       ├── population.json          # all surviving candidates, scored
│       ├── iterations.jsonl         # per-iter diagnosis + module + acceptance
│       ├── calls.jsonl              # every LLM call with tokens + cost
│       ├── budget.json              # final rollup
│       └── holdout.json             # 3-way held-out comparison output
└── tmp/
    ├── research_optimizers.md       # why GEPA beat Maestro / RoboPHD / etc.
    └── research_repos.md            # review of 4 candidate repos
```

### Final result (run_004 best, held-out 40 positions, RNG seed = 999)

| Setup | mean cp_loss | legal | fmt | blunder >200cp | perfect =0cp |
|---|---|---|---|---|---|
| v1 baseline (single-call, v1 prompt) | 116.8 | 0.97 | 1.00 | 15% | 15% |
| v2 seed pipeline (unoptimized prompts) | 141.5 | 1.00 | 1.00 | 23% | 10% |
| **v2 optimized (run_001 → run_004 edits)** | **50.0** | 1.00 | 0.97 | **3%** | **25%** |

**v1 → v2 optimized: −66.8 cp_loss (−57%). Blunder rate 15% → 3% (5× fewer).**

Important: the harness change alone (v1 → v2 seed pipeline) is
*neutral-to-negative*. Temperature=1.0 + style-hint rotation on an
unoptimized SELECT produces diverse-but-bad candidates the filter can't
reject. **GEPA prompt evolution (especially of SELECT, via forced module
rotation in run_002) is what makes the pipeline win.**

Full writeup including algorithm choice, budget math, run-by-run history
(run_001–run_004), plateau diagnosis, and lessons learned:
**[v2/METHODOLOGY.md](v2/METHODOLOGY.md)**.

---

## Testing

```bash
# v1 integration
uv run python test_integration.py

# v2 smoke test (no API calls needed)
uv run python -c "from v2.src.evaluator import load_positions; print(len(load_positions()), 'positions')"
```

## Credits

- Pygame chess GUI based on python-chess-ai-yt
- v2 optimization is hand-rolled multi-component GEPA (Agrawal et al. 2025)
  with Stockfish-specific reflection-prompt formatting. See
  [v2/METHODOLOGY.md](v2/METHODOLOGY.md) §4 for the algorithm landscape
  (Maestro, RoboPHD, ADAS, AlphaEvolve) and why we picked GEPA.
