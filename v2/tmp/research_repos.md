# Repo research: four "or similar" candidates for chess-prompt GEPA

Goal: optimize a single system prompt for a chess-playing LLM, graded by Stockfish, $10 budget, pure LLM. Below: what each repo actually is, and what (if anything) we should take.

---

## 1. `huggingface/ml-intern`

- **What it actually does:** An autonomous Hugging Face "ML engineer" CLI agent. A submission queue feeds a 300-iteration `run_agent()` loop that does LLM-call → parse tool calls → approval gate → tool execution → context update. Tooling covers HF docs/repos/datasets, GitHub search, sandbox execution, and MCP servers. Headline features are a `ContextManager` with auto-compaction at 170k tokens and a "Doom Loop Detector" that injects corrective prompts when tool-call patterns repeat.
- **License:** None declared (`license: null` in the GitHub API). Not safe to vendor.
- **Reusable?** Reference only — CLI app with internal events, not designed as a library. Key files are `agent/main.py`, `agent/core/`, `agent/prompts/`, `configs/`.
- **Worth stealing:** The Doom Loop Detector idea is cute for long agent runs, but irrelevant here — we score a prompt on positions, not run a tool-using agent. **Nothing relevant** to single-prompt optimization.

## 2. `greyhaven-ai/autocontext`

- **What it actually does:** A "recursive self-improving harness" for agents with a multi-role loop: competitor proposes strategy, analyst explains what happened, coach updates playbooks, architect suggests improvements, curator gates what persists. Weak changes are rolled back; successful ones accumulate as reusable knowledge. Published as installable Python (`autocontext`) and TypeScript (`autoctx`) packages with a CLI (`autoctx solve`, `autoctx serve`). This is the closest match to GEPA in the list.
- **License:** Apache-2.0. Safe to vendor or copy with attribution.
- **Reusable?** Technically a library, but the `src/autocontext/` package is sprawling (40+ subpackages: `agentos`, `consultation`, `evaluation`, `harness`, `loop`, `openclaw`, `rlm`, `scenarios`, `training`, `tournament`…). Pulling it in whole would dwarf our $10 project. Best treated as a reference for design patterns.
- **Worth stealing:** `src/autocontext/loop/refinement_prompt.py` — specifically `build_refinement_prompt(...)`. Its signature is exactly the GEPA reflective step: `(scenario_rules, strategy_interface, evaluation_criteria, parent_strategy, match_feedback, current_playbook, score_trajectory, operational_lessons)`. The template structure ("You are refining an existing strategy, not creating one from scratch. Keep what works, fix what doesn't.") is a clean, steal-worthy pattern for our mutation prompt. Also worth a look: `loop/tournament_helpers.py` (Pareto-ish candidate selection) and `loop/cost_control.py` (budget gating — relevant to our $10 cap).

## 3. `karpathy/autoresearch`

- **What it actually does:** An autonomous *code*-editing loop, not a prompt-evolution loop. An agent edits `train.py` (a nanochat GPT trainer), runs a fixed 5-min training experiment, reads `val_bpb` (validation bits-per-byte), and commits or reverts. Three files total: `train.py` (modified), `prepare.py` (read-only data), `program.md` (baseline instructions for the agent).
- **License:** None declared. Not vendor-safe.
- **Reusable?** Reference only. Deliberately minimal; designed to be read, not imported.
- **Worth stealing:** Two pattern ideas, not code:
  1. The `program.md` convention — a short, human-authored baseline prompt that seeds the loop. We should have our own `program.md`-equivalent checked into `v2/` describing chess objectives, Stockfish scoring, and what "improvement" means.
  2. The fixed per-experiment wall-clock budget (5 min) + keep-or-revert rule. Our analog: fixed token budget per candidate, keep-if-score-improves on a held-out position set. That is basically GEPA anyway, so this is confirmation more than inspiration.

## 4. `davebcn87/pi-autoresearch`

- **What it actually does:** A TypeScript extension + skill framework for the "pi" coding agent. Agent edits code → commits → runs benchmark → logs to `autoresearch.jsonl` + `autoresearch.md` → keeps or reverts. Domain-agnostic; the skill defines the benchmark command and metric. Notable detail: uses **Median Absolute Deviation** on benchmark samples to distinguish real gains from noise.
- **License:** MIT. Vendor-safe.
- **Reusable?** It's a pi extension, not a standalone library. Only reusable inside pi.
- **Worth stealing:** The MAD-based noise filter. If our Stockfish grader is noisy across position sets, rejecting "improvements" that lie within the MAD band of the parent is a cheap, correct gate. This is the only concrete, portable idea in this repo for us. Everything else is pi-specific plumbing.

---

## Verdict

Three of these four (`ml-intern`, `karpathy/autoresearch`, `pi-autoresearch`) are **autonomous code-editing / ML-experiment agents**, not prompt optimizers. They share a keep-or-revert loop structure with GEPA, but none of them do reflective prompt mutation over a population against a held-out metric. For a $10, pure-LLM, single-prompt evolution project, pulling any of them in as a dependency would be absurd overkill. Only `greyhaven-ai/autocontext` is genuinely on-topic — it *is* reflective prompt evolution, Apache-2.0, and has a clean `refinement_prompt.py` we can adapt almost verbatim as our mutation step. Recommendation: **ignore ml-intern, karpathy/autoresearch, and pi-autoresearch entirely**; skim the rest of `autocontext/loop/` for the refinement-prompt template, tournament selection, and cost-control patterns, but do not vendor the package — reimplement the ~30 lines we actually need. Steal two small ideas from the others for free: a `program.md`-style seed doc (karpathy), and MAD-based noise gating on Stockfish scores if we see variance issues (pi-autoresearch).
