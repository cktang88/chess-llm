"""Evaluate a PromptSet by running the 2-module pipeline on each position."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from .budget import Budget
from .grader import Grader
from .pipeline import Pipeline, PipelineMoveResult, PromptSet

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "positions.jsonl"


@dataclass
class PositionScore:
    fen: str
    tag: str
    phase: str
    move_uci: str | None
    legal: bool
    cp_loss: int
    best_uci: str
    best_eval_cp: int
    played_eval_cp: int | None
    fmt_ok: bool
    selector_used: bool
    candidate_moves: list[str | None]       # proposer outputs (UCI or None if illegal)
    candidate_reasonings: list[str]
    error: str | None = None


@dataclass
class EvalReport:
    prompts: PromptSet
    scores: list[PositionScore]

    @property
    def mean_cp_loss(self) -> float:
        return sum(s.cp_loss for s in self.scores) / max(1, len(self.scores))

    @property
    def legal_rate(self) -> float:
        return sum(1 for s in self.scores if s.legal) / max(1, len(self.scores))

    @property
    def fmt_rate(self) -> float:
        return sum(1 for s in self.scores if s.fmt_ok) / max(1, len(self.scores))

    def aggregate(self) -> tuple[float, float, float]:
        """Higher is better on all three: (-cp_loss, legal_rate, fmt_rate)."""
        return (-self.mean_cp_loss, self.legal_rate, self.fmt_rate)


def load_positions(path: Path = DATA_PATH) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _evaluate_one(pipeline: Pipeline, grader: Grader, pos: dict) -> PositionScore:
    res: PipelineMoveResult = pipeline.get_move(pos["fen"])
    g = grader.grade(pos["fen"], res.move_uci)
    return PositionScore(
        fen=pos["fen"], tag=pos["tag"], phase=pos["phase"],
        move_uci=res.move_uci, legal=g.legal, cp_loss=g.centipawn_loss,
        best_uci=g.best_uci, best_eval_cp=g.best_eval_cp,
        played_eval_cp=g.played_eval_cp,
        fmt_ok=res.fmt_ok(),
        selector_used=res.selector_used,
        candidate_moves=[c.move_uci for c in res.candidates],
        candidate_reasonings=[c.reasoning for c in res.candidates],
        error=res.error,
    )


def evaluate(prompts: PromptSet, positions: list[dict], budget: Budget,
             *, n_workers_positions: int = 4, n_propose: int = 3,
             tag: str = "eval") -> EvalReport:
    """Grade `prompts` on `positions`.

    We run `n_workers_positions` positions in parallel; each runs n_propose
    parallel proposer calls + (maybe) one select call. Effective concurrency
    is therefore up to n_workers_positions * (n_propose+1) LLM calls in flight.
    """
    # One Stockfish process per position-worker (engine is single-threaded).
    graders = [Grader() for _ in range(n_workers_positions)]
    for g in graders:
        g._engine_or_open()

    # One Pipeline per position-worker so each uses its own Grader.
    pipelines = [Pipeline(prompts, budget, n_propose=n_propose, tag=tag)
                 for _ in range(n_workers_positions)]

    scores: list[PositionScore | None] = [None] * len(positions)

    def task(idx: int) -> None:
        p = positions[idx]
        worker = idx % n_workers_positions
        scores[idx] = _evaluate_one(pipelines[worker], graders[worker], p)

    try:
        with ThreadPoolExecutor(max_workers=n_workers_positions) as ex:
            futures = [ex.submit(task, i) for i in range(len(positions))]
            for f in as_completed(futures):
                f.result()
    finally:
        for g in graders:
            g.close()

    return EvalReport(prompts=prompts, scores=[s for s in scores if s is not None])


def report_to_dict(rep: EvalReport) -> dict:
    return {
        "mean_cp_loss": rep.mean_cp_loss,
        "legal_rate": rep.legal_rate,
        "fmt_rate": rep.fmt_rate,
        "n_positions": len(rep.scores),
        "scores": [asdict(s) for s in rep.scores],
    }
