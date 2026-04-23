"""Evaluate a system prompt by playing one move per position and grading."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import chess

from .budget import Budget
from .grader import Grader
from .player import Player

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
    played_eval_cp: int | None
    best_eval_cp: int
    fmt_ok: bool          # parseable JSON with "move" field
    raw_response: str


@dataclass
class EvalReport:
    prompt: str
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
        """Higher = better on all three: (-cp_loss, legal_rate, fmt_rate)."""
        return (-self.mean_cp_loss, self.legal_rate, self.fmt_rate)


def load_positions(path: Path = DATA_PATH) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _is_fmt_ok(raw: str) -> bool:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:].strip()
    try:
        data = json.loads(text)
        return isinstance(data, dict) and "move" in data
    except Exception:
        return False


def _evaluate_one(player: Player, grader: Grader, pos: dict) -> PositionScore:
    res = player.get_move(pos["fen"])
    g = grader.grade(pos["fen"], res.move_uci)
    return PositionScore(
        fen=pos["fen"], tag=pos["tag"], phase=pos["phase"],
        move_uci=res.move_uci, legal=g.legal, cp_loss=g.centipawn_loss,
        best_uci=g.best_uci, played_eval_cp=g.played_eval_cp,
        best_eval_cp=g.best_eval_cp,
        fmt_ok=_is_fmt_ok(res.raw_response),
        raw_response=res.raw_response,
    )


def evaluate(prompt: str, positions: list[dict], budget: Budget,
             *, n_workers: int = 8, tag: str = "eval") -> EvalReport:
    """Run one move per position. Stockfish is single-threaded per process —
    we use one Grader per worker to avoid contention."""
    player = Player(prompt, budget=budget, tag=tag)
    graders = [Grader() for _ in range(n_workers)]
    for g in graders:
        g._engine_or_open()

    try:
        scores: list[PositionScore | None] = [None] * len(positions)

        def task(i_pos):
            i, pos = i_pos
            g = graders[i % n_workers]
            scores[i] = _evaluate_one(player, g, pos)

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            list(ex.map(task, list(enumerate(positions))))
    finally:
        for g in graders:
            g.close()

    return EvalReport(prompt=prompt, scores=[s for s in scores if s is not None])


def report_to_dict(rep: EvalReport) -> dict:
    return {
        "mean_cp_loss": rep.mean_cp_loss,
        "legal_rate": rep.legal_rate,
        "fmt_rate": rep.fmt_rate,
        "scores": [asdict(s) for s in rep.scores],
    }
