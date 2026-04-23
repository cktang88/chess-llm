"""Stockfish-based grader: scores a played UCI move on a FEN by centipawn loss."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import chess
import chess.engine

STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
DEFAULT_DEPTH = 14
MATE_SCORE_CP = 10_000  # value used when mapping mate scores to centipawns
LOSS_CLIP = 1000  # cap centipawn loss to keep one blunder from dominating


@dataclass
class MoveCandidate:
    uci: str
    eval_cp: int  # from the side-to-move perspective; higher = better for mover


@dataclass
class GradeResult:
    fen: str
    played_uci: str | None
    legal: bool
    best_uci: str
    best_eval_cp: int
    played_eval_cp: int | None
    centipawn_loss: int  # clipped, >= 0
    top_k: list[MoveCandidate] = field(default_factory=list)

    def feedback_text(self) -> str:
        lines = []
        if not self.legal:
            lines.append(f"ILLEGAL move: {self.played_uci!r} is not legal in this position.")
        else:
            lines.append(
                f"You played {self.played_uci} (eval {self.played_eval_cp:+d}cp). "
                f"Stockfish best was {self.best_uci} (eval {self.best_eval_cp:+d}cp). "
                f"Centipawn loss: {self.centipawn_loss}."
            )
        if self.top_k:
            lines.append("Top engine candidates (UCI, eval cp from your perspective):")
            for c in self.top_k:
                marker = "  <-- you" if c.uci == self.played_uci else ""
                lines.append(f"  {c.uci}: {c.eval_cp:+d}{marker}")
        return "\n".join(lines)


def _score_to_cp(score: chess.engine.PovScore) -> int:
    """Map a python-chess score (from side-to-move POV) to centipawns."""
    pov = score.relative
    if pov.is_mate():
        m = pov.mate()
        # closer mate -> larger magnitude
        return MATE_SCORE_CP - abs(m) if m > 0 else -(MATE_SCORE_CP - abs(m))
    return pov.score()


class Grader:
    def __init__(self, stockfish_path: str = STOCKFISH_PATH, depth: int = DEFAULT_DEPTH, multipv: int = 5):
        self.path = stockfish_path
        self.depth = depth
        self.multipv = multipv
        self._engine: chess.engine.SimpleEngine | None = None

    def __enter__(self) -> "Grader":
        self._engine = chess.engine.SimpleEngine.popen_uci(self.path)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._engine is not None:
            try:
                self._engine.quit()
            finally:
                self._engine = None

    def _engine_or_open(self) -> chess.engine.SimpleEngine:
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.path)
        return self._engine

    def grade(self, fen: str, played_uci: str | None) -> GradeResult:
        eng = self._engine_or_open()
        board = chess.Board(fen)
        limit = chess.engine.Limit(depth=self.depth)
        infos = eng.analyse(board, limit, multipv=self.multipv)

        top_k: list[MoveCandidate] = []
        for info in infos:
            mv = info["pv"][0]
            top_k.append(MoveCandidate(uci=mv.uci(), eval_cp=_score_to_cp(info["score"])))

        best = top_k[0]

        # Validate played move
        if played_uci is None:
            return GradeResult(
                fen=fen, played_uci=None, legal=False,
                best_uci=best.uci, best_eval_cp=best.eval_cp,
                played_eval_cp=None, centipawn_loss=LOSS_CLIP, top_k=top_k,
            )
        try:
            played = chess.Move.from_uci(played_uci)
        except Exception:
            return GradeResult(
                fen=fen, played_uci=played_uci, legal=False,
                best_uci=best.uci, best_eval_cp=best.eval_cp,
                played_eval_cp=None, centipawn_loss=LOSS_CLIP, top_k=top_k,
            )
        if played not in board.legal_moves:
            return GradeResult(
                fen=fen, played_uci=played_uci, legal=False,
                best_uci=best.uci, best_eval_cp=best.eval_cp,
                played_eval_cp=None, centipawn_loss=LOSS_CLIP, top_k=top_k,
            )

        # Eval of the played move = -engine_eval(after move) from the new side-to-move's POV
        board.push(played)
        info_after = eng.analyse(board, limit)
        played_eval = -_score_to_cp(info_after["score"])
        loss = max(0, best.eval_cp - played_eval)
        loss = min(loss, LOSS_CLIP)

        # If played_uci matches one in top_k, prefer that listing's eval for consistency
        for c in top_k:
            if c.uci == played_uci:
                played_eval = c.eval_cp
                loss = max(0, min(LOSS_CLIP, best.eval_cp - played_eval))
                break

        return GradeResult(
            fen=fen, played_uci=played_uci, legal=True,
            best_uci=best.uci, best_eval_cp=best.eval_cp,
            played_eval_cp=played_eval, centipawn_loss=loss, top_k=top_k,
        )
