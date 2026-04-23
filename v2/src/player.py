"""Single-call baseline player — used only to grade the v1 prompt, for
apples-to-apples comparison in evaluate.py. The optimization pipeline uses
pipeline.py instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import chess

from .budget import Budget
from .llm import DEFAULT_MODEL, chat, make_client


@dataclass
class MoveResult:
    move_uci: str | None
    raw_response: str
    fmt_ok: bool


def _parse(text: str, board: chess.Board) -> tuple[str | None, bool]:
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        t = "\n".join(lines[1:-1]) if len(lines) > 2 else t
        if t.lstrip().startswith("json"):
            t = t.lstrip()[4:].strip()
    fmt_ok = False
    try:
        data = json.loads(t)
        if isinstance(data, dict) and "move" in data:
            fmt_ok = True
            try:
                mv = chess.Move.from_uci(str(data["move"]).strip())
                if mv in board.legal_moves:
                    return mv.uci(), fmt_ok
            except Exception:
                pass
    except Exception:
        pass
    # fallback scan
    for tok in text.replace(",", " ").replace('"', " ").split():
        tok = tok.strip(".,!?;:()[]{}\"' ").lower()
        try:
            mv = chess.Move.from_uci(tok)
            if mv in board.legal_moves:
                return mv.uci(), fmt_ok
        except Exception:
            continue
    return None, fmt_ok


class SingleCallPlayer:
    """One LLM call per move. Mirrors the v1 harness."""

    def __init__(self, system_prompt: str, budget: Budget, *,
                 model: str = DEFAULT_MODEL, client=None, tag: str = "baseline"):
        self.system_prompt = system_prompt
        self.budget = budget
        self.model = model
        self.client = client or make_client()
        self.tag = tag

    def _user_msg(self, fen: str) -> str:
        board = chess.Board(fen)
        legal = [m.uci() for m in board.legal_moves]
        return (
            f"Current Position (FEN): {fen}\n\n"
            f"Side to move: {'White' if board.turn else 'Black'}\n"
            f"Legal moves ({len(legal)}): {', '.join(legal)}\n\n"
            "Analyze and provide your move. You MUST respond with UCI notation.\n"
            "Respond with JSON:\n"
            '{"move": "<uci>", "reasoning": "<one sentence>"}'
        )

    def get_move(self, fen: str) -> MoveResult:
        board = chess.Board(fen)
        try:
            content, _ = chat(
                messages=[{"role": "system", "content": self.system_prompt},
                          {"role": "user", "content": self._user_msg(fen)}],
                budget=self.budget, tag=self.tag, client=self.client, model=self.model,
            )
        except Exception as e:
            return MoveResult(None, f"ERROR: {e}", False)
        move_uci, fmt_ok = _parse(content, board)
        return MoveResult(move_uci, content, fmt_ok)
