"""LLM chess player wrapper. Same model/format as v1 ai_controller.py but with
parameterized system prompt and budget tracking."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import chess
from openai import OpenAI

from .budget import Budget

DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_REASONING_EFFORT = "low"


@dataclass
class MoveResult:
    move_uci: str | None  # None on parse/illegal failure
    raw_response: str
    reasoning: str | None
    input_tokens: int
    output_tokens: int


class Player:
    def __init__(
        self,
        system_prompt: str,
        budget: Budget,
        *,
        client: OpenAI | None = None,
        model: str = DEFAULT_MODEL,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        tag: str = "player",
    ):
        self.system_prompt = system_prompt
        self.budget = budget
        self.client = client or OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.tag = tag

    def _user_message(self, fen: str) -> str:
        board = chess.Board(fen)
        legal = [m.uci() for m in board.legal_moves]
        return (
            f"Current Position (FEN): {fen}\n\n"
            f"Side to move: {'White' if board.turn else 'Black'}\n"
            f"Legal moves ({len(legal)}): {', '.join(legal)}\n\n"
            "Choose your move. Respond with JSON ONLY:\n"
            '{"move": "<uci>", "reasoning": "<one short sentence>"}'
        )

    def get_move(self, fen: str) -> MoveResult:
        board = chess.Board(fen)
        user = self._user_message(fen)
        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user},
            ],
        )
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort

        resp = self.client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        in_toks = getattr(resp.usage, "prompt_tokens", 0)
        out_toks = getattr(resp.usage, "completion_tokens", 0)
        self.budget.charge(
            tag=self.tag, model=self.model,
            input_tokens=in_toks, output_tokens=out_toks,
        )

        move_uci, reasoning = _parse_move(text, board)
        return MoveResult(move_uci=move_uci, raw_response=text, reasoning=reasoning,
                          input_tokens=in_toks, output_tokens=out_toks)


def _parse_move(text: str, board: chess.Board) -> tuple[str | None, str | None]:
    """Try JSON first, fall back to scanning for any legal UCI token."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
        if cleaned.lstrip().startswith("json"):
            cleaned = cleaned.lstrip()[4:].strip()
    try:
        data = json.loads(cleaned)
        mv = str(data.get("move", "")).strip()
        reasoning = data.get("reasoning")
        try:
            move = chess.Move.from_uci(mv)
            if move in board.legal_moves:
                return move.uci(), reasoning
        except Exception:
            pass
    except json.JSONDecodeError:
        pass

    # Fallback: any legal UCI substring
    for tok in text.replace(",", " ").replace('"', " ").split():
        tok = tok.strip(".,!?;:()[]{}\"' ").lower()
        try:
            mv = chess.Move.from_uci(tok)
            if mv in board.legal_moves:
                return mv.uci(), None
        except Exception:
            continue
    return None, None


def call_llm_text(
    *, prompt: str, budget: Budget, tag: str,
    client: OpenAI | None = None,
    model: str = DEFAULT_MODEL, reasoning_effort: str | None = "low",
) -> str:
    """Generic text completion helper used by the optimizer (reflector calls)."""
    client = client or OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    kwargs = dict(model=model, messages=[{"role": "user", "content": prompt}])
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    resp = client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content or ""
    budget.charge(
        tag=tag, model=model,
        input_tokens=getattr(resp.usage, "prompt_tokens", 0),
        output_tokens=getattr(resp.usage, "completion_tokens", 0),
    )
    return text
