"""2-module chess pipeline: propose (parallel × N) → select.

- Module 1 (PROPOSE): gpt-5.4-mini called N=3 times in parallel at temp=1.0 to
  elicit diverse candidate moves.
- Module 2 (SELECT):  called once on the de-duplicated legal candidates to pick
  the best. If only one unique legal candidate exists, we skip SELECT.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import chess

from .budget import Budget
from .llm import DEFAULT_MODEL, chat, make_client


@dataclass
class PromptSet:
    propose: str
    select: str

    def to_dict(self) -> dict[str, str]:
        return {"propose": self.propose, "select": self.select}

    def key(self) -> str:
        """Deterministic hash key for dedupe/snapshot."""
        import hashlib
        h = hashlib.sha1()
        h.update(self.propose.encode())
        h.update(b"||")
        h.update(self.select.encode())
        return h.hexdigest()[:12]


@dataclass
class Candidate:
    move_uci: str | None
    reasoning: str
    fmt_ok: bool
    legal: bool
    raw_response: str


@dataclass
class PipelineMoveResult:
    move_uci: str | None
    candidates: list[Candidate]
    selector_used: bool
    selector_fmt_ok: bool
    selector_raw: str | None
    error: str | None = None

    def fmt_ok(self) -> bool:
        """Overall format compliance: all proposer fmts OK and (if used) selector OK."""
        if not all(c.fmt_ok for c in self.candidates):
            return False
        if self.selector_used and not self.selector_fmt_ok:
            return False
        return True

    def legal(self) -> bool:
        return self.move_uci is not None


def _parse_json(text: str) -> dict | None:
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        t = "\n".join(lines[1:-1]) if len(lines) > 2 else t
        if t.lstrip().startswith("json"):
            t = t.lstrip()[4:].strip()
    try:
        data = json.loads(t)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _parse_candidate(raw: str, board: chess.Board) -> Candidate:
    data = _parse_json(raw)
    if data is not None and "move" in data:
        mv_str = str(data.get("move", "")).strip()
        reasoning = str(data.get("reasoning", "")).strip()
        try:
            mv = chess.Move.from_uci(mv_str)
            legal = mv in board.legal_moves
            return Candidate(move_uci=mv_str if legal else None,
                             reasoning=reasoning, fmt_ok=True, legal=legal, raw_response=raw)
        except Exception:
            return Candidate(None, reasoning, fmt_ok=True, legal=False, raw_response=raw)
    # Fallback scan for any legal UCI token
    for tok in raw.replace(",", " ").replace('"', " ").split():
        tok = tok.strip(".,!?;:()[]{}\"' ").lower()
        try:
            mv = chess.Move.from_uci(tok)
            if mv in board.legal_moves:
                return Candidate(mv.uci(), "", fmt_ok=False, legal=True, raw_response=raw)
        except Exception:
            pass
    return Candidate(None, "", fmt_ok=False, legal=False, raw_response=raw)


class Pipeline:
    def __init__(
        self,
        prompts: PromptSet,
        budget: Budget,
        *,
        n_propose: int = 3,
        client=None,
        model: str = DEFAULT_MODEL,
        temperature: float = 1.0,
        tag: str = "pipeline",
    ):
        self.prompts = prompts
        self.budget = budget
        self.n_propose = n_propose
        self.client = client or make_client()
        self.model = model
        self.temperature = temperature
        self.tag = tag

    # ----- module 1: propose -----

    def _propose_user_msg(self, fen: str, board: chess.Board) -> str:
        legal = [m.uci() for m in board.legal_moves]
        return (
            f"Current Position (FEN): {fen}\n"
            f"Side to move: {'White' if board.turn else 'Black'}\n"
            f"Legal moves ({len(legal)}): {', '.join(legal)}\n\n"
            "Propose ONE candidate move. Respond with JSON only."
        )

    def _propose_one(self, fen: str, board: chess.Board, seed_idx: int) -> Candidate:
        user = self._propose_user_msg(fen, board)
        # Small style hint to encourage diversity across parallel calls even
        # if temperature variance is weak. Hints never override the prompt's
        # "pick the best move" directive — they nudge which kinds of moves
        # to give priority to.
        hints = [
            "",  # no hint
            "Priority hint: if you see a direct tactical motif (fork/pin/skewer/discovery/mate-threat), prefer it.",
            "Priority hint: if tactics are unclear, prefer a quiet move that improves your worst-placed piece.",
            "Priority hint: consider a pawn break or structural move that opens lines for your pieces.",
            "Priority hint: double-check king safety and prophylaxis before choosing.",
        ]
        hint = hints[seed_idx % len(hints)]
        if hint:
            user = user + f"\n\n{hint}"
        try:
            content, _ = chat(
                messages=[{"role": "system", "content": self.prompts.propose},
                          {"role": "user", "content": user}],
                budget=self.budget, tag=f"{self.tag}_propose",
                client=self.client, model=self.model,
                temperature=self.temperature,
            )
        except Exception as e:
            return Candidate(None, f"ERROR: {e}", fmt_ok=False, legal=False, raw_response="")
        return _parse_candidate(content, board)

    # ----- module 2: select -----

    def _select_user_msg(self, fen: str, board: chess.Board, legal_cands: list[Candidate]) -> str:
        lines = [
            f"Current Position (FEN): {fen}",
            f"Side to move: {'White' if board.turn else 'Black'}",
            "",
            f"Candidate moves (from {len(legal_cands)} proposers):",
        ]
        for i, c in enumerate(legal_cands, 1):
            lines.append(f"  {i}. {c.move_uci} — {c.reasoning or '(no reasoning)'}")
        lines.append("")
        lines.append("Pick the strongest candidate. Respond with JSON only.")
        return "\n".join(lines)

    def _select(self, fen: str, board: chess.Board,
                legal_cands: list[Candidate]) -> tuple[str | None, bool, str]:
        user = self._select_user_msg(fen, board, legal_cands)
        try:
            content, _ = chat(
                messages=[{"role": "system", "content": self.prompts.select},
                          {"role": "user", "content": user}],
                budget=self.budget, tag=f"{self.tag}_select",
                client=self.client, model=self.model,
                temperature=0.0,
            )
        except Exception as e:
            return None, False, f"ERROR: {e}"
        data = _parse_json(content)
        if data is not None and "move" in data:
            mv_str = str(data.get("move", "")).strip()
            legal_ucis = {c.move_uci for c in legal_cands}
            try:
                mv = chess.Move.from_uci(mv_str)
                if mv.uci() in legal_ucis and mv in board.legal_moves:
                    return mv.uci(), True, content
            except Exception:
                pass
        # Fallback: scan for any candidate UCI in the text
        for c in legal_cands:
            if c.move_uci and c.move_uci in content:
                return c.move_uci, False, content
        # Final fallback: majority vote from proposers, then first legal
        from collections import Counter
        votes = Counter(c.move_uci for c in legal_cands if c.move_uci)
        if votes:
            return votes.most_common(1)[0][0], False, content
        return None, False, content

    # ----- top-level -----

    def get_move(self, fen: str) -> PipelineMoveResult:
        board = chess.Board(fen)

        with ThreadPoolExecutor(max_workers=self.n_propose) as ex:
            cands = list(ex.map(
                lambda i: self._propose_one(fen, board, i),
                range(self.n_propose),
            ))

        legal_cands = [c for c in cands if c.legal]
        unique_ucis = {c.move_uci for c in legal_cands}

        if not legal_cands:
            # No legal proposals at all
            return PipelineMoveResult(
                move_uci=None, candidates=cands, selector_used=False,
                selector_fmt_ok=False, selector_raw=None,
                error="no_legal_proposals",
            )

        if len(unique_ucis) == 1:
            # Unanimous — skip select to save a call
            return PipelineMoveResult(
                move_uci=legal_cands[0].move_uci, candidates=cands,
                selector_used=False, selector_fmt_ok=True, selector_raw=None,
            )

        chosen, sel_fmt, raw = self._select(fen, board, legal_cands)
        return PipelineMoveResult(
            move_uci=chosen, candidates=cands,
            selector_used=True, selector_fmt_ok=sel_fmt, selector_raw=raw,
        )
