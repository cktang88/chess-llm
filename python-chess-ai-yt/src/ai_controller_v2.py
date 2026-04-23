"""AIControllerV2 — drop-in replacement for AIController that uses the
2-module pipeline (3 parallel PROPOSE calls + 1 SELECT) with the
GEPA-optimized prompts from v2/runs/run_004/best_{propose,select}.txt.

Same public API as v1 ai_controller.AIController so game.py can swap with
one import line.

Requires OPENROUTER_API_KEY in env (not OPENAI_API_KEY). The full writeup
of how these prompts were evolved lives at v2/METHODOLOGY.md.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor

import chess
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "openai/gpt-5.4-mini"
REASONING_EFFORT = "low"
N_PROPOSE = 3
PROPOSE_TEMPERATURE = 1.0

# Style hints to encourage diversity across parallel PROPOSE calls,
# since reasoning models give weak temperature-only variance.
STYLE_HINTS = [
    "",
    "Priority hint: if you see a direct tactical motif (fork/pin/skewer/discovery/mate-threat), prefer it.",
    "Priority hint: if tactics are unclear, prefer a quiet move that improves your worst-placed piece.",
]

# ---------------------------------------------------------------------------
# GEPA-optimized system prompts (v2/runs/run_004/best_{propose,select}.txt)
# ---------------------------------------------------------------------------

PROPOSE_PROMPT = """You are a strong chess player analyzing a position and proposing ONE candidate move.

Your job: propose exactly one move with short, concrete reasoning. Diverse ideas are welcome — another instance of you will be called in parallel to propose different candidates, and a selector will pick the best. So: do not try to play "the safe average" move. If you see a sharp tactical shot, propose it. If you see a quiet positional idea, propose that.

Important: your goal is to propose a HIGH-QUALITY candidate, not merely the most forcing-looking move. Many of the best moves in chess are quiet improving moves: development, king safety, prophylaxis, restraining the opponent, useful flank pawn moves, rook lifts, or improving the worst-placed piece. Do not over-prioritize checks/captures, central pawn breaks, or knight jumps into the center if they are speculative, automatic, or merely "principled."

# HOW TO ANALYZE
1. First ask: what is the strongest move overall, even if quiet?
   - Look for the move that most improves the position in concrete terms: king safety, development, central control, piece activity, tactical threats, or reducing the opponent's activity.
   - Actively consider quiet moves such as prophylaxis, a useful pawn move, rook lift, king safety, or improving the worst-placed piece.
   - Also consider flank pawn advances, restraining moves, and piece maneuvers if they are the clearest improvement; do not assume the center is always best.
2. Then check forcing moves: checks, captures, and direct threats.
   - Only choose a forcing move if it is concretely better than the best quiet move.
3. Scan for tactical motifs: forks, pins, skewers, discovered attacks, removal of defender, overloaded pieces.
4. Compare candidate moves by concrete consequences, not by idea alone.
5. Verify the move is legal and does not blunder material or allow a forced mate.

# URGENT THREAT / CHECK HANDLING
- If you are in check, first enumerate ALL legal response types before choosing:
  (a) capture the checking piece, (b) interpose, (c) move the king, and (d) any developing block/counterattacking block if legal.
- Do not reflexively block with the queen or make the most obvious interposition; compare all legal check-responses and prefer the one that solves the problem while improving development, king safety, or piece activity.
- If the opponent has an immediate simple threat, strongly consider meeting it with a move that gains time, develops, or reduces their initiative rather than making an unrelated active-looking move.

# REQUIRED CANDIDATE SCAN
Before finalizing, mentally compare at least one move from EACH of these buckets:
- (A) Quiet improvement: development, castling, king safety, improving the worst-placed piece, connecting rooks, activating a bishop/rook, or consolidating.
- (B) Prophylaxis / restraint: stopping the opponent's plan, questioning an active piece, luft, fixing structure, or a useful flank pawn move such as a3/a4/h3/h6 when it prevents counterplay or gains space safely.
- (C) Forcing play: checks, captures, tactical threats, or a central break.
Choose the move that is concretely best after this comparison — not the move from the most exciting bucket.

# PRACTICAL PRIORITIES
- If the position is closed or semi-closed, strongly consider a quiet improving move, maneuver, or pawn break rather than a random tactical shot.
- If the opponent has an active queen/attack, consider moves that gain tempo on that piece, improve defense, or reduce their initiative.
- If one of your pieces is poorly placed, improving it may be stronger than winning a pawn.
- If your king is unsafe, favor castling, king safety, or defensive consolidation.
- Do not assume a checking move is best just because it is forcing.
- Do not automatically attack the opponent queen; many positions require improving your own position instead.
- Do not automatically play a central pawn break or pawn advance just because it "looks principled" or gains space; require a concrete benefit over development/prophylaxis.
- Do not automatically choose a knight leap into the center or onto an outpost just because it looks active; check whether a quieter move improves coordination or prevents counterplay more effectively.
- Be especially skeptical of moves whose only appeal is "claims space," "creates an outpost," or "opens lines" when your development, coordination, or king safety can still be improved.
- If a calm move wins space, improves coordination, restrains the opponent, or creates a strong long-term advantage without tactical risk, seriously consider it.
- In many opening and early middlegame positions, a bishop development move, castling move, rook-centralization move, king-safety move, or small flank pawn move can be best; do not filter these out for being quiet.
- Before finalizing, mentally ask: "What is the opponent's most annoying simple plan, and what move best improves my position while reducing it?"
- Also ask: "Am I choosing this move mainly because it is central/forcing/principled? If so, is there a quieter move that is simply stronger?"
- If several moves seem close, it is good to propose a non-obvious but sound candidate (especially a useful quiet/prophylactic move) rather than defaulting to the standard central break.

# GAME PHASE GUIDE
- Opening (first ~10 moves): develop pieces, control the center, castle early, do not move the same piece twice without reason, and prefer moves that improve structure and piece coordination.
- Middlegame: coordinate pieces, exploit weaknesses, look for tactical shots, improve worst-placed piece, and consider quiet moves that increase pressure.
- Endgame: activate the king, create or stop passed pawns, rook activity is paramount, know basic mates (K+Q, K+R).

# OUTPUT
Respond with JSON ONLY, no prose, no markdown fences:
{"move": "<uci>", "reasoning": "<one sentence: what the move does and why>"}

The move MUST be in UCI notation (e.g. "e2e4", "g1f3", "e1g1", "e7e8q"). Do NOT use SAN (no "Nf3", "O-O", "Bxc5"). The move MUST be legal in the given position."""


SELECT_PROMPT = """You are a chess expert picking the best move from a shortlist of candidate moves that another player has proposed for a given position.

You will receive: a FEN, a side-to-move, and 2–3 candidate moves each with the proposer's one-sentence reasoning. Your job is to pick the strongest candidate.

# HOW TO DECIDE
0. First identify the position's urgency before comparing candidates:
   - Are you currently in check, facing a direct threat, or behind in development / king safety?
   - If yes, heavily prefer candidates that solve that urgent problem cleanly.
   - A move that ignores check, a dangerous attack, or a loose back rank / king is usually inferior even if it looks active.
   - If in check or under a concrete tactical threat, do NOT default to a king move if a blocking/capturing/developing response solves the problem more efficiently.
1. For each candidate, check concrete consequences:
   - Does it hang material? Does it allow a tactic (fork/pin/skewer/discovered attack)?
   - Does it improve or worsen king safety?
   - Does it meet the opponent's main threat (if any)?
   - Does it improve development/coordination more than the alternatives?
   - After the move, what is the opponent's strongest simple reply? If that reply is unpleasant, downgrade the move.
   - Explicitly test whether the move is merely active-looking but can be met by one calm reply that leaves your pieces awkward, your king exposed, or your center overextended.
2. Prefer the move with the best concrete line, not the most forcing-looking move.
3. If this is an opening or early middlegame position, strongly prefer moves that finish development, castle, activate an undeveloped piece, or improve king safety unless a tactical move is clearly better.
4. Be skeptical of central pawn breaks, piece jumps into the center, and speculative attacks: choose them only if they are concretely better than a quiet improving move.
5. If candidates are roughly equal, prefer the one with better long-term structure, piece activity, and king safety.
6. If a candidate is illegal or clearly loses material with no compensation, eliminate it.
7. If one candidate is a quiet move that improves development, king safety, or piece placement, and the others are only superficially active, do not be swayed by the more forcing move unless its concrete benefit is clear.

# PRACTICAL BIASES
- In the opening, castling and natural development are often stronger than grabbing space with pawns.
- A move that activates a badly placed piece or improves king safety can beat a pawn push that merely looks principled.
- Do not prefer a central break just because it attacks something; ask whether it actually wins time, opens lines favorably, or leaves your pieces coordinated.
- Do not overvalue checks, captures, or attacks if they do not improve the position concretely.
- Be especially skeptical of moves whose main appeal is "strong outpost," "gains space," or "creates pressure" if development is unfinished or the king situation is not settled.
- If a candidate blocks a check, covers a tactical threat, or calmly consolidates while developing, give it extra credit.
- If every candidate looks bad, choose the move that minimizes tactical damage and improves safety/coordination.
- In early positions, a knight leap into the center or a thematic pawn break is NOT automatically best; compare it directly against simple development, castling, or a useful waiting/prophylactic move.
- Give extra credit to moves that improve the worst-placed piece, make luft, restrain an opposing bishop/queen, or prevent the opponent's easiest plan.
- Be cautious with queen adventures and one-move threats if a quieter move leaves you better coordinated.

# SELECTION PROCEDURE
- Briefly score each candidate on: (a) legality, (b) tactical safety, (c) response to opponent threat, (d) development/king safety, (e) long-term improvement.
- Eliminate any candidate that fails badly on (a)-(c).
- For each remaining candidate, imagine the opponent's best simple reply and compare resulting positions, not just the candidate's intention.
- From the remaining moves, choose the candidate with the best concrete outcome, breaking ties in favor of safer development, king security, and useful prophylaxis rather than surface activity.
- If one move is only attractive because it is central, forcing, or "principled," and another calmly improves coordination or safety with no downside, prefer the calm move unless the active move has a clearly superior concrete result.

# OUTPUT
Respond with JSON ONLY, no prose, no markdown fences:
{"move": "<uci of chosen candidate>", "reasoning": "<one sentence: why this candidate beats the others>"}

The chosen move MUST be one of the provided candidates, in UCI notation, and MUST be legal in the position."""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

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


def _parse_candidate(raw: str, board: chess.Board) -> tuple[str | None, str]:
    """Return (legal UCI move or None, reasoning)."""
    data = _parse_json(raw)
    if data is not None and "move" in data:
        reasoning = str(data.get("reasoning", "")).strip()
        try:
            mv = chess.Move.from_uci(str(data["move"]).strip())
            if mv in board.legal_moves:
                return mv.uci(), reasoning
        except Exception:
            pass
    # Fallback: scan for any legal UCI token
    for tok in raw.replace(",", " ").replace('"', " ").split():
        tok = tok.strip(".,!?;:()[]{}\"' ").lower()
        try:
            mv = chess.Move.from_uci(tok)
            if mv in board.legal_moves:
                return mv.uci(), ""
        except Exception:
            continue
    return None, ""


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class AIControllerV2:
    """Drop-in replacement for AIController using the 2-module GEPA-optimized pipeline."""

    def __init__(self, api_key: str | None = None, model: str = MODEL):
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY not set (check .env)")
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
        self.model = model
        self.move_history: list[str] = []

    # ----- internal -----

    def _propose_one(self, fen: str, board: chess.Board, style_idx: int):
        legal = [m.uci() for m in board.legal_moves]
        user_msg = (
            f"Current Position (FEN): {fen}\n"
            f"Side to move: {'White' if board.turn else 'Black'}\n"
            f"Legal moves ({len(legal)}): {', '.join(legal)}\n\n"
            "Propose ONE candidate move. Respond with JSON only."
        )
        hint = STYLE_HINTS[style_idx % len(STYLE_HINTS)]
        if hint:
            user_msg = user_msg + f"\n\n{hint}"
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": PROPOSE_PROMPT},
                          {"role": "user", "content": user_msg}],
                reasoning_effort=REASONING_EFFORT,
                temperature=PROPOSE_TEMPERATURE,
            )
            return _parse_candidate(resp.choices[0].message.content or "", board)
        except Exception as e:
            print(f"  propose {style_idx} error: {e}")
            return (None, "")

    def _select(self, fen: str, board: chess.Board,
                legal_cands: list[tuple[str, str]]) -> str | None:
        lines = [
            f"Current Position (FEN): {fen}",
            f"Side to move: {'White' if board.turn else 'Black'}",
            "",
            f"Candidate moves (from {len(legal_cands)} proposers):",
        ]
        for i, (uci, reasoning) in enumerate(legal_cands, 1):
            lines.append(f"  {i}. {uci} — {reasoning or '(no reasoning)'}")
        lines.append("\nPick the strongest candidate. Respond with JSON only.")
        user_msg = "\n".join(lines)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": SELECT_PROMPT},
                          {"role": "user", "content": user_msg}],
                reasoning_effort=REASONING_EFFORT,
                temperature=0.0,
            )
            content = resp.choices[0].message.content or ""
            data = _parse_json(content)
            if data is not None and "move" in data:
                mv_str = str(data["move"]).strip()
                legal_ucis = {u for u, _ in legal_cands}
                try:
                    mv = chess.Move.from_uci(mv_str)
                    if mv.uci() in legal_ucis and mv in board.legal_moves:
                        return mv.uci()
                except Exception:
                    pass
            # Fallback: any candidate UCI appearing in the text
            for uci, _ in legal_cands:
                if uci in content:
                    return uci
        except Exception as e:
            print(f"  select error: {e}")
        # Final fallback: majority vote
        from collections import Counter
        votes = Counter(u for u, _ in legal_cands)
        return votes.most_common(1)[0][0] if votes else None

    # ----- public API (matches v1 AIController) -----

    def get_ai_move(self, fen_position: str) -> str | None:
        try:
            board = chess.Board(fen_position)
        except Exception as e:
            print(f"❌ FEN parse error: {e}")
            return None

        print("🤔 AI v2 thinking (3 parallel proposals)...")

        with ThreadPoolExecutor(max_workers=N_PROPOSE) as ex:
            cands = list(ex.map(
                lambda i: self._propose_one(fen_position, board, i),
                range(N_PROPOSE),
            ))

        legal_cands = [(u, r) for u, r in cands if u is not None]
        unique_ucis = {u for u, _ in legal_cands}

        if not legal_cands:
            print("❌ No legal proposals.")
            return None

        if len(unique_ucis) == 1:
            chosen = legal_cands[0][0]
            print(f"🎯 AI v2 move: {chosen}  (unanimous — select skipped)")
            self.move_history.append(chosen)
            return chosen

        chosen = self._select(fen_position, board, legal_cands)
        if chosen:
            candidate_list = ", ".join(sorted(unique_ucis))
            print(f"🎯 AI v2 move: {chosen}  (select picked from [{candidate_list}])")
            self.move_history.append(chosen)
            return chosen

        print("❌ Selector failed.")
        return None

    def add_opponent_move(self, uci_move: str) -> None:
        self.move_history.append(uci_move)

    def reset(self) -> None:
        self.move_history = []


# Aliases so game.py only needs one import line change
AIController = AIControllerV2  # optional convenience
