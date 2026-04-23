"""Generate the position dataset (~50 FENs across phases) used for grading.

We assemble:
- Famous opening positions reached after a few moves of well-known lines
- Middlegame snapshots from short Stockfish self-play with varied openings
- Tactical/endgame positions hand-curated from canonical FENs

Side-to-move is mixed (white and black) so the LLM is exercised on both colors.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import chess
import chess.engine

from .grader import STOCKFISH_PATH

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "positions.jsonl"


# Opening lines — UCI move sequences from the start position.
OPENING_LINES = [
    # Italian
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3"],
    # Ruy Lopez
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"],
    # Sicilian Najdorf-ish
    ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"],
    # French
    ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6", "c1g5"],
    # Caro-Kann
    ["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4", "c3e4"],
    # Queen's Gambit Declined
    ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5"],
    # King's Indian
    ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6"],
    # English
    ["c2c4", "e7e5", "b1c3", "g8f6", "g1f3", "b8c6"],
    # Scandinavian
    ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5"],
    # Slav
    ["d2d4", "d7d5", "c2c4", "c7c6", "g1f3", "g8f6", "b1c3", "d5c4"],
    # London
    ["d2d4", "d7d5", "c1f4", "g8f6", "e2e3", "c7c5", "c2c3"],
    # Pirc
    ["e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6"],
]


# Hand-curated standalone FENs (tactics + endgames). Each must be valid.
CURATED_FENS = [
    # --- Tactics ---
    # Knight fork available
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 5",
    # Pin tactic
    "r2qkbnr/ppp2ppp/2np4/4p3/2B1P1b1/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 5",
    # Open Sicilian middlegame, tactical
    "r1bqkb1r/pp2pppp/2np1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 2 6",
    # Greek Gift setup
    "r1bqk2r/pppp1ppp/2n2n2/4p3/1bB1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 5",
    # King hunt
    "r3k2r/ppp2ppp/2n1bn2/2bqp3/3P4/2N1BN2/PPP1QPPP/R3KB1R w KQkq - 0 9",

    # --- Endgames ---
    # K+Q vs K (mate technique)
    "4k3/8/4K3/8/8/8/4Q3/8 w - - 0 1",
    # K+R vs K
    "8/8/4k3/8/8/4K3/4R3/8 w - - 0 1",
    # K+P vs K, white to promote
    "8/8/8/3k4/8/3P4/3K4/8 w - - 0 1",
    # Rook endgame, queening race
    "8/1P5k/8/8/8/2K5/r7/4R3 w - - 0 1",
    # R+P vs R, basic
    "8/8/4k3/8/8/4K3/4P3/4r3 w - - 0 1",
    # Opposite-color bishop ending
    "8/4kp2/4p3/8/4P3/4KP2/8/8 w - - 0 1",
    # Knight endgame
    "8/8/3k4/8/3P4/3K4/8/3N4 w - - 0 1",

    # --- Mid-game positional ---
    # IQP
    "r1bqr1k1/pp3ppp/2n2n2/3p4/3P4/2N1PN2/PP3PPP/R1BQR1K1 w - - 0 11",
    # Maroczy bind
    "r1bq1rk1/pp2ppbp/2n3p1/3p4/2PNP3/2N1B3/PP2BPPP/R2QK2R w KQ - 0 9",
    # Hanging pawns
    "r1bqr1k1/1p3pbp/p1n1pnp1/8/2pP4/2N1PNB1/PP2BPPP/R2QR1K1 w - - 0 11",
    # Stonewall Dutch
    "r1bq1rk1/pp1n1pbp/2p2np1/3p4/3P1B2/2NQPN2/PP3PPP/2KR3R w - - 0 11",

    # --- Black-to-move positions ---
    # Sicilian dragon mid-game
    "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1B3/PPP2PPP/R2QKB1R b KQ - 0 7",
    # KID with attack on the kingside
    "r1bq1rk1/pp1nppbp/3p1np1/2pP4/2P1P3/2N2N2/PP2BPPP/R1BQ1RK1 b - - 0 8",
    # French winawer black to play
    "r1bqk2r/pp1nbppp/4pn2/3pN3/3P4/2N1P3/PP3PPP/R1BQKB1R b KQkq - 0 7",
    # Endgame black to defend
    "8/5pk1/6p1/8/8/6PK/5P2/8 b - - 0 1",
]


def _make_engine() -> chess.engine.SimpleEngine:
    return chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)


def _generate_selfplay_positions(n_games: int = 5, snapshots_per_game: int = 3,
                                 depth: int = 8, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    out: list[dict] = []
    eng = _make_engine()
    try:
        for g in range(n_games):
            board = chess.Board()
            # Inject randomness: pick one of top moves at low depth occasionally
            for ply in range(rng.randint(20, 36)):
                if board.is_game_over():
                    break
                infos = eng.analyse(board, chess.engine.Limit(depth=depth), multipv=3)
                choices = [info["pv"][0] for info in infos if info.get("pv")]
                if not choices:
                    break
                # 70% best, 30% random of top-3
                pick = choices[0] if rng.random() < 0.7 else rng.choice(choices)
                board.push(pick)
            # Sample snapshots from the played-out game
            move_list = list(board.move_stack)
            replay = chess.Board()
            ply_indices = sorted(rng.sample(range(8, max(9, len(move_list) - 2)),
                                            k=min(snapshots_per_game, max(1, len(move_list) - 10))))
            for i, mv in enumerate(move_list):
                if i in ply_indices and not replay.is_game_over():
                    out.append({"fen": replay.fen(), "phase": "selfplay", "tag": f"g{g}_p{i}"})
                replay.push(mv)
    finally:
        eng.quit()
    return out


def _opening_positions() -> list[dict]:
    out: list[dict] = []
    for i, line in enumerate(OPENING_LINES):
        b = chess.Board()
        for uci in line:
            b.push(chess.Move.from_uci(uci))
        out.append({"fen": b.fen(), "phase": "opening", "tag": f"open_{i}"})
    return out


def _curated_positions() -> list[dict]:
    out = []
    for i, fen in enumerate(CURATED_FENS):
        # Validate
        b = chess.Board(fen)
        if not b.is_valid():
            raise ValueError(f"invalid curated FEN at index {i}: {fen}")
        out.append({"fen": fen, "phase": "curated", "tag": f"cur_{i}"})
    return out


def build_dataset() -> list[dict]:
    positions: list[dict] = []
    positions += _opening_positions()
    positions += _curated_positions()
    positions += _generate_selfplay_positions(n_games=4, snapshots_per_game=3, seed=42)
    # Deduplicate by FEN
    seen = set()
    unique = []
    for p in positions:
        if p["fen"] in seen:
            continue
        seen.add(p["fen"])
        unique.append(p)
    return unique


def write_dataset(path: Path = OUT_PATH) -> int:
    positions = build_dataset()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for p in positions:
            f.write(json.dumps(p) + "\n")
    return len(positions)


if __name__ == "__main__":
    n = write_dataset()
    print(f"wrote {n} positions to {OUT_PATH}")
