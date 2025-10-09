#!/usr/bin/env python3
"""Test script to verify the integration."""
import sys
sys.path.append('python-chess-ai-yt/src')

from board import Board
import chess

def test_fen_conversion():
    """Test FEN conversion from pygame board."""
    print("Testing FEN conversion...")

    # Create a new board
    board = Board()

    # Get FEN
    fen = board.to_fen()
    print(f"Generated FEN: {fen}")

    # Expected starting position
    expected = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    print(f"Expected FEN:  {expected}")

    # Validate with python-chess
    try:
        chess_board = chess.Board(fen)
        print("✓ FEN is valid!")
        print(f"Python-chess board:\n{chess_board}")
        return True
    except Exception as e:
        print(f"❌ FEN validation failed: {e}")
        return False

def test_uci_conversion():
    """Test UCI to move conversion."""
    print("\nTesting UCI to Move conversion...")

    board = Board()

    # Test e2e4
    uci = "e2e4"
    move = board.uci_to_move(uci)
    if move:
        print(f"✓ UCI '{uci}' converted successfully")
        print(f"  From: row={move.initial.row}, col={move.initial.col}")
        print(f"  To: row={move.final.row}, col={move.final.col}")
        return True
    else:
        print(f"❌ Failed to convert UCI '{uci}'")
        return False

if __name__ == "__main__":
    print("="*60)
    print("Integration Test Suite")
    print("="*60)

    success = True
    success = test_fen_conversion() and success
    success = test_uci_conversion() and success

    print("\n" + "="*60)
    if success:
        print("✓ All tests passed!")
    else:
        print("❌ Some tests failed")
    print("="*60)
