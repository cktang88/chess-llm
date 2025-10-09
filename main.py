#!/usr/bin/env python3
"""
Chess Game: Human vs LLM
Play chess against an AI powered by GPT-5 with advanced tactical knowledge.
"""

import os
import chess
import chess.svg
from openai import OpenAI
from dotenv import load_dotenv
from chess_prompt import CHESS_TACTICS_PROMPT

# Load environment variables from .env file
load_dotenv()


class ChessGame:
    def __init__(self, api_key: str = None):
        """Initialize chess game with LLM opponent."""
        self.board = chess.Board()
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=self.api_key)
        self.move_history = []

    def display_board(self):
        """Display the current board state."""
        print("\n" + "=" * 50)
        print(self.board)
        print("=" * 50)
        print(f"FEN: {self.board.fen()}")
        print(f"Turn: {'White' if self.board.turn == chess.WHITE else 'Black'}")
        print("=" * 50 + "\n")

    def get_legal_moves_str(self) -> str:
        """Get formatted list of legal moves."""
        legal_moves = [move.uci() for move in self.board.legal_moves]
        return ", ".join(legal_moves[:20]) + ("..." if len(legal_moves) > 20 else "")

    def parse_player_move(self, move_str: str) -> chess.Move:
        """Parse player move from various formats."""
        move_str = move_str.strip()

        # Try UCI format first (e2e4)
        try:
            move = chess.Move.from_uci(move_str)
            if move in self.board.legal_moves:
                return move
        except:
            pass

        # Try SAN format (Nf3, e4, etc.)
        try:
            move = self.board.parse_san(move_str)
            if move in self.board.legal_moves:
                return move
        except:
            pass

        return None

    def get_llm_move(self) -> chess.Move:
        """Get move from LLM using extended thinking."""
        # Build context with recent moves
        move_history_str = (
            " ".join(self.move_history[-10:]) if self.move_history else "Game start"
        )

        user_message = f"""Current Position (FEN): {self.board.fen()}

Recent moves: {move_history_str}

Legal moves available: {self.get_legal_moves_str()}

Analyze the position and provide your next move. Respond with just the move in UCI format (e.g., 'e2e4') followed by a brief explanation."""

        print("🤔 LLM is thinking...")

        response = self.client.chat.completions.create(
            model="gpt-5-thinking",
            reasoning_effort="high",
            messages=[
                {
                    "role": "user",
                    "content": CHESS_TACTICS_PROMPT + "\n\n" + user_message,
                }
            ],
        )

        # Extract move and reasoning from response
        response_text = response.choices[0].message.content

        # Show reasoning summary if available
        if (
            hasattr(response.choices[0].message, "reasoning_content")
            and response.choices[0].message.reasoning_content
        ):
            reasoning = response.choices[0].message.reasoning_content
            print(f"\n💭 LLM Reasoning (first 500 chars):\n{reasoning[:500]}...\n")

        print(f"🎯 LLM Response: {response_text}\n")

        # Try to extract move from response
        lines = response_text.strip().split("\n")
        for line in lines:
            words = line.split()
            for word in words:
                # Clean the word
                word = word.strip(".,!?;:()[]{}\"' ")
                move = self.parse_player_move(word)
                if move:
                    return move

        # If no move found, ask LLM to clarify
        raise ValueError(f"Could not parse move from LLM response: {response_text}")

    def play_game(self, human_color: str = "white"):
        """Main game loop."""
        human_is_white = human_color.lower() in ["white", "w"]

        print("♟️  Chess Game: Human vs LLM ♟️")
        print(f"You are playing as: {'White' if human_is_white else 'Black'}")
        print("Enter moves in UCI format (e.g., 'e2e4') or SAN format (e.g., 'Nf3')")
        print("Type 'quit' to exit, 'fen' to show FEN notation\n")

        while not self.board.is_game_over():
            self.display_board()

            is_human_turn = (self.board.turn == chess.WHITE) == human_is_white

            if is_human_turn:
                # Human move
                while True:
                    move_input = input("Your move: ").strip()

                    if move_input.lower() == "quit":
                        print("Game ended by player.")
                        return

                    if move_input.lower() == "fen":
                        print(f"FEN: {self.board.fen()}")
                        continue

                    move = self.parse_player_move(move_input)

                    if move:
                        self.board.push(move)
                        self.move_history.append(move.uci())
                        print(f"✓ Played: {move.uci()}\n")
                        break
                    else:
                        print(f"❌ Invalid move. Try again.")
                        print(f"Legal moves: {self.get_legal_moves_str()}\n")
            else:
                # LLM move
                try:
                    move = self.get_llm_move()
                    self.board.push(move)
                    self.move_history.append(move.uci())
                    print(f"✓ LLM played: {move.uci()}\n")
                except Exception as e:
                    print(f"❌ Error getting LLM move: {e}")
                    print("Game ended due to LLM error.")
                    return

        # Game over
        self.display_board()
        print("\n🏁 GAME OVER 🏁")

        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            print(f"Checkmate! {winner} wins!")
        elif self.board.is_stalemate():
            print("Stalemate! Game is a draw.")
        elif self.board.is_insufficient_material():
            print("Draw by insufficient material.")
        elif self.board.is_fifty_moves():
            print("Draw by fifty-move rule.")
        elif self.board.is_repetition():
            print("Draw by threefold repetition.")
        else:
            print("Game ended.")

        print(f"\nFinal FEN: {self.board.fen()}")
        print(f"Move history: {' '.join(self.move_history)}")


def main():
    """Main entry point."""
    try:
        game = ChessGame()

        # Ask which color the human wants to play
        while True:
            color = input("Choose your color (white/black) [white]: ").strip().lower()
            if not color:
                color = "white"
            if color in ["white", "w", "black", "b"]:
                break
            print("Invalid choice. Enter 'white' or 'black'.")

        game.play_game(human_color=color)

    except KeyboardInterrupt:
        print("\n\nGame interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
