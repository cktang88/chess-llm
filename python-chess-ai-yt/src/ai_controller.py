"""AI Controller for Chess Game using LLM."""

import os
import json
import chess
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the chess tactics prompt
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from chess_prompt import CHESS_TACTICS_PROMPT


class AIController:
    """Handles LLM-based AI moves for the chess game."""

    def __init__(self, api_key=None):
        """Initialize AI controller with OpenAI client."""
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=self.api_key)
        self.move_history = []

    def get_ai_move(self, fen_position):
        """
        Get AI move using LLM for the given FEN position.

        Args:
            fen_position: FEN string representing current board state

        Returns:
            UCI move string (e.g., 'e2e4') or None if error
        """
        # Create python-chess board from FEN
        try:
            board = chess.Board(fen_position)
        except Exception as e:
            print(f"Error parsing FEN: {e}")
            return None

        # Get legal moves
        legal_moves = [move.uci() for move in board.legal_moves]
        legal_moves_str = ", ".join(legal_moves[:20]) + (
            "..." if len(legal_moves) > 20 else ""
        )

        # Build move history string
        move_history_str = (
            " ".join(self.move_history[-10:]) if self.move_history else "Game start"
        )

        # Build prompt for LLM
        user_message = f"""Current Position (FEN): {fen_position}

Recent moves: {move_history_str}

Legal moves available: {legal_moves_str}

Analyze the position and provide your next move. Respond with JSON in this exact format:
{{
  "move": "e2e4",
  "reasoning": "Brief explanation of why this move is good"
}}"""

        print("🤔 AI is thinking...")

        try:
            response = self.client.chat.completions.create(
                model="gpt-5",
                reasoning_effort="low",
                messages=[
                    {
                        "role": "user",
                        "content": CHESS_TACTICS_PROMPT + "\n\n" + user_message,
                    }
                ],
            )

            # Extract response
            response_text = response.choices[0].message.content

            # Show reasoning if available
            if (
                hasattr(response.choices[0].message, "reasoning_content")
                and response.choices[0].message.reasoning_content
            ):
                reasoning = response.choices[0].message.reasoning_content
                print(f"\n💭 AI Reasoning (first 300 chars):\n{reasoning[:300]}...\n")

            # Parse move from response
            move_uci, reasoning_text = self._extract_move_and_reasoning(
                response_text, board
            )

            if reasoning_text:
                print(f"🎯 AI Move: {move_uci}")
                print(f"💡 Reasoning: {reasoning_text}\n")

            if move_uci:
                self.move_history.append(move_uci)
                return move_uci
            else:
                print(
                    f"❌ Could not parse valid move from AI response: {response_text}"
                )
                return None

        except Exception as e:
            print(f"❌ Error getting AI move: {e}")
            return None

    def _extract_move_and_reasoning(self, response_text, board):
        """
        Extract UCI move and reasoning from LLM response.

        Returns:
            tuple: (move_uci, reasoning_text) or (None, None) if parsing fails
        """
        # Try to parse as JSON first
        try:
            # Clean up response - sometimes LLMs wrap JSON in markdown code blocks
            cleaned_text = response_text.strip()
            if cleaned_text.startswith("```"):
                # Remove code block markers
                lines = cleaned_text.split("\n")
                cleaned_text = (
                    "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned_text
                )
                if cleaned_text.startswith("json"):
                    cleaned_text = cleaned_text[4:].strip()

            data = json.loads(cleaned_text)
            move_str = data.get("move", "")
            reasoning = data.get("reasoning", "")

            # Validate move
            try:
                move = chess.Move.from_uci(move_str)
                if move in board.legal_moves:
                    return move.uci(), reasoning
            except:
                pass
        except json.JSONDecodeError:
            pass

        # Fallback: try to extract move from plain text
        lines = response_text.strip().split("\n")
        for line in lines:
            words = line.split()
            for word in words:
                # Clean the word
                word = word.strip(".,!?;:()[]{}\"' ")
                # Try to parse as UCI
                try:
                    move = chess.Move.from_uci(word)
                    if move in board.legal_moves:
                        # Try to get reasoning from the rest of the text
                        remaining_text = response_text[
                            response_text.find(word) + len(word) :
                        ].strip()
                        return move.uci(), remaining_text if remaining_text else None
                except:
                    pass

        return None, None

    def add_opponent_move(self, uci_move):
        """Add opponent's move to history."""
        self.move_history.append(uci_move)

    def reset(self):
        """Reset move history."""
        self.move_history = []
