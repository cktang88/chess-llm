"""Seed system prompt for the optimizer. Starts from the v1 baseline.

The optimizer mutates this string. Sections are separated by blank lines so
the reflector can target a specific section ("Maestro trick").
"""

# Re-export the v1 prompt verbatim so we measure improvement against it.
import os
import sys

_V1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _V1_DIR not in sys.path:
    sys.path.insert(0, _V1_DIR)

from chess_prompt import CHESS_TACTICS_PROMPT as SEED_PROMPT  # noqa: E402

__all__ = ["SEED_PROMPT"]
