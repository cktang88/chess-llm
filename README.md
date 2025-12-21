# Chess LLM - Play Chess Against AI

A chess game with a beautiful pygame GUI where you can play against an AI powered by GPT-5.2 with advanced tactical knowledge.

## Features

- **Beautiful GUI**: Pygame-based graphical interface with drag-and-drop piece movement
- **LLM-Powered AI**: Play against GPT-5.2 with deep chess tactics knowledge
- **FEN Integration**: Seamless conversion between GUI board state and standard chess notation
- **Move History Tracking**: AI learns from the game progression
- **Keyboard Controls**:
  - `T` - Change theme
  - `R` - Reset game
  - `A` - Toggle AI on/off during play

## Installation

```bash
# Install dependencies using uv
uv sync

# Or using pip
pip install -r requirements.txt
```

## Setup

Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=your-api-key-here
```

## Usage

### Play with GUI (Recommended)

```bash
# Run the GUI version
uv run python run_gui.py

# Or directly with Python
python run_gui.py
```

When the GUI starts, you'll be prompted:
1. Whether to play against AI (y/n)
2. Your color choice (white/black)

Then use your mouse to drag and drop pieces on the board!

### Play in Terminal (Original)

```bash
# Run the CLI version
uv run python main.py

# Or directly with Python
python main.py
```

## How It Works

The integration works by:

1. **Board State Serialization**: The pygame board is converted to FEN (Forsyth-Edwards Notation) format
2. **AI Move Generation**: The FEN is sent to the LLM along with tactical chess knowledge
3. **Move Translation**: The LLM's UCI move (e.g., "e2e4") is converted back to pygame Move objects
4. **Turn Management**: The game automatically alternates between human and AI moves

## Project Structure

```
chess-llm/
├── run_gui.py                 # GUI launcher script
├── main.py                    # CLI version
├── chess_prompt.py            # LLM chess tactics prompt
├── test_integration.py        # Integration tests
└── python-chess-ai-yt/        # Pygame chess GUI
    └── src/
        ├── main.py            # GUI main loop
        ├── game.py            # Game logic with AI integration
        ├── board.py           # Board with FEN conversion
        ├── ai_controller.py   # LLM AI controller
        └── ...                # Other GUI components
```

## Chess Tactics

The AI is instructed with advanced chess knowledge including:
- Tactical patterns (forks, pins, skewers, discovered attacks)
- Strategic principles (opening, middlegame, endgame)
- Position evaluation criteria
- Deep calculation with extended thinking

See `chess_prompt.py` for the full tactical knowledge base.

## Testing

```bash
# Run integration tests
uv run python test_integration.py
```

This verifies:
- FEN conversion from pygame board to standard notation
- UCI move translation from AI to pygame format

## Requirements

- Python 3.10+
- pygame
- python-chess
- openai
- python-dotenv

## Credits

- Pygame chess GUI based on [python-chess-ai-yt](https://github.com/...)
- LLM integration and tactical knowledge custom implementation
