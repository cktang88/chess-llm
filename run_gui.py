#!/usr/bin/env python3
"""
Run the chess GUI with AI integration.

Usage:
    python run_gui.py
"""
import os
import sys

# Change to the pygame directory so assets can be found
os.chdir('python-chess-ai-yt')

# Add src to path
sys.path.insert(0, 'src')

# Import and run the game
from main import Main

if __name__ == "__main__":
    print("Starting Chess GUI with AI integration...")
    print("\nControls:")
    print("  - Drag and drop pieces to move")
    print("  - Press 'T' to change theme")
    print("  - Press 'R' to reset game")
    print("  - Press 'A' to toggle AI on/off")
    print("  - Close window to quit")
    print()

    main = Main()
    main.mainloop()
