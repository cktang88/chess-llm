import pygame

from const import *
from board import Board
from dragger import Dragger
from config import Config
from square import Square
from ai_controller import AIController

class Game:

    def __init__(self):
        self.next_player = 'white'
        self.hovered_sqr = None
        self.board = Board()
        self.dragger = Dragger()
        self.config = Config()
        self.ai_enabled = False
        self.ai_color = 'black'  # AI plays black by default
        self.ai_controller = None
        self.ai_thinking = False

    # blit methods

    def show_bg(self, surface):
        theme = self.config.theme
        
        for row in range(ROWS):
            for col in range(COLS):
                # color
                color = theme.bg.light if (row + col) % 2 == 0 else theme.bg.dark
                # rect
                rect = (col * SQSIZE, row * SQSIZE, SQSIZE, SQSIZE)
                # blit
                pygame.draw.rect(surface, color, rect)

                # row coordinates
                if col == 0:
                    # color
                    color = theme.bg.dark if row % 2 == 0 else theme.bg.light
                    # label
                    lbl = self.config.font.render(str(ROWS-row), 1, color)
                    lbl_pos = (5, 5 + row * SQSIZE)
                    # blit
                    surface.blit(lbl, lbl_pos)

                # col coordinates
                if row == 7:
                    # color
                    color = theme.bg.dark if (row + col) % 2 == 0 else theme.bg.light
                    # label
                    lbl = self.config.font.render(Square.get_alphacol(col), 1, color)
                    lbl_pos = (col * SQSIZE + SQSIZE - 20, HEIGHT - 20)
                    # blit
                    surface.blit(lbl, lbl_pos)

    def show_pieces(self, surface):
        for row in range(ROWS):
            for col in range(COLS):
                # piece ?
                if self.board.squares[row][col].has_piece():
                    piece = self.board.squares[row][col].piece
                    
                    # all pieces except dragger piece
                    if piece is not self.dragger.piece:
                        piece.set_texture(size=80)
                        img = pygame.image.load(piece.texture)
                        img_center = col * SQSIZE + SQSIZE // 2, row * SQSIZE + SQSIZE // 2
                        piece.texture_rect = img.get_rect(center=img_center)
                        surface.blit(img, piece.texture_rect)

    def show_moves(self, surface):
        theme = self.config.theme

        if self.dragger.dragging:
            piece = self.dragger.piece

            # loop all valid moves
            for move in piece.moves:
                # color
                color = theme.moves.light if (move.final.row + move.final.col) % 2 == 0 else theme.moves.dark
                # rect
                rect = (move.final.col * SQSIZE, move.final.row * SQSIZE, SQSIZE, SQSIZE)
                # blit
                pygame.draw.rect(surface, color, rect)

    def show_last_move(self, surface):
        theme = self.config.theme

        if self.board.last_move:
            initial = self.board.last_move.initial
            final = self.board.last_move.final

            for pos in [initial, final]:
                # color
                color = theme.trace.light if (pos.row + pos.col) % 2 == 0 else theme.trace.dark
                # rect
                rect = (pos.col * SQSIZE, pos.row * SQSIZE, SQSIZE, SQSIZE)
                # blit
                pygame.draw.rect(surface, color, rect)

    def show_hover(self, surface):
        if self.hovered_sqr:
            # color
            color = (180, 180, 180)
            # rect
            rect = (self.hovered_sqr.col * SQSIZE, self.hovered_sqr.row * SQSIZE, SQSIZE, SQSIZE)
            # blit
            pygame.draw.rect(surface, color, rect, width=3)

    # other methods

    def next_turn(self):
        self.next_player = 'white' if self.next_player == 'black' else 'black'

    def set_hover(self, row, col):
        self.hovered_sqr = self.board.squares[row][col]

    def change_theme(self):
        self.config.change_theme()

    def play_sound(self, captured=False):
        if captured:
            self.config.capture_sound.play()
        else:
            self.config.move_sound.play()

    def reset(self):
        ai_enabled = self.ai_enabled
        ai_color = self.ai_color
        ai_controller = self.ai_controller
        self.__init__()
        if ai_enabled and ai_controller:
            self.enable_ai(ai_color)
            self.ai_controller = ai_controller
            self.ai_controller.reset()

    def enable_ai(self, color='black', controller_cls=None):
        """Enable AI opponent.

        controller_cls: class matching the AIController interface (get_ai_move,
        add_opponent_move, reset). Defaults to the v1 AIController.
        """
        try:
            self.ai_enabled = True
            self.ai_color = color
            if not self.ai_controller:
                cls = controller_cls or AIController
                self.ai_controller = cls()
            print(f"✓ AI enabled, playing as {color} "
                  f"(controller: {type(self.ai_controller).__name__})")
        except Exception as e:
            print(f"❌ Failed to enable AI: {e}")
            self.ai_enabled = False

    def disable_ai(self):
        """Disable AI opponent."""
        self.ai_enabled = False
        print("✓ AI disabled")

    def is_ai_turn(self):
        """Check if it's AI's turn to play."""
        return self.ai_enabled and self.next_player == self.ai_color

    def get_ai_move(self):
        """Get AI move for current position."""
        if not self.ai_enabled or self.ai_thinking:
            return None

        self.ai_thinking = True

        # Get FEN from current board state
        fen = self.board.to_fen()
        fen = self.board.update_fen_color(fen, self.next_player)

        # Get AI move
        uci_move = self.ai_controller.get_ai_move(fen)

        self.ai_thinking = False

        if uci_move:
            # Convert UCI to Move object
            move = self.board.uci_to_move(uci_move)
            return move
        return None

    def make_ai_move(self):
        """Execute AI move if it's AI's turn."""
        if not self.is_ai_turn():
            return False

        move = self.get_ai_move()
        if move:
            # Get the piece to move
            piece = self.board.squares[move.initial.row][move.initial.col].piece
            if piece:
                # Calculate valid moves for the piece
                self.board.calc_moves(piece, move.initial.row, move.initial.col, bool=True)

                if self.board.valid_move(piece, move):
                    # Check for capture
                    captured = self.board.squares[move.final.row][move.final.col].has_piece()

                    # Make the move
                    self.board.move(piece, move)
                    self.board.set_true_en_passant(piece)

                    # Play sound
                    self.play_sound(captured)

                    # Add opponent move to AI history (for tracking)
                    # This will be done when human plays

                    return True
        return False