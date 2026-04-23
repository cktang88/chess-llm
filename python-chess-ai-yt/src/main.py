import pygame
import sys

from const import *
from game import Game
from square import Square
from move import Move
from controller_select import run_select


class Main:

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Chess")
        self.game = Game()

    def mainloop(self):

        screen = self.screen
        game = self.game
        board = self.game.board
        dragger = self.game.dragger

        # Pre-game setup screen — pick controller + color via pygame UI.
        sel = run_select(screen)
        if sel.enable_ai:
            ai_color = "white" if sel.human_color == "black" else "black"
            game.enable_ai(ai_color, controller_cls=sel.controller_cls)

        while True:
            # Check if it's AI's turn and make AI move
            if game.is_ai_turn() and not game.ai_thinking:
                # Show board before AI thinks
                game.show_bg(screen)
                game.show_last_move(screen)
                game.show_pieces(screen)
                pygame.display.update()

                # Make AI move
                if game.make_ai_move():
                    game.next_turn()
                else:
                    print("AI failed to make a move")
                    game.disable_ai()
            # show methods
            game.show_bg(screen)
            game.show_last_move(screen)
            game.show_moves(screen)
            game.show_pieces(screen)
            game.show_hover(screen)

            if dragger.dragging:
                dragger.update_blit(screen)

            for event in pygame.event.get():

                # click
                if event.type == pygame.MOUSEBUTTONDOWN:
                    dragger.update_mouse(event.pos)

                    clicked_row = dragger.mouseY // SQSIZE
                    clicked_col = dragger.mouseX // SQSIZE

                    # if clicked square has a piece ?
                    if board.squares[clicked_row][clicked_col].has_piece():
                        piece = board.squares[clicked_row][clicked_col].piece
                        # valid piece (color) ?
                        if piece.color == game.next_player:
                            board.calc_moves(piece, clicked_row, clicked_col, bool=True)
                            dragger.save_initial(event.pos)
                            dragger.drag_piece(piece)
                            # show methods
                            game.show_bg(screen)
                            game.show_last_move(screen)
                            game.show_moves(screen)
                            game.show_pieces(screen)

                # mouse motion
                elif event.type == pygame.MOUSEMOTION:
                    motion_row = event.pos[1] // SQSIZE
                    motion_col = event.pos[0] // SQSIZE

                    game.set_hover(motion_row, motion_col)

                    if dragger.dragging:
                        dragger.update_mouse(event.pos)
                        # show methods
                        game.show_bg(screen)
                        game.show_last_move(screen)
                        game.show_moves(screen)
                        game.show_pieces(screen)
                        game.show_hover(screen)
                        dragger.update_blit(screen)

                # click release
                elif event.type == pygame.MOUSEBUTTONUP:

                    if dragger.dragging:
                        dragger.update_mouse(event.pos)

                        released_row = dragger.mouseY // SQSIZE
                        released_col = dragger.mouseX // SQSIZE

                        # create possible move
                        initial = Square(dragger.initial_row, dragger.initial_col)
                        final = Square(released_row, released_col)
                        move = Move(initial, final)

                        # valid move ?
                        if board.valid_move(dragger.piece, move):
                            # normal capture
                            captured = board.squares[released_row][
                                released_col
                            ].has_piece()
                            board.move(dragger.piece, move)

                            board.set_true_en_passant(dragger.piece)

                            # Track human move for AI
                            if game.ai_enabled and game.ai_controller:
                                # Convert move to UCI
                                src_algebraic = Square.get_alphacol(
                                    move.initial.col
                                ) + str(8 - move.initial.row)
                                dst_algebraic = Square.get_alphacol(
                                    move.final.col
                                ) + str(8 - move.final.row)
                                uci = src_algebraic + dst_algebraic
                                game.ai_controller.add_opponent_move(uci)

                            # sounds
                            game.play_sound(captured)
                            # show methods
                            game.show_bg(screen)
                            game.show_last_move(screen)
                            game.show_pieces(screen)
                            # next turn
                            game.next_turn()

                    dragger.undrag_piece()

                # key press
                elif event.type == pygame.KEYDOWN:

                    # changing themes
                    if event.key == pygame.K_t:
                        game.change_theme()

                    # reset game
                    if event.key == pygame.K_r:
                        game.reset()
                        game = self.game
                        board = self.game.board
                        dragger = self.game.dragger

                    # toggle AI (enable/disable)
                    if event.key == pygame.K_a:
                        if game.ai_enabled:
                            game.disable_ai()
                        else:
                            # Enable AI for opposite color
                            ai_color = (
                                "black" if game.next_player == "white" else "white"
                            )
                            game.enable_ai(ai_color)

                # quit application
                elif event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            pygame.display.update()


main = Main()
main.mainloop()
