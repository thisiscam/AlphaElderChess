from .elder_chess_native import Board
import numpy as np

class ElderChessGameServer(object):
    """game server"""

    def __init__(self, board=Board()):
        self.initial_board = board
        self.init_board()

    def init_board(self):
        self.board = Board(self.initial_board)

    def reset_player(self, *players):
        for player in players:
            player.reset_player()

    def graphic(self, board):
        """Draw the board and show game info"""
        print(board)

    def env_move(self, *players):
        if self.board.is_env_move():
            env_move = self.board.env_do_move()
            for player in players:
                player.other_do_move(self.board, env_move)

    def start_play(self, player1, player2, is_shown=True):
        """start a game between two players"""
        self.init_board()
        self.reset_player(player1, player2)
        players = {0: player1, 1: player2}
        if is_shown:
            self.graphic(self.board)

        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)

            if self.board.do_move_safe(move):
                players[1 - current_player].other_do_move(self.board, move)
                self.env_move(player1, player2)
            else:
                print("{} attemped a valid move".format(player_in_turn))
                continue

            if is_shown:
                self.graphic(self.board)
            if self.board.game_ended():
                if is_shown:
                    winner = self.board.get_winner()
                    if winner == 2:
                        winner = -1
                    if winner >= 0:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return players[winner] if winner >= 0 else None

    def start_self_play(self, player, is_shown=False):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.init_board()
        self.reset_player(player)
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board, return_prob=True)
            # store the data
            states.append(self.board.get_compact_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.get_current_player())
            # perform a move
            if self.board.do_move_safe(move):
                self.env_move(player)
            else:
                raise Exception("self play failed to make valid move")
            if is_shown:
                self.graphic(self.board)
            if self.board.game_ended():
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                winner = self.board.get_winner()
                if winner == 2:
                    winner = -1
                if winner >= 0:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner >= 0:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return player if winner >= 0 else None, zip(states, mcts_probs, winners_z)
