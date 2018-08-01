from elder_chess_native import Board, Move
from mcts_player import MCTSPlayer
from elder_chess_game_server import ElderChessGameServer
from tensorflow_policy import PolicyValueNet

from xmlrpc.server import SimpleXMLRPCServer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="models/best_policy.model", help='model path')
args = parser.parse_args()

FLIP = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4

def try_parse(cmd):
    words = cmd.split()
    print(words)
    if words[0][0] in ("f", "F"):
        return Move(FLIP, int(words[1]), int(words[2]))
    elif words[0][0] in ("m", "M"):
        if words[3][0] in ("u", "U"):
            return Move(UP, int(words[1]), int(words[2]))
        if words[3][0] in ("d", "D"):
            return Move(DOWN, int(words[1]), int(words[2]))
        if words[3][0] in ("l", "L"):
            return Move(LEFT, int(words[1]), int(words[2]))
        if words[3][0] in ("r", "R"):
            return Move(RIGHT, int(words[1]), int(words[2]))
    else:
        raise Exception()

class MyObject:

    def __init__(self):
        self.policy_value_net = PolicyValueNet(4, 4, model_file=args.model)
        self.boards = {}
        self.mcts_players = {}

    def start_game(self, id, n_playout=10000):
        self.boards[id] = Board()
        if id in self.mcts_players:
            self.mcts_players[id].reset_player()
        else:
            self.mcts_players[id] = MCTSPlayer(self.policy_value_net.policy_value, c_puct=5, n_playout=n_playout, is_selfplay=False)

    def _get_game(self, id):
        if id not in self.boards or id not in self.mcts_players:
            raise Exception("id not found")
        board = self.boards[id]
        mcts_player = self.mcts_players[id]
        return board, mcts_player

    def check_game_started(self, id):
        try:
            self._get_game(id)
            return True
        except:
            print(id, "game not started")
            return False

    def make_move(self, id, move_str):
        board, mcts_player = self._get_game(id)
        try:
            move = try_parse(move_str)
            print(move)
        except:
            return False
        if board.do_move_safe(move):
            print(board)
            mcts_player.other_do_move(board, move)
            if board.is_env_move():
                env_move = board.env_do_move()
                mcts_player.other_do_move(board, env_move)
            return True
        else:
            return False

    def ai_make_move(self, id):
        board, mcts_player = self._get_game(id)
        move = mcts_player.get_action(board)
        if board.do_move_safe(move):
            if board.is_env_move():
                env_move = board.env_do_move()
                mcts_player.other_do_move(board, env_move)
        return str(move)

    def display_board(self, id):
        board, _ = self._get_game(id)
        return str(board)

    def get_winner(self, id):
        board, _ = self._get_game(id)
        return board.get_winner()

    def game_ended(self, id):
        board, _ = self._get_game(id)
        return board.game_ended()

obj = MyObject()
server = SimpleXMLRPCServer(("127.0.0.1", 1027), allow_none=True)
server.register_instance(obj)

print("Listening on port 1027")
server.serve_forever()

