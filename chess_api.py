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
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value, c_puct=5, n_playout=10000, is_selfplay=False)

    def start_game(self):
        self.board = Board()
        print(self.board)
        self.mcts_player.reset_player()

    def make_move(self, move_str):
        try:
            move = try_parse(move_str)
            print(move)
        except:
            return False
        if self.board.do_move_safe(move):
            print(self.board)
            self.mcts_player.other_do_move(self.board, move)
            print("other done")
            if self.board.is_env_move():
                print("env start")
                env_move = self.board.env_do_move()
                print("env done")
                self.mcts_player.other_do_move(self.board, env_move)
                print("mcts done")
            return True
        else:
            return False

    def ai_make_move(self):
        move = self.mcts_player.get_action(self.board)
        if self.board.do_move_safe(move):
            if self.board.is_env_move():
                env_move = self.board.env_do_move()
                self.mcts_player.other_do_move(self.board, env_move)
        return str(move)

    def display_board(self):
        return str(self.board)

    def get_winner(self):
        return self.board.get_winner()

    def game_ended(self):
        return self.board.game_ended()

obj = MyObject()
server = SimpleXMLRPCServer(("127.0.0.1", 1027), allow_none=True)
server.register_instance(obj)

print("Listening on port 1027")
server.serve_forever()

