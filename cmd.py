import argparse
from human_player import HumanPlayer
from mcts_player import MCTSPlayer
from elder_chess_game_server import ElderChessGameServer
from tensorflow_policy import PolicyValueNet # Tensorflow
from elder_chess_native import Board

parser = argparse.ArgumentParser(description='command line game')
parser.add_argument('--p1', type=str, default="HumanPlayer", help='player 1')
parser.add_argument('--p2', type=str, default="HumanPlayer", help='player 2')
args = parser.parse_args()

def mcts_player():
    policy_value_net = PolicyValueNet(4, 4, model_file="models/best_policy.model")
    return MCTSPlayer(policy_value_net.policy_value, c_puct=5, n_playout=100000, is_selfplay=False)

def mcts_player2():
    policy_value_net = PolicyValueNet(4, 4, model_file="models/best_policy.model")
    return MCTSPlayer(policy_value_net.policy_value, c_puct=5, n_playout=10000, is_selfplay=False)

player_dict = {
    "HumanPlayer": HumanPlayer,
    "MCTSPlayer": mcts_player
}

game_server = ElderChessGameServer()
p1, p2 = player_dict[args.p1](), player_dict[args.p2]()
game_server.start_play(p1, p2, is_shown=True)
