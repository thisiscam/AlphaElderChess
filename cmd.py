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
    return MCTSPlayer(policy_value_net.policy_value, c_puct=5, n_playout=10000, is_selfplay=False)

def mcts_player2():
    policy_value_net = PolicyValueNet(4, 4, model_file="models/best_policy.model")
    return MCTSPlayer(policy_value_net.policy_value, c_puct=5, n_playout=10000, is_selfplay=False)

player_dict = {
    "HumanPlayer": HumanPlayer,
    "MCTSPlayer": mcts_player
}

def mcts_player_at_branch(commit_tag, tmp_repo_path='./tmp/AEC'):
    subprocess.check_call(['git','checkout', commit_tag], cwd=tmp_repo_path)
    subprocess.check_call(['git','submodule', 'update', '--init', '--recursive'], cwd=tmp_repo_path)
    print("Compiling native...")
    subprocess.check_call(['mkdir', '-p', 'native/build'], cwd=tmp_repo_path)
    subprocess.check_call(['cmake', '..'], cwd=os.path.join(tmp_repo_path, "native/build/"))
    subprocess.check_call(['make', '-C', 'native/build'], cwd=tmp_repo_path)
    
mcts_player_at_branch('0334e0e')
# game_server = ElderChessGameServer()
# p1, p2 = player_dict[args.p1](), player_dict[args.p2]()
# game_server.start_play(p1, p2, is_shown=True)
