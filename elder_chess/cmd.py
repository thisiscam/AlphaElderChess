import argparse, subprocess, os
from .human_player import HumanPlayer
from .mcts_player import MCTSPlayer
from .elder_chess_game_server import ElderChessGameServer
from .elder_chess_native import Board


def mcts_player(name=""):
    from .tensorflow_policy import PolicyValueNet # Tensorflow
    policy_value_net = PolicyValueNet(model_file="models/best_policy.model")
    return MCTSPlayer(policy_value_net.policy_value, c_puct=5, n_playout=10000, is_selfplay=False, name=name)

def mcts_player_at_branch(git_commit, n_playout, name='', tmp_repo_path='./tmp/AEC'):
    build_dir = "build_" + git_commit
    subprocess.check_call(['git', 'fetch', '--all'], cwd=tmp_repo_path)
    subprocess.check_call(['git','checkout', git_commit], cwd=tmp_repo_path)
    subprocess.check_call(['git','submodule', 'update', '--init', '--recursive'], cwd=tmp_repo_path)
    print("Compiling native...")
    subprocess.check_call(['mkdir', '-p', os.path.join('native', build_dir)], cwd=tmp_repo_path)
    subprocess.check_call(['cmake', '..'], cwd=os.path.join(tmp_repo_path,  os.path.join('native', build_dir)))
    subprocess.check_call(['make', '-C',  os.path.join('native', build_dir)], cwd=tmp_repo_path)
    os.environ['LD_LIBRARY_PATH'] = os.path.join(tmp_repo_path, os.path.join('native', build_dir))
    import importlib.util
    spec1 = importlib.util.spec_from_file_location(git_commit + "_module", os.path.join(tmp_repo_path, 'mcts_player.py'))
    player_module = importlib.util.module_from_spec(spec1)
    spec1.loader.exec_module(player_module)

    spec2 = importlib.util.spec_from_file_location(git_commit + "_module", os.path.join(tmp_repo_path, 'tensorflow_policy.py'))
    network_module = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(network_module)

    value_net = network_module.PolicyValueNet(model_file=os.path.join("models_"+git_commit, 'best_policy.model'))
    return player_module.MCTSPlayer(value_net.policy_value, c_puct=5, n_playout=n_playout, is_selfplay=False, name=name)

def player_argument(args):
    print(args)
    player_dict = {
        "HumanPlayer": HumanPlayer,
        "MCTSPlayer": mcts_player
    }
    player_str = args[0]
    if player_str not in player_dict:
        return mcts_player_at_branch(player_str, int(args[1]), name=args[0])
    else:
        return player_dict[player_str](name=args[0])

parser = argparse.ArgumentParser(description='command line game')
parser.add_argument('--p1', type=str, default="HumanPlayer", help='player 1')
parser.add_argument('--p2', type=str, default="HumanPlayer", help='player 2')
parser.add_argument('--p1-n-playouts', type=int, default=1000, help='player 1 num playouts')
parser.add_argument('--p2-n-playouts', type=int, default=1000, help='player 2 num playouts')
parser.add_argument('--num-plays', type=int, default=1, help='number of plays')

args = parser.parse_args()

args.p1 = player_argument((args.p1, args.p1_n_playouts))
args.p2 = player_argument((args.p2, args.p2_n_playouts))

game_server = ElderChessGameServer()
win_cnt = {None: 0, args.p1: 0, args.p2: 0}
for i in range(args.num_plays):
    if i % 2 == 0:
        p1, p2 = args.p1, args.p2
    else:
        p1, p2 = args.p2, args.p1
    winner = game_server.start_play(p1, p2, is_shown=True)
    win_cnt[winner] += 1
win_ratio = 1.0*(win_cnt[args.p1] + 0.5*win_cnt[None]) / args.num_plays
print("win: {}, lose: {}, tie:{}".format(win_cnt[args.p1], win_cnt[args.p2], win_cnt[None]))
