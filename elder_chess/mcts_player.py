from .elder_chess_native import MCTS, BatchMCTS, move_probs_to_one_hot
import numpy as np

class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policy_value_function,
                 c_puct=5, 
                 n_playout=2000, 
                 is_selfplay=False, 
                 name="",
                 num_parallel_workers=4,
                 parallel_mcts_eval_batch_size=256
        ):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self.batch_mcts = BatchMCTS(policy_value_function, float(c_puct), n_playout, num_parallel_workers, parallel_mcts_eval_batch_size)
        self._is_selfplay = is_selfplay
        self.name = name

    def reset_player(self):
        self.mcts.reset()

    def other_do_move(self, nextBoard, move):
        self.mcts.update_with_move(nextBoard, move)

    def get_action(self, board, return_prob=False):
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        if len(board.get_moves()) > 0:
            small_temp = board.get_total_steps() > 10
            acts, probs = self.mcts.get_move_probs(board, small_temp)
            return self.sample_move(acts, probs, small_temp=small_temp, update_mcts=True, return_prob=return_prob, board=board)
        else:
            print("WARNING: the board is full")

    def get_action_batch(self, boards, return_prob=False):
        self.batch_mcts.reset()
        small_temps = [b.get_total_steps() > 10 for b in boards]
        move_probs = self.batch_mcts.get_move_probs(boards, small_temps)
        ret = []
        for (moves, probs), small_temp in zip(move_probs, small_temps):
            ret.append(self.sample_move(moves, probs, small_temp=small_temp, update_mcts=False, return_prob=return_prob))
        return ret

    def sample_move(self, moves, probs, small_temp=False, update_mcts=False, return_prob=False, board=None):
        probs = np.array(probs)
        if self._is_selfplay and small_temp:
            # add Dirichlet Noise for exploration (needed for
            # self-play training)
            move_index = np.random.choice(
                len(moves),
                p=0.75*probs + 0.25*np.random.dirichlet(0.03*np.ones(len(probs)))
            )
        else:
            # with the default temp=1e-3, it is almost equivalent
            # to choosing the move with the highest prob
            if not self._is_selfplay:
                print(probs, small_temp)
            move_index = np.random.choice(len(moves), p=probs)
            # reset the root node
        if update_mcts:
            self.mcts.update_with_move_index(board, move_index)

        move = moves[move_index]
        if return_prob:
            probs_1_h = move_probs_to_one_hot(moves, probs)
            return move, probs_1_h
        else:
            return move
        
    def __str__(self):
        return "MCTSPlayer" + self.name

