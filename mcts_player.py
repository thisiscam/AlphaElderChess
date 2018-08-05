from elder_chess_native import MCTS, move_probs_to_one_hot
import numpy as np

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=2000, is_selfplay=False, name="", num_parallel_workers=1, parallel_mcts_eval_batch_size=8):
        self.mcts = MCTS(policy_value_function, float(c_puct), n_playout, num_parallel_workers, parallel_mcts_eval_batch_size)
        self._is_selfplay = is_selfplay
        self.name = name
        self.steps = 0

    def reset_player(self):
        self.steps = 0
        self.mcts.reset()

    def other_do_move(self, nextBoard, move):
        self.mcts.update_with_move(nextBoard, move)
        if not move.by_env():
            self.steps += 1

    def counts_to_move_probs(self, counts, temp=1.):
        return softmax(1.0/temp * np.log(np.array(counts) + 1e-10))

    def get_action(self, board, temp=1., return_prob=False, verbose=False):
        # the pi vector returned by MCTS as in the alphaGo Zero paper

        if len(board.get_moves()) > 0:
            temp = 1e-3 if self.steps > 14 or not self._is_selfplay else 1.
            acts, counts = self.mcts.get_move_counts(board)
            probs = self.counts_to_move_probs(counts, temp)

            if verbose:
                print(counts)
            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for
                # self-play training)
                move_index = np.random.choice(
                    len(acts),
                    p=0.75*probs + 0.25*np.random.dirichlet(0.03 * np.ones(len(probs)))
                )
                # update the root node and reuse the search tree
                self.mcts.update_with_move_index(board, move_index)
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move_index = np.random.choice(len(acts), p=probs)
                # reset the root node
                self.mcts.update_with_move_index(board, move_index)
#                location = board.move_to_location(move)
#                print("AI move: %d,%d\n" % (location[0], location[1]))
            self.steps += 1
            move = acts[move_index]
            if return_prob:
                probs_1_h = move_probs_to_one_hot(acts, probs)
                return move, probs_1_h
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTSPlayer" + self.name

