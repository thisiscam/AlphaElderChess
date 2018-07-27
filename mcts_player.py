from elder_chess_native import MCTS, move_probs_to_one_hot
import numpy as np

class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=2000, is_selfplay=False, name=""):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        self.name = name

    def reset_player(self):
        self.mcts.reset()

    def other_do_move(self, nextBoard, move):
        self.mcts.update_with_move(nextBoard, move)

    def get_action(self, board, temp=1., return_prob=False):
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(5 * 4 * 4)
        if len(board.get_moves()) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            probs = np.array(probs)
            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for
                # self-play training)
                move_index = np.random.choice(
                    len(acts),
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                # update the root node and reuse the search tree
                self.mcts.update_with_move_index(board, move_index)
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                print(probs)
                move_index = np.random.choice(len(acts), p=probs)
                # reset the root node
                self.mcts.update_with_move_index(board, move_index)
#                location = board.move_to_location(move)
#                print("AI move: %d,%d\n" % (location[0], location[1]))
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

