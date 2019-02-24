import numpy as np
from .elder_chess_native import Move

class NNPlayer(object):
    """AI player based on NN"""

    def __init__(self, policy_value_function, is_selfplay=False, name=""):
        self.policy_value_fn = policy_value_function
        self.name = name
        self.is_selfplay = is_selfplay

    def reset_player(self):
        pass

    def other_do_move(self, nextBoard, move):
        pass

    def get_action(self, board, return_prob=False):
        ret = self.get_action_batch([board], [board.get_compact_state()], return_prob)
        return ret[0]

    def get_action_batch(self, boards, state_batch, return_prob=False):
        state_batch = (
                [data[0] for data in state_batch], 
                [data[1] for data in state_batch], 
                [(data[2],) for data in state_batch]
        )
        probs, values = self.policy_value_fn(state_batch)
        legal_move_masks = np.array([b.get_moves_one_hot().ravel() for b in boards])
        legal_probs = probs * legal_move_masks
        legal_probs /= legal_probs.sum(axis=1,keepdims=1)

        moves_encoded = []
        for row in legal_probs:
            moves_encoded.append(np.random.choice(legal_probs.shape[1], p=row))

        moves = []
        for i, me in enumerate(moves_encoded):
            t, x, y = me // 16, (me % 16) // 4, me % 4
            m = Move(t, x, y)
            moves.append(m)
        return moves

    def __str__(self):
        return "NNPlayer" + self.name

