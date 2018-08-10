from .elder_chess_native import Move

class HumanPlayer(object):
    """
    human player
    """

    def __init__(self):
        pass

    def reset_player(self):
        pass

    def other_do_move(self, nextBoard, move):
        pass

    def get_action(self, board, temp=None):
        try:
            s = input("[{}]Your move: ".format(self))
            move = Move(s)
        except Exception as e:
            print(e)
            move = None
        if move not in board.get_moves():
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "HumanPlayer"
